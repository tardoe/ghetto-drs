#!/usr/bin/python

# Stopping Criteria
# if cluster score < worst / 5 and last move caused an increase in cluster score
# perform (num_hosts) / 2 extra moves

# Due to insecure SSL warnings in embedded urllib3 inside requests.
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

from pyproxmox.pyproxmox import *
from pprint import pprint as pp
import numpy
from operator import itemgetter
import time, math, logging
import concurrent.futures
import argparse

# Weights used for calculating statistical scores
# for balancing load. Default to 50/50.
CPU_WEIGHT = 0.50
MEM_WEIGHT = 0.50

class ProxmoxBalancer:

	def __init__(self):
		self.proxmox_client = None
		self.node_stats = {}
		self.nodes = []
		self.cpu = {}
		self.memory = {}
		self.num_nodes = 0
		self.cluster_score = 0

	def authenticate(self, host, username, password):
		logging.info("Connecting to %s with username %s...", host, username)
		self.proxmox_client =  pyproxmox(prox_auth(host, username, password))

	# helper method
	def calculate_stats(self, metrics):
		return { "mean" : numpy.mean(metrics, axis=0), "std" : numpy.std(metrics, axis=0) }


	def get_node_stats(self, node):
		data = self.proxmox_client.getNodeStatus(node)["data"]

		cpu_load = data["loadavg"][0]
		memory = float(data["memory"]["used"]) / float(data["memory"]["total"])

		return (cpu_load, memory)


	def calculate_cluster_score(self):
		logging.info("Calculating cluster score...")

		# Grab the names of hosts - the API puts both nodes and clusters into this one call... dumb!
		self.nodes = [ x['name'] for x in self.proxmox_client.getClusterStatus()['data'] if x['type'] == 'node']

		# grab the CPU Load (15m) and Memory used percentage for each host.
		self.node_stats = {}
		
		with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.nodes)) as executor:
			future_node_stats = {executor.submit(self.get_node_stats, node): node for node in self.nodes}

			results = {}

			for future in concurrent.futures.as_completed(future_node_stats):
				node = future_node_stats[future]

				try:
					result = future.result()

					results[node] = { 'cpu_load' : result[0] }
					results[node]['mem_percent'] = result[1]
				except Exception as exc:
					print('%r generated an exception: %s' % (node, exc))

			self.node_stats = results

		# calculate mean and standard deviation of the hosts in the cluster
		self.cpu = self.calculate_stats(numpy.array([ float(x["cpu_load"]) for x in self.node_stats.values()]))
		self.memory = self.calculate_stats(numpy.array( [ float(x["mem_percent"]) for x in self.node_stats.values()]))

		# calculate a weighted, normalised score for the 'balance' of the cluster.
		# we will try and minimise this by migrating VMs.
		# This score is calculated by the standard dev. as a percentage of the mean
		# combined and weighted for the two datasets (15m CPU Load Avg and Memory Used %)
		current_balance = (float(self.cpu["std"]) / float(self.cpu["mean"])*CPU_WEIGHT + float(self.memory["std"]) / float(self.memory["mean"])*MEM_WEIGHT)

		logging.info("Current cluster score: %f", current_balance)
		self.cluster_score = current_balance
		
                return current_balance



	# Calculate a Z-Score (normalisation) for each host and choose the worst to migrate a VM from.
	def calculate_node_scores(self):
		# Z-Score = value - mean(cpu) / stddev
		node_scores = {}

		for key,value in self.node_stats.iteritems():
			node_scores[key] = ((float(value["cpu_load"]) - float(self.cpu["mean"])) / float(self.cpu["std"]))*CPU_WEIGHT + ((float(value["mem_percent"]) - float(self.memory["mean"])) / float(self.memory["std"]))*MEM_WEIGHT 

		max_node = max(node_scores, key=node_scores.get)
		min_node = min(node_scores, key=node_scores.get)

		return { "scores" : node_scores, "max" : max_node, "min" : min_node}

	def get_vm_stats(self, node, vmid):
		vm_stats = self.proxmox_client.getVirtualStatus(node, vmid)["data"]
		cpu = vm_stats["cpu"]
		mem = float(vm_stats["mem"]) / float(vm_stats["maxmem"])

		return (cpu, mem)

	def vms_to_migrate(self, candidate_vms):
		# Worst possible score =  sqrt(num_nodes-1)
		# when score > worst / 3, move 3 VMs at a time
		# when score > worst / 5, move 2 VMs at a time
		# else, move 1 VM.

		worst_score = math.sqrt(len(self.nodes) - 1)
		num = 1 # backstop

		if self.cluster_score > (worst_score / 3.0):
			num =  3
		else:
			if self.cluster_score > (worst_score / 5.0):
				num =  2
			else:
				num =  1

		# return that number either side of the median, prefering the lower end.
		median_vm_index = len(candidate_vms) / 2

		# return just one VM.
		if num == 1:
			return [candidate_vms[median_vm_index]] # pick the median (ish) VM.
		# return two VM ids.
		if num == 2:
			#incase there is only two VMs left.
			if len(candidate_vms) <= 2:
				return candidate_vms[median_vm_index]
			else:
				return [candidate_vms[median_vm_index-1], candidate_vms[median_vm_index]]
		if num == 3:
			if len(candidate_vms) < 3:
				return candidate_vms[median_vm_index]
			else:
				return [candidate_vms[median_vm_index-1], candidate_vms[median_vm_index], candidate_vms[median_vm_index+1]]


	def select_vms_to_migrate(self, node, cluster_score):
		logging.info("Choosing VM to move...")

		# Get the list of VM IDs on this host
		vms = self.proxmox_client.getNodeVirtualIndex(node)
		vm_ids = [x["vmid"] for x in vms["data"] if x["status"] == "running"]

		with concurrent.futures.ThreadPoolExecutor(max_workers=len(vm_ids)) as executor:
			future_vm_stats = {executor.submit(self.get_vm_stats, node, vmid): vmid for vmid in vm_ids}

			vm_stats = {}

			for future in concurrent.futures.as_completed(future_vm_stats):
				vmid = future_vm_stats[future]

				try:
					result = future.result()

					vm_stats[vmid] = { 'cpu_percent' : result[0] }
					vm_stats[vmid]['mem_percent'] = result[1]
				except Exception as exc:
					print('%r / %s generated an exception: %s' % (node, vmid, exc))

		# Caluclate Mean and Std of CPU and Memory
		vm_cpu = self.calculate_stats(numpy.array([ float(x["cpu_percent"]) for x in vm_stats.values()]))
		vm_mem = self.calculate_stats(numpy.array([ float(x["mem_percent"]) for x in vm_stats.values()]))

		# Calculate Z-scores for each VM and return the median VM ID.
		zscores = {}
		for key,value in vm_stats.iteritems():
			zscores[key] = float(value["cpu_percent"]) - float(vm_cpu["mean"]) / float(vm_cpu["std"])*CPU_WEIGHT + float(value["mem_percent"]) - float(vm_mem["mean"]) / float(vm_mem["std"])*MEM_WEIGHT 
	
		# sort the vms by zscore
		sorted_vms = sorted(zscores, key=lambda x: zscores[x])

		return self.vms_to_migrate(sorted_vms)


	def migrate_vm(self, vmid, from_node, to_node):
		logging.info("Moving VM: %s from: %s to: %s", str(vmid), from_node, to_node)

		# migrate the machine and return the job ID
		migrate_data = self.proxmox_client.migrateVirtualMachine(from_node,vmid,to_node, online=True)

		# make sure the migration began alright
		if migrate_data['status']['ok']:
			# check if the migration is done.
			while True:
				# check the migraton job
				migrate_status = self.proxmox_client.getNodeTaskStatusByUPID(from_node, migrate_data['data'])
				# make sure it's not stopped.
				if migrate_status['data']['status'] == "stopped":
					break
				else:
					time.sleep(1)
			return True
		else:
			logging.warning("Migration failed: " + migrate_data['status']['reason'])
			return False

	def migrate_vms(self, vms, from_node, to_node):
		with concurrent.futures.ThreadPoolExecutor(max_workers=len(vms)) as executor:
			future_vm_migrations = {executor.submit(self.migrate_vm, vmid, from_node, to_node): vmid for vmid in vms}

			results = []

			for future in concurrent.futures.as_completed(future_vm_migrations):
				try:
					result = future.result()
					results.append(result)

				except Exception as exc:
					print('VM Failed to move! %s', exc)

			for r in results:
				if not r:
					return False
		
		return True

        # Parent function to control Ghetto DRS
	def balance_cluster(self):

		for i in range(0,100):
			# get the current cluster balance score (to minimise)
			cluster_score = self.calculate_cluster_score()

			# get the scores for each node to find the most loaded node.
			node_scores = self.calculate_node_scores()

			# find the median VM on the most loaded node (to move)
			vms_to_move = self.select_vms_to_migrate(node_scores["max"], cluster_score)

			# move the VM on the most loaded node
			move_result = self.migrate_vms(vms_to_move, node_scores["max"], node_scores["min"])

			if move_result:
				logging.info("Successfully moved VMs: %s from: %s to: %s", str(vms_to_move), node_scores["max"], node_scores["min"])
			else:
				logging.warning("Failed to move VMs: %s from : %s to: %s", str(vms_to_move), node_scores["max"], node_scores["min"])

			new_cluster_score = self.calculate_cluster_score()

			logging.info("Cluster score delta: %s", str(new_cluster_score - cluster_score))

			log = open('./balancer.log', 'a')
			log.write(str(vms_to_move) + "," + node_scores["max"] + "," + node_scores["min"] + "," + str(cluster_score) + "," + str(new_cluster_score) + "\n")
			log.close()

	# DRS Placement ghetto haxxx - see, not hard!
	def getNextNodePlacement(self):
		self.calculate_cluster_score()
		node_scores = self.calculate_node_scores()
		return node_scores["min"]


if __name__ == "__main__":	

	parser = argparse.ArgumentParser(description='Ghetto DRS - Warning, I really mean Ghetto... YMMV.')
	parser.add_argument('-u', metavar='proxmox_user', dest='proxmox_user',  help="Username for the proxmox host including domain e.g. root@pam.")
	parser.add_argument('-p', metavar='proxmox_password', dest='proxmox_password',  help="Password for the proxmox host.")
	parser.add_argument('-H', metavar='proxmox_host', dest='proxmox_host',  help="Proxmox Host DNS name or IP address.")

	args = parser.parse_args()

	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
	balancer = ProxmoxBalancer()
	balancer.authenticate(args.proxmox_host, args.proxmox_user, args.proxmox_password)
	balancer.balance_cluster()
