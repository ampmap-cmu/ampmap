import socket
import sys
import traceback
import subprocess
from threading import Thread
import json 
import pickle
import argparse
import libs.definition as df
import libs.inputs as inputs
import copy, os
import libs.search_common as search_common
import libs.blackbox as bb
import struct
import configparser
import libs.query_json as query_json
from collections import OrderedDict


'''
helper functions
'''
def get_fields(params, url=None):
	proto_fields =  inputs.generate_proto_fields(params["in_dir"], url)
	return proto_fields


def get_params(config): 
	f = open(config["common_path"]["common_config_file"], 'r')
	params = json.load(f)
	f.close()
	return params

def get_phase(config, node_ip, server_ip): 
	f = open(os.path.join(config["common_path"]["command_out_dir"], node_ip+"/"+server_ip), 'r')
	phase = f.readlines()[0].strip()
	f.close()
	return phase 

def convert_to_starting_queries( pfs_inst ):
	starting_queries = [] 
	for inst in pfs_inst:
		if "threshold" in inst: 
			continue 
		entry = OrderedDict()
		print("inst ", inst)
		entry["amp_factor"] = float(inst["amp_factor"])
		entry["server_ip"] = inst["server_ip"]
		entry["fields"] =  inst["fields"]
		starting_queries.append(entry)
	return starting_queries


'''
For each (measurer, server), master.py will run job.py as a separate process based on the current stage
'''
def main():
	# read common_path.ini
	print("before reading ")
	config = configparser.ConfigParser()
	config.read("common_path.ini")
	print("read config ")

	# read params from .ini file
	params = get_params(config)
	print("params" , params)

	pcap_paths = ["/ampmap/pcap", "/ampmap/pcap_temp"]
	for p in pcap_paths:
		if not os.path.exists(p):
			os.makedirs(p)


	url=None
	if params["url"]: 
		url = params["url"]
	fields = get_fields(params, url) 


	# read phase 
	node_ip = sys.argv[1]
	server_ip = sys.argv[2]
	phase = get_phase(config, node_ip, server_ip)

	BB = bb.blackbox(params[df.TIMEOUT])
	BB.register_protocol(params[df.PROTO])
	
	#	Random Stage 
	if phase == "RANDOM STARTED":
		print("RANDOM Stage Started")

		BB.register_phase("random")
		params["server_ip"] = server_ip
		search_common.generate_random_queries(BB, fields, node_ip, params)
		
		print("\n\nRANDOM Stage Ended")
		f = open(os.path.join(config["common_path"]["command_out_dir"], node_ip+"/"+server_ip), 'w')
		
		f.write("RANDOM ENDED\n")
		f.close()

	#	Probe Stage 
	elif phase == "PROBE STARTED": 
		print("Probe Stage Started")
		BB.register_phase("probe")
		params["server_ip"] = server_ip
		f = os.path.join(config["common_path"]["command_out_dir"], node_ip+"/"+server_ip+".probe")
		probing_queries = query_json.read_json(f) 
		
		# Send probing queries for a given server 
		search_common.send_probing_queries(BB, fields, node_ip, probing_queries, params) 
		print("\n\nPROBE ENDED")

		f = open(os.path.join(config["common_path"]["command_out_dir"], node_ip+"/"+server_ip), 'w')
		f.write("PROBE ENDED\n")
		f.close()

	#	Per-Field Search (PFS) Stage
	elif phase == "PFS STARTED": 
		print("PFS STARTED")

		BB.register_phase("pfs")
		params["server_ip"] = server_ip
		params["node_ip"] = node_ip
		
		f = os.path.join(config["common_path"]["command_out_dir"], node_ip+"/"+server_ip+".pfs")

		# read the PFS instruction (starting point and the AF threshold)
		pfs_inst = query_json.read_json(f) 
		AF_THRESH = float(pfs_inst[0]["threshold"])
		starting_queries = convert_to_starting_queries( pfs_inst )
		print(pfs_inst, starting_queries)

		# run the PFS search
		search_common.per_field_search(BB, fields, server_ip, starting_queries,  AF_THRESH,  params, None,  None  )
		
		print("\n\nPFS ENDED")

		f = open(os.path.join(config["common_path"]["command_out_dir"], node_ip+"/"+server_ip), 'w')
		f.write("PFS ENDED\n")
		f.close()

	elif phase == "SIMULATED STARTED":
		print("SIMULATED STARTED")

		BB.register_phase("simulated")

		params["server_ip"] = server_ip
		search_strawman.generate_simulated_queries(BB, fields, node_ip, params)


		print("\n\nRANDOM ENDED")
		f = open(os.path.join(config["common_path"]["command_out_dir"], node_ip+"/"+server_ip), 'w')
		f.write("RANDOM ENDED\n")
		f.close()

	else:
		print("Invalid state!")

  
if __name__== "__main__":
	main()