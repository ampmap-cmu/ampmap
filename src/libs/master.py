import copy, os, sys
from collections import OrderedDict
import shutil
import subprocess
import json
import threading
import configparser
from queue import PriorityQueue
from copy import deepcopy 
import math
import logging
import pandas as pd
import datetime
import platform
import time
import configparser

import libs.inputs as inputs
import libs.helper as helper
import libs.query_json as query_json

import libs.definition as df
import libs.search_common as search_common
import libs.cluster as cluster


# assign servers to measurers by Round-robin
def assign_servers_to_node(servers, measurers):
    PORT_NUM = 7002
    server_to_measurer = []
    index = 0 
    while index < len(servers):
        entry = OrderedDict()  
        for measurer in measurers: 

            entry[(measurer, PORT_NUM)] = servers[index]
            index = index + 1
            PORT_NUM += 1 
            if index >= len(servers):
                break 
        server_to_measurer.append(entry)
    return server_to_measurer

# DEPRECATED: assign servers ids
def assign_server_ids(servers):
    server_ids = OrderedDict()
    sid = 0  
    for server in servers: 
        if server in server_ids: 
            raise ValueError("Duplicate Server entry for IDS ", server)
        server_ids[server] = sid 
        sid = sid + 1
    return server_ids

# Currently, we are not using PORT_NUM...
# procs: {node_ip: [(server_ip, port, status)]}
class master_procs:
    def __init__(self, server_to_node, batch_id):
        self.procs = {}
        self.batch_id = batch_id 
        for item in server_to_node:
            for key, value in item.items():
                node_ip = key[0]
                PORT_NUM = key[1]
                server_ip = value
                status = ""

                if node_ip not in self.procs:
                    self.procs[node_ip] = []

                self.procs[node_ip].append((server_ip, PORT_NUM, status))

    # update status regarding one measurer/server
    def update_status(self, node_ip, server_ip, status):
        server_list = self.procs[node_ip]
        for server_tuple in server_list:
            server_tuple_copy = server_tuple
            if server_tuple_copy[0] == server_ip:
                self.procs[node_ip].remove(server_tuple)
                self.procs[node_ip].append((server_ip, server_tuple_copy[1], status))

    # update all servers to one status (such as NOT STARTED, RANDOM STARTED, PROBING STARTED)
    def update_all_servers(self, status):
        for node_ip, server_list in self.procs.items():
            server_list_insert = []
            for server in server_list:
                tuple_insert = (server[0], server[1], status)
                server_list_insert.append(tuple_insert)

            self.procs[node_ip] = server_list_insert

    # write each server's status to file
    def write_status_to_file(self, out_dir, node_index):
        config = configparser.ConfigParser()
        config.read("common_path.ini")
        out_dir = config["common_path"]["command_out_dir"]
        for node_ip, server_list in self.procs.items():
            for server in server_list:
                per_node_folder = os.path.join(out_dir, node_ip)
                per_server_file = os.path.join(out_dir, node_ip+"/"+server[0])

                if not os.path.exists(per_node_folder):
                    os.makedirs(per_node_folder)
                try: 
                    f = open(per_server_file, 'w')
                    f.write(server[2])
                    f.write("\n")
                    f.close()
                except RuntimeError: 
                    print( "Writing status failed for some reason")

    # write the instance info to file
    def write_inst_to_file_for_server( self, inst, out_dir, node_index, server_ip, ext):
        config = configparser.ConfigParser()
        config.read("common_path.ini")
        out_dir = config["common_path"]["command_out_dir"]
        for node_ip, server_list in self.procs.items():
            for server in server_list:
                print("server ", server, "server ip ", server_ip)
                if server[0] == server_ip: 
                    print("equal")
                    per_node_folder = os.path.join(out_dir, node_ip)
                    per_server_file = os.path.join(out_dir, node_ip+"/"+server[0]+ ext)

                    if not os.path.exists(per_node_folder):
                        os.makedirs(per_node_folder)
                    try: 
                        query_json.write_to_json(inst, per_server_file )
                    except RuntimeError: 
                        print( "Writing status failed for some reason for server ", server[0])

    # write each server's status to file
    def write_inst_to_file(self, inst, out_dir, node_index, ext):
        config = configparser.ConfigParser()
        config.read("common_path.ini")
        out_dir = config["common_path"]["command_out_dir"]
        for node_ip, server_list in self.procs.items():
            for server in server_list:
                per_node_folder = os.path.join(out_dir, node_ip)
                per_server_file = os.path.join(out_dir, node_ip+"/"+server[0]+ ext)

                if not os.path.exists(per_node_folder):
                    os.makedirs(per_node_folder)

                query_json.write_to_json(inst, per_server_file )



    # check whether one stage is finished (such as RANDOM ENDED, PROBING ENDED)
    def check_status_finished(self, out_dir, status):
        config = configparser.ConfigParser()
        config.read("common_path.ini")
        out_dir = config["common_path"]["command_out_dir"]
        is_finished = True
        for node_ip, server_list in self.procs.items():
            for server in server_list:
                per_server_file = os.path.join(out_dir, node_ip+"/"+server[0])

                try: 
                    f = open(per_server_file, 'r')
                    read_status = deepcopy(f.readlines())

                    if len(read_status) == 0 : 
                        phase = "" 
                        print("Status file is empty ", per_server_file)
                    else: 
                        print("File is ", per_server_file)
                        phase = read_status[0].strip()

                    if phase != status:
                        is_finished = False
                        print("node_ip %s, server_ip %s: NOT %s" %(node_ip, server[0], status))

                except FileNotFoundError: 
                    print("Writing Status: File {} not exists ".format(per_server_file))
                        

        return is_finished

    # read queries from file
    def read_query(self): 
        config = configparser.ConfigParser()
        config.read("common_path.ini")
        amp_data = OrderedDict() 
        print("Reading queries ")
        for node_ip, server_list in self.procs.items():
            query_dir = os.path.join(config["common_path"]["query_out_dir"], node_ip)
            for server in server_list:
                server_ip = server[0]
                query_file_path = os.path.join(query_dir, server_ip )
                
                try: 
                    data = query_json.read_json(query_file_path)
                    amp_data[server_ip] = data
                except FileNotFoundError:
                    print("File {} not exists".format(query_file_path) )

        return amp_data

    # read queries from file and return a Priority Queue of queries
    def read_query_priority_queue(self, exclude_set=False): 
        config = configparser.ConfigParser()
        config.read("common_path.ini")
        amp_map = OrderedDict() 
        for node_ip, server_list in self.procs.items():
            query_dir = os.path.join(config["common_path"]["query_out_dir"], node_ip)
            print("query_dir ", query_dir)
            for server in server_list:
                
                server_ip = server[0]
                print("server is ", server_ip)
                query_file_path = os.path.join(query_dir, server_ip )
                print("query_file_path is ", query_file_path)
                try:    
                    data = query_json.read_json(query_file_path)

                    p = PriorityQueue() 
                    for entry in data: 
                        amp_factor = float(entry["amp_factor"]) 
                        fv = entry["fields"]

                        if exclude_set and entry["phase"] not in ["random", "probe", "pfs", "BO", "SA"]:
                            continue  
                        p.put((-1.0*amp_factor, entry["server_ip"], [(k, v) for k, v in fv.items()] ))
                    amp_map[server_ip] = p
                except FileNotFoundError: 
                    print("Query file for ", server_ip , "does not exist")  
                    amp_map[server_ip] = PriorityQueue() 

        return amp_map

    # kill all measurers' jobs
    def kill_all_jobs(self):
        for node_ip, server_list in self.procs.items():
            cmd = "sudo ssh -o StrictHostKeyChecking=no root@{} \"sudo pkill python &\""
            cmd = cmd.format(node_ip)
            print(cmd)
            print("killing all process in %s due to timeout" %node_ip)
            subprocess.Popen(cmd, stderr=sys.stdout.fileno(), shell=True)
            time.sleep(1)


    def get_procs(self):
        return self.procs

    def print_procs(self):
        print(self.procs)

# Monitoring running simulated annealing
def run_simulated_annealing(procs, params, node_index): 
    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > float(params.random_timeout)*3600:
            batch_log = "\tSimulated Batch {} {}".format( procs.batch_id,  "TIMEOUT (KILLING JOBS )" )  
            logging.info(batch_log)

            procs.kill_all_jobs()
            break

        time_left = (float(params.random_timeout)*3600 - elapsed_time) / 3600.00
        print("Remaining ", time_left, " hour until Timeout ")

        is_finished = procs.check_status_finished(params.out_dir, "RANDOM ENDED")

        if is_finished:

            batch_log = "\tSimulated Batch {} {}".format( procs.batch_id,  "STATUS FINISHED PROPERLY (BEFORE TIMEOUT)" )  
            logging.info(batch_log)

            print("Simulated Annealing ENDED!!")
            break

        else:
            time.sleep(30)
            print("RANDOM IN PROGRESS!!")
            continue
        #time.sleep(5)
    procs.update_all_servers("RANDOM ENDED")
    procs.write_status_to_file(params.out_dir, node_index)


# Monitor running random phase
def run_random(procs, params, node_index): 
    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time

        # If one phase has been running longer than timeout, kill it
        if elapsed_time > float(params.random_timeout)*3600:
            batch_log = "\tRandom Batch {} {}".format( procs.batch_id,  "TIMEOUT (KILLING JOBS )" )  
            logging.info(batch_log)

            procs.kill_all_jobs()
            break

        # remaining time for the phase
        time_left = (float(params.random_timeout)*3600 - elapsed_time) / 3600.00
        print("Remaining ", time_left, " hour until Timeout ")

        is_finished = procs.check_status_finished(params.out_dir, "RANDOM ENDED")

        # If all measurers finish, finish this stage and update status
        if is_finished:
            batch_log = "\tRandom Batch {} {}".format( procs.batch_id,  "STATUS FINISHED PROPERLY (BEFORE TIMEOUT)" )  
            logging.info(batch_log)

            print("RANDOM ENDED!!")
            break

        # If not all measurers finish, keep monitoring
        else:
            time.sleep(30)
            print("RANDOM IN PROGRESS!!")
            continue

    # When all measurers finish or timeout happens, end this stage and update status for all measurers
    procs.update_all_servers("RANDOM ENDED")
    procs.write_status_to_file(params.out_dir, node_index)


# Monitor running probe phase
def run_probe(procs, params, node_index):
    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time

        # If one phase has been running longer than timeout, kill it
        if elapsed_time > float(params.probe_timeout)*3600:
            batch_log = "\tPROBE Batch {} {}".format( procs.batch_id,  "TIMEOUT (KILLING JOBS )" )  
            logging.info(batch_log)
            procs.kill_all_jobs()
            break

        # remaining time for the phase
        time_left = (float(params.probe_timeout)*3600 - elapsed_time) / 3600.00
        print("Remaining ", time_left, " hour until Timeout ")

        is_finished = procs.check_status_finished(params.out_dir, "PROBE ENDED")

        # If all measurers finish, finish this stage and update status
        if is_finished:
            batch_log = "\tProbe Batch {} {}".format( procs.batch_id,  "STATUS FINISHED PROPERLY (BEFORE TIMEOUT)" )  
            logging.info(batch_log)


            print("PROBE ENDED!!")
            break

        # If not all measurers finish, keep monitoring
        else:
            time.sleep(30)
            print("PROBE IN PROGRESS!!")
            continue

    # When all measurers finish or timeout happens, end this stage and update status for all measurers
    procs.update_all_servers("PROBE ENDED")
    procs.write_status_to_file(params.out_dir, node_index)
    time.sleep(5)

# Monitor running PFS phase
def run_pfs(procs, params, node_index): 
    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time

        # If one phase has been running longer than timeout, kill it
        if elapsed_time > float(params.pfs_timeout)*3600:
            batch_log = "\tPFS Batch {} {}".format( procs.batch_id,  "TIMEOUT (KILLING JOBS )" )  
            logging.info(batch_log)
            procs.kill_all_jobs()
            break

        
        # remaining time for the phase
        time_left = (float(params.pfs_timeout)*3600 - elapsed_time) / 3600.00
        print("Remaining ", time_left, " hour until Timeout ")

        is_finished = procs.check_status_finished(params.out_dir, "PFS ENDED")


        # If one phase has been running longer than timeout, kill it
        if is_finished:
            batch_log = "\tPFS Batch {} {}".format( procs.batch_id,  "STATUS FINISHED PROPERLY (BEFORE TIMEOUT)" )  
            logging.info(batch_log)
            print("PFS ENDED!!")
            break

        # If not all measurers finish, keep monitoring
        else:
            time.sleep(30)
            print("PFS IN PROGRESS!!")
            continue
    
    # When all measurers finish or timeout happens, end this stage and update status for all measurers
    procs.update_all_servers("PFS ENDED")
    procs.write_status_to_file(params.out_dir, node_index)


'''
Run random phase:
    For each measurer and each server in the batch, start random phase
'''
def prepare_random(procs, params, node_index, config): 

    # update all servers' status to RANDOM STARTED
    procs.update_all_servers("NOT STARTED")
    procs.update_all_servers("RANDOM STARTED")
    procs.write_status_to_file(params.out_dir, node_index)

    # prepare arguments for running
    config = configparser.ConfigParser()
    config.read("common_path.ini")
    log_out_dir = config["common_path"]["log_out_dir"]
    tcpdump_dir = config["common_path"]["tcpdump_dir"]

    # get list of all measurers and servers in this batch
    procs_list = procs.get_procs()

    # For each measurer
    for node_ip, server_list in procs_list.items():
        print("node ip ", node_ip)

        # create folder for each measurer
        per_node_dir = os.path.join(tcpdump_dir, node_ip + "/batch_" + str(procs.batch_id))
        if not os.path.exists(per_node_dir):
            os.makedirs(per_node_dir)

        # pcap for storing tcpdump
        per_node_random_file = os.path.join(per_node_dir, "random.pcap")

        # tcpdump for each measurer
        cmd = "sudo ssh -o StrictHostKeyChecking=no root@{} \"cd {} && sudo tcpdump udp port 53 -w {} &\""
        cmd = cmd.format(node_ip, config["measurer_config"]["src_dir"], per_node_random_file)
        print(cmd)
        subprocess.Popen(cmd, stderr=sys.stdout.fileno(), shell=True)
        time.sleep(1.2)

        # for each server that this measurer is assigned with
        for server in server_list:
            server_ip = server[0]

            # create folder for each server
            per_server_dir = os.path.join(log_out_dir, node_ip + "/" + server_ip)

            if not os.path.exists(per_server_dir):
                os.makedirs(per_server_dir)

            # log file for each server
            per_server_random_file = os.path.join(per_server_dir, "random")

            # command for launching random phase
            cmd = "sudo ssh -o StrictHostKeyChecking=no root@{} \"cd {} && {} &&  python3 job.py {} {} 1> /dev/null  2> {} &\""
            cmd = cmd.format(node_ip, config["measurer_config"]["src_dir"], df.ACTIVATE_CONDA, \
                node_ip, server_ip, per_server_random_file)
            print(cmd)
            subprocess.Popen(cmd, stderr=sys.stdout.fileno(), shell=True)
            time.sleep(1.2)


'''
Run probe phase:
    For each measurer and each server in the batch, start probe phase
'''
def prepare_probe(chosen_queries, procs, params, node_index, config): 
    # update all servers' status to RANDOM STARTED
    procs.update_all_servers("PROBE NOT STARTED")
    procs.update_all_servers("PROBE STARTED")
    procs.write_status_to_file(params.out_dir, node_index)

    # store the {measurer:servers} mappings to the file
    procs.write_inst_to_file(chosen_queries, params.out_dir, node_index , ".probe")

    # prepare arguments for running
    config = configparser.ConfigParser()
    config.read("common_path.ini")
    log_out_dir = config["common_path"]["log_out_dir"]
    tcpdump_dir = config["common_path"]["tcpdump_dir"]
    
    # get list of all measurers and servers in this batch
    procs_list = procs.get_procs()
    print("procs list ", procs_list)

    # For each measurer
    for node_ip, server_list in procs_list.items():
        print("node ip ", node_ip)

        # create folder for each measurer
        per_node_dir = os.path.join(tcpdump_dir, node_ip + "/batch_" + str(procs.batch_id))
        if not os.path.exists(per_node_dir):
            os.makedirs(per_node_dir)

        # pcap for storing tcpdump
        per_node_probe_file = os.path.join(per_node_dir, "probe.pcap")

        # tcpdump for each measurer
        cmd = "sudo ssh -o StrictHostKeyChecking=no root@{} \"cd {} && sudo tcpdump udp port 53 -w {} &\""
        cmd = cmd.format(node_ip, config["measurer_config"]["src_dir"], per_node_probe_file)
        print(cmd)
        subprocess.Popen(cmd, stderr=sys.stdout.fileno(), shell=True)
        time.sleep(1.2)

        # for each server that this measurer is assigned with
        for server in server_list:
            server_ip = server[0]

            # create folder for each server
            per_server_dir = os.path.join(log_out_dir, node_ip + "/" + server_ip)
            if not os.path.exists(per_server_dir):
                os.makedirs(per_server_dir)

            # log file for each server
            per_server_probe_file = os.path.join(per_server_dir, "probe")

            # command for launching probe phase
            cmd = "sudo ssh -o StrictHostKeyChecking=no root@{} \"cd {} && {} &&  python3 job.py {} {} 1> /dev/null  2> {} &\""
            cmd = cmd.format(node_ip, config["measurer_config"]["src_dir"], df.ACTIVATE_CONDA, \
                node_ip, server_ip, per_server_probe_file)
            print(cmd)
            subprocess.Popen(cmd, stderr=sys.stdout.fileno(), shell=True)
            time.sleep(1.5)


'''
Run PFS phase:
    For each measurer and each server in the batch, start PFS phase
'''
def prepare_pfs(pfs_inst, procs, params, node_index, config): 
    # update all servers' status to RANDOM STARTED
    procs.update_all_servers("PFS NOT STARTED")
    procs.update_all_servers("PFS STARTED")
    procs.write_status_to_file(params.out_dir, node_index)

    time.sleep(1)

    # prepare arguments for running
    config = configparser.ConfigParser()
    config.read("common_path.ini")
    log_out_dir = config["common_path"]["log_out_dir"]
    tcpdump_dir = config["common_path"]["tcpdump_dir"]
    
    # get list of all measurers and servers in this batch
    procs_list = procs.get_procs()
    print("procs list ", procs_list)

    # For each measurer
    for node_ip, server_list in procs_list.items():
        print("node ip ", node_ip)

        # create folder for each measurer
        per_node_dir = os.path.join(tcpdump_dir, node_ip + "/batch_" + str(procs.batch_id))
        if not os.path.exists(per_node_dir):
            os.makedirs(per_node_dir)

        # pcap for storing tcpdump
        per_node_pfs_file = os.path.join(per_node_dir, "pfs.pcap")

        # tcpdump for each measurer
        cmd = "sudo ssh -o StrictHostKeyChecking=no root@{} \"cd {} && sudo tcpdump udp port 53 -w {} &\""
        cmd = cmd.format(node_ip, config["measurer_config"]["src_dir"], per_node_pfs_file)
        print(cmd)
        subprocess.Popen(cmd, stderr=sys.stdout.fileno(), shell=True)
        time.sleep(1.2)

        # for each server that this measurer is assigned with
        for server in server_list:
            server_ip = server[0]

            if server_ip not in pfs_inst: 
                raise ValueError("PFS inst doees not exist for " , server_ip)
            print("server ip", server_ip)

            # DO NOT start PFS if random fails
            if len(pfs_inst[server_ip]) > 0:
                procs.write_inst_to_file_for_server(pfs_inst[server_ip], params.out_dir, node_index , server_ip, ".pfs")
                time.sleep(0.3)

                # create folder for each server
                per_server_dir = os.path.join(log_out_dir, node_ip + "/" + server_ip)
                if not os.path.exists(per_server_dir):
                    os.makedirs(per_server_dir)

                # log file for each server
                per_server_pfs_file = os.path.join(per_server_dir, "pfs")

                # command for launching PFS phase
                cmd = "sudo ssh -o StrictHostKeyChecking=no root@{} \"cd {} && {} &&  python3 job.py {} {} 1> /dev/null  2> {} &\""
                cmd = cmd.format(node_ip, config["measurer_config"]["src_dir"], df.ACTIVATE_CONDA, \
                    node_ip, server_ip, per_server_pfs_file)
                print(cmd)
                subprocess.Popen(cmd, stderr=sys.stdout.fileno(), shell=True)
                time.sleep(1.5)

'''

    Constructing per-field search instruction
    Need to find (1) the minAF threshold used to prune low AF queries during per-field search
                 (2) Starting queries 
'''

def construct_pfs_instruction(procs, ampmap, params):
    pfs_inst = {}
    for node_ip, server_list in procs.procs.items():

        for server in server_list:
            server_ip = server[0]
            instructions = []
            entry = OrderedDict() 

            # Deal with random failure
            if search_common.pick_server_threshold(ampmap, server_ip, params.minAF) == -1:
                print("ERROR: pick_server_threshold return -1!")
                pfs_inst[server[0]] = instructions

            else:
                # Pick the AF threshold that is considered low (to be used for pruning )
                entry["threshold"] = search_common.pick_server_threshold(ampmap, server_ip, params.minAF)
                instructions.append(entry)

                num_choose_K = params.choose_K
                minAF = params.minAF 

                # Pick starting queries 
                if params.choose_K_max_val: 
                    starting_queries = search_common.find_starting_queries_wrt_max(ampmap, server_ip, num_choose_K)
                elif params.choose_K_random: 
                    

                    starting_queries = search_common.find_starting_queries_wrt_random(ampmap, server_ip, num_choose_K, minAF)
                    print("Starting queries -- RANDOM")
                    print(starting_queries )
                elif params.choose_K_max_dist: 
                    starting_queries = search_common.find_starting_queries_wrt_max_af_dist(ampmap, server_ip, num_choose_K, minAF)
                    print(starting_queries )
                else:                     
                    starting_queries = search_common.find_starting_queries_wrt_max(ampmap, server_ip, params.choose_K)

                for q in starting_queries: 
                    entry = OrderedDict() 
                    entry["amp_factor"] = -1.0* float(q[0])
                    entry["server_ip"] = server_ip #q[1]
                    entry["fields"] = OrderedDict(q[2]) 
                    instructions.append(entry)

                pfs_inst[server[0]] = instructions
    return pfs_inst



def prepare_simulated_annealing(procs, params, node_index, config): 
    procs.update_all_servers("NOT STARTED")

    procs.update_all_servers("SIMULATED STARTED")
    procs.write_status_to_file(params.out_dir, node_index)

    config = configparser.ConfigParser()
    config.read("common_path.ini")
    log_out_dir = config["common_path"]["log_out_dir"]


    procs_list = procs.get_procs()

    for node_ip, server_list in procs_list.items():
        print("node ip ", node_ip)
        for server in server_list:
            server_ip = server[0]

            per_server_dir = os.path.join(log_out_dir, node_ip + "/" + server_ip)

            if not os.path.exists(per_server_dir):
                os.makedirs(per_server_dir)

            per_server_random_file = os.path.join(per_server_dir, "random")
            cmd = "sudo ssh -o StrictHostKeyChecking=no root@{} \"cd {}  && {} &&   python3 job.py {} {} 1> /dev/null  2> {} &\""
            

            cmd = cmd.format(node_ip, config["measurer_config"]["src_dir"], df.ACTIVATE_CONDA,  node_ip, \
                server_ip, per_server_random_file)


            subprocess.Popen(cmd, stderr=sys.stdout.fileno(), shell=True)
            time.sleep(1.2)




'''
    Runs the entire workflow (Controlller side)
'''

def start_measurement_batch(params, input_path):
    print("start measurement batch!")
    # read common_path.ini
    config = configparser.ConfigParser()
    config.read("common_path.ini")

    # store servers
    servers = []
    fileds = ""

    # Read protocol fields (irrespective of protocol this should be same )
    proto_fields =  inputs.generate_proto_fields(input_path)
    field_names = list(proto_fields.keys())
    servers = inputs.load_proto_servers(input_path)

    servers = [item[0] for item in servers]

    # split servers for batch processing
    server_size = len(servers) 
    block_size = int(params.block_size)
    print("server size: %d, block_size: %d" %(server_size, block_size))
    servers_split = list(helper.chunks(servers, math.ceil(float(server_size)/float(block_size))))

    for i, server_split in enumerate(servers_split):
        print("server batch #%d" %int(i+1))
        print(server_split)


    
    nodes = inputs.load_nodes(input_path)

    node_index = {}
    node_host = {}

    args = vars(params)
    f = open(config["common_path"]["common_config_file"], 'w')
    json.dump(args, f)
    f.close()


    log_file = os.path.join( config["common_path"]["log_out_dir"], "time_status.log") 


    logging.basicConfig(filename=log_file,format='%(levelname)s  :  %(asctime)s \t %(message)s', \
        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)

    logging.info("Experiment Start")

    
    procs_list = []


    '''
        Assign server to node
        Batch processing
        Kill all jobs only once
    '''
    logging.info("Start Assigning servers to batches")
    is_killed = False
    batch_id = 1 
    for servers in servers_split:   
        logging.info("batch {}".format(batch_id))
        server_to_node = assign_servers_to_node(servers, nodes)
        print("server to nodes ", server_to_node)
        logging.info("server to nodes ")
        logging.info(server_to_node)
        
        procs = master_procs(server_to_node, batch_id)
        batch_id = batch_id + 1
        #kill all jobs just in case 
        if is_killed == False:
            procs.kill_all_jobs()
            is_killed = True

        # add procs to procs list
        procs_list.append(procs)

        procs.print_procs()

    logging.info("Finished cleaning up existing processes")
    

    '''
        Start of 3-stage pipeline
        Stage 1: Random Sampling 
    '''

    for procs in procs_list:
        batch_log = "Random Batch {}/{} {}".format( procs.batch_id, len(procs_list), "STARTED" )  
        logging.info(batch_log)
        prepare_random(procs, params, node_index, config)
        run_random(procs, params, node_index)

        batch_log = "Random Batch {}/{} {}".format( procs.batch_id, len(procs_list), "ENDED" )  
        logging.info(batch_log)
    


    if params.random_accepted == False:

        '''
            Stage 2: Probing stage 
        '''
        logging.info(" Picking chosen queries ")

        ampdata = OrderedDict()
        for procs in procs_list:
            ampdata_ = procs.read_query()
            for key, value in ampdata_.items():
                ampdata[key] = value



        if params.num_probe > 0: 
            # if random > num probe exceeds the total budget 
            # flatten the number of probe queries 
            if params.per_server_random_sample + params.num_probe >= params.per_server_budget:
                orig_probe = params.num_probe  
                params.num_probe = params.per_server_budget - params.per_server_random_sample 
                logging.debug(" Scaled  num probe to {} from {}".format(params.num_probe , orig_probe) )


            assert(params.probe_numcluster > 0  )



            if params.probe_numcluster == 0: 
                chosen_queries = []
                logging.debug(" Skipping clustering as num cluster is {}".format(params.probe_numcluster ))
            else: 
                '''
                    Run the clustering  
                '''
                chosen_queries = cluster.cluster(ampdata, proto_fields, params.probe_numcluster, \
                    params.minAF, params.num_probe, params.measurement, params,  config,  log_file )
            logging.debug("Finished clustering  - return length {}".format( len(chosen_queries)))

            
            # sometimes there are no queries with AF >= the min threshold 
            if len(chosen_queries) == 0: 
                logging.debug("CLUSTER returns  AS NO DATA with AF > MIN THRESHOLD ")#.format(args.probe_numcluster ))
            else: 


                # For debugging purposes, print the chosen queries 
                logging.debug("\n\nChosen queries : " )
                for i in chosen_queries: 
                    logging.debug(i)

                for procs in procs_list:
                    batch_log = "Probe Batch {}/{} {}".format( procs.batch_id, len(procs_list), "STARTED")  
                    logging.info(batch_log)
                    prepare_probe(chosen_queries, procs, params, node_index, config)
                    run_probe(procs, params, node_index)
                    batch_log = "Probe Batch {}/{} {}".format( procs.batch_id, len(procs_list), "ENDED")  
                    logging.info(batch_log)

        else:
            logging.debug("Num probe is {} so skpping PROBING".format(params.num_probe))
        
        

        '''
           Stage 3: Per-Field Search 
        '''
        for procs in procs_list:
            batch_log = "PFS Batch {}/{} {}".format( procs.batch_id, len(procs_list), "STARTED")  
            logging.info(batch_log)


            #if params.start_fro
            ampmap = procs.read_query_priority_queue()

            pfs_inst = construct_pfs_instruction(procs, ampmap, params) 
            logging.debug("pfs_inst: ")
            logging.debug(pfs_inst)
            prepare_pfs(pfs_inst, procs, params, node_index, config)

            run_pfs(procs, params, node_index)

            batch_log = "PFS Batch {}/{} {}".format( procs.batch_id, len(procs_list), "ENDED")  
            logging.info(batch_log)

    '''
       Finally prints the output 
    '''
    logging.info("DUMPING starts")
    ampmap = OrderedDict()
    for procs in procs_list:
        ampmap_ = procs.read_query_priority_queue(exclude_set=True)
        for key, value in ampmap_.items():
            ampmap[key] = value
    
    if params.local_exp_cmu == True:
        base_file = config["cmu_local_exp_config"]["search_out_dir_controller"]
    else:
        base_file = os.path.join(config["measurer_config"]["src_dir"], config["common_path"]["search_out_dir"])

    query_json.print_ampmap(ampmap, base_file)
    logging.info("DUMPING ends")




def read_pfs_instruction_from_file(file): 
    pfs_inst = {} 
    with open(file) as f:
        data = json.load(f)
    pfs_inst[data[1]["server_ip"]] = data 
    return pfs_inst



def start_random(params, input_path):
    # read common_path.ini
    config = configparser.ConfigParser()
    config.read("common_path.ini")

    # store servers
    servers = []
    fileds = ""

    #Read protocol fields (irrespective of protocol this should be same )
    proto_fields =  inputs.generate_proto_fields(input_path)
    field_names = list(proto_fields.keys())
    servers = inputs.load_proto_servers(input_path)


    #servers = ["8.8.8.8", "1.1.1.1"]
    servers = [item[0] for item in servers]

    #read from ini .. change load_measurers_servers to load_nodes 
    nodes = inputs.load_nodes(input_path)

    node_index = {}
    node_host = {}


    server_to_node = assign_servers_to_node(servers, nodes)
    print("server to nodes ", server_to_node)
    
    args = vars(params)
    f = open(config["common_path"]["common_config_file"], 'w')
    json.dump(args, f)
    f.close()

    procs = master_procs(server_to_node,1)
    procs.kill_all_jobs()


    prepare_random_simulated(procs, params, node_index, config)
    run_random(procs, params, node_index)


    # '''
    #   Finally prints the output 
    # '''
    ampmap = procs.read_query_priority_queue()
    # use separate path for CMU local exp
    if params.local_exp_cmu == True:
        base_file = config["cmu_local_exp_config"]["search_out_dir_controller"]
    else:
        base_file = os.path.join(config["measurer_config"]["src_dir"], config["common_path"]["search_out_dir"])
    query_json.print_ampmap(ampmap, base_file)


def start_simulated_batch(params, input_path):
    print("start simulated batch!")
    # read common_path.ini
    config = configparser.ConfigParser()
    config.read("common_path.ini")

    # store servers
    servers = []
    fileds = ""


    # Read protocol fields (irrespective of protocol this should be same )
    proto_fields =  inputs.generate_proto_fields(input_path)
    field_names = list(proto_fields.keys())
    servers = inputs.load_proto_servers(input_path)

    #servers = ["8.8.8.8", "1.1.1.1"]
    servers = [item[0] for item in servers]

    # split servers for batch processing
    server_size = len(servers) #int(params.server_size)
    block_size = int(params.block_size)
    print("server size: %d, block_size: %d" %(server_size, block_size))
    servers_split = list(helper.chunks(servers, math.ceil(float(server_size)/float(block_size))))

    for i, server_split in enumerate(servers_split):
        print("server batch #%d" %int(i+1))
        print(server_split)


    
    nodes = inputs.load_nodes(input_path)

    node_index = {}
    node_host = {}

    args = vars(params)
    f = open(config["common_path"]["common_config_file"], 'w')
    json.dump(args, f)
    f.close()


    log_file = os.path.join( config["common_path"]["log_out_dir"], "time_status.log") 


    logging.basicConfig(filename=log_file,format='%(levelname)s  :  %(asctime)s \t %(message)s', \
        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)

    logging.info("Experiment Start")

    
    procs_list = []

    # Assign server to node
    # Batch processing
    # Kill all jobs only once
    logging.info("Start Assigning servers to batches")
    is_killed = False
    batch_id = 1 
    for servers in servers_split:   
        logging.info("batch {}".format(batch_id))
        server_to_node = assign_servers_to_node(servers, nodes)
        print("server to nodes ", server_to_node)
        logging.info("server to nodes ")
        logging.info(server_to_node)
        
        procs = master_procs(server_to_node, batch_id)
        batch_id = batch_id + 1
        #kill all jobs just in case 
        if is_killed == False:
            procs.kill_all_jobs()
            is_killed = True

        # add procs to procs list
        procs_list.append(procs)

        procs.print_procs()

    logging.info("Finished cleaning up existing processes")
    
    
    for procs in procs_list:
        batch_log = "Simulated Random Batch {}/{} {}".format( procs.batch_id, len(procs_list), "STARTED" )  
        logging.info(batch_log)
        prepare_simulated_annealing(procs, params, node_index, config)
        run_simulated_annealing(procs, params, node_index)

        batch_log = "Simulated  Random Batch {}/{} {}".format( procs.batch_id, len(procs_list), "ENDED" )  
        logging.info(batch_log)
    
    
    # '''
    #   Finally prints the output 
    # '''
    logging.info("DUMPING starts")
    ampmap = OrderedDict()
    for procs in procs_list:
        ampmap_ = procs.read_query_priority_queue(exclude_set=True)
        for key, value in ampmap_.items():
            print(key, value )
            ampmap[key] = value
    
    # use separate path for CMU local exp
    if params.local_exp_cmu == True:
        base_file = config["cmu_local_exp_config"]["search_out_dir_controller"]
    else:
        base_file = os.path.join(config["measurer_config"]["src_dir"], config["common_path"]["search_out_dir"])

    query_json.print_ampmap(ampmap, base_file)
    logging.info("DUMPING ends")




