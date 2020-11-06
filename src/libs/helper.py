import numpy as np
import time
import itertools
from queue import PriorityQueue
import argparse
from copy import deepcopy
from collections import OrderedDict
import string, random, math 
from datetime import datetime
import sys
import pickle
import libs.definition as df
import subprocess



def execute_in_shell(command):
    output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
    return output.strip()

def execute_in_shell_async(command):
    return subprocess.Popen(command, stderr=subprocess.STDOUT, shell=True)


def kill_measurer_process(measurer): 

	#for measurer in measurers: 
	print(measurer)
	privateip = measurer[0]
	port = measurer[1]
	user = measurer[3]
	address = measurer[4]
	directory = measurer[5]
	script = measurer[6]
	command = "ssh -o StrictHostKeyChecking=no %s@%s \"sudo killall python3 || true\""
	command = command % (user, address) 
	return execute_in_shell(command)

def start_measurer_processes(measurer): 

    #for measurer in measurers: 
    print(measurer)
    privateip = measurer[0]
    port = measurer[1]
    user = measurer[3]
    address = measurer[4]
    directory = measurer[5]
    script = measurer[6]
    
    command = "ssh -o StrictHostKeyChecking=no %s@%s \"cd %s; sudo python3 %s/%s %s %s &\""
    command = command % (user, address, directory, directory, script, privateip, port) 
    print(command)
    return execute_in_shell_async(command)




def align_fields(dict_to_align, fields): 
    d = OrderedDict() 

    for fid, f_metadata in fields.items(): 
        if f_metadata.is_int == True: 
            val = int(dict_to_align[fid])
        else: 
            val = str( dict_to_align[fid])
        d[fid] = val
    return list(d.values())

def convert_field_list_to_dict(fields, field_name):
	query = OrderedDict()
	if (len(fields) != len(field_name)):
		print("fields ", fields)
		print("field names ", field_name)
		raise ValueError("lenght not same ")
	for i in range(len(field_name)): 
		query[field_name[i]] = fields[i]
	return query

def update_amp_map_v2(amp_found_map, amp_results, db_handler, phase):
	query_counter = 1

	if "total" not in amp_found_map:
		amp_found_map["total"] = PriorityQueue()
	print("length of update is ", len(amp_results))

	for index in range(len(amp_results)):
		query_field_values =  list( amp_results[index]["fields"].values())
		amp_factor = round(amp_results[index]["amp_factor"], 3)
		server = amp_results[index]["server_ip"]
		server_id =  amp_results[index]["server_id"]
		node = amp_results[index]["node_ip"]
		query_id = generate_query_id(server_id, phase, query_counter)
		amp_results[index]["query_id"] = query_id

		query_counter += 1
		amp_found_map["total"].put((-amp_factor, server, query_field_values))
		if server not in amp_found_map:
		    pq = PriorityQueue()
		    pq.put((-amp_factor, server, query_field_values))
		    amp_found_map[server] = pq 
		else:
		    amp_found_map[server].put((-amp_factor, server, query_field_values))


def update_map(amp_map, record_to_push): 

    while not record_to_push.empty():
        entry = record_to_push.get()
        server_id = entry[1]
        amp_map[ server_id ].put(( deepcopy(entry[0]), server_id, deepcopy(entry[2]) ) )

    return amp_map 


def update_map_from_list(amp_map, record_to_push): 

    if amp_map == None:
        return amp_map

    for entry in record_to_push: 
        server_id = entry[1]
      
        #update ampmap 
        amp_map[ server_id ].put(( deepcopy(-1.0*entry[0]), server_id, deepcopy(entry[2]) ) )

    return amp_map 



def update_amp_map(amp_found_map, amp_results):
    query_counter = 1
    for index in range(len(amp_results)):
        server = amp_results[index][3]
        amp_factor = int(amp_results[index][4])
        query_field_values = amp_results[index][1]

        #testing db inserts
        print("inserting query " + str(query_counter), end='\r')
        db_handler.insert(db_handler.parse_results(amp_results[index]))
        query_counter += 1
        if server not in amp_found_map:
            pq = PriorityQueue()
            pq.put((-amp_factor, server, query_field_values))
            amp_found_map[server] = pq
        else:
            amp_found_map[server].put((-amp_factor, amp_results[index][3], amp_results[index][1]))


def update_querybuffer_from_list(record_to_push, node_ip, phase, proto_fields):
    new_buf = [] 
    field_names = list(proto_fields.keys())
    for entry in record_to_push: 
        #entry = record_to_push.get()
        try: 
            assert(len(field_names) == len(entry[2]))
        except: 
            print("field_name ", field_names)
            print("field val ", entry[2])
            raise ValueError( " Field names and field valus size do not match ")
        server_ip = entry[1]
        af =  deepcopy(entry[0])
        field_values = deepcopy(entry[2])
        field_dict_insert  = {}

        for i in range(len(field_names)): 
            fid = field_names[i] 

            try: 
                f_metadata = proto_fields[fid]
                if f_metadata.is_int == True: 
                    fv = int(field_values[i])
                else: 
                    fv = str(field_values[i])
            except ValueError: 
                print("Casting failed for fv ", fv , "for fid ", fid )
                print("Cast to STR ")
                fv = str(field_values[i])

            field_dict_insert[fid] = fv
        print("field values ", field_dict_insert)

        buf_entry = gen_query_buffer_entry(field_dict_insert, af, server_ip, node_ip, phase )
        new_buf.append( buf_entry)

    return new_buf



def gen_query_buffer_entry(field_values, af, server_ip, measurer_ip, phase):
    insert_data = OrderedDict() 
    insert_data["fields"] =  field_values
    insert_data["amp_factor"] = af 
    insert_data["server_ip"] = server_ip

    insert_data["node_ip"] = measurer_ip
    insert_data["phase"] = phase 
    return insert_data


def chunks(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
