import numpy as np
import time, os
import itertools
from queue import PriorityQueue
import argparse
import copy
from collections import OrderedDict
import string, random, math 
from datetime import datetime
import sys
import pickle
import libs.definition as df
import libs.query_node as qnode
from copy import deepcopy
import queue

import libs.inputs as inputs
import libs.helper as helper

import libs.budget as budget
import libs.query_json as query_json
import configparser

import platform



'''
    Random Stage for AmpMap: randomly picks a value for each field 
    and sends to the server 

'''
def generate_random_queries(BB, proto_fields, measurer_ip,  args, phase="random"):# proto,  server_ip, nmapDnsObj, num_queries, buffer_query, time_sleep):

    config = configparser.ConfigParser()
    config.read("common_path.ini")
    query_out_dir = os.path.join(config["common_path"]["query_out_dir"], measurer_ip)


    if not os.path.exists(query_out_dir):
        os.makedirs(query_out_dir)

    proto = args[df.PROTO]
    #number of queries to spend on the random search 
    num_rand_queries = args[df.PER_SERVER_RANDOM_SAMPLE]
    server_ip = args["server_ip"]
    
    # timeout between subsequent queries 
    time_sleep = float(args[df.TIME_SLEEP])


    buffer_query = int(args["update_db_at_once"])
    queryBuffer = []


    for i in range(num_rand_queries):
        insert_data = OrderedDict()
        print("query num : ", i )


        # construct field values for a query 
        field_values = pick_random_query(proto_fields)
        # gets the response      
        af = BB.get_af_dict( server_ip,  field_values )
        insert_data = helper.gen_query_buffer_entry(field_values, af, server_ip, measurer_ip, phase ) 
        queryBuffer.append(insert_data)

        # flush the buffer when the buffer is full 
        if len(queryBuffer) >= buffer_query:
            print("updating query buffer with len ", len(queryBuffer))
            query_out_filename = os.path.join(query_out_dir, server_ip)
            query_json.write_to_json(queryBuffer, query_out_filename)
            queryBuffer.clear()
        time.sleep(time_sleep)


    # flush the buffer for the remaining queries 
    if len(queryBuffer) != 0 : 
        print("updating query buffer with len ", len(queryBuffer))
        query_out_filename = os.path.join(query_out_dir, server_ip)
        query_json.write_to_json(queryBuffer, query_out_filename)
        queryBuffer.clear()  

    return None 



'''
    Sends probing queries to the servers 
    and stores the AF 
'''
def send_probing_queries(BB, proto_fields, measurer_ip, chosen_queries,  args): 


    #  Setup  and Parameters 
    phase = "probe"
    config = configparser.ConfigParser()
    config.read("common_path.ini")
    query_out_dir = os.path.join(config["common_path"]["query_out_dir"], measurer_ip)

    if not os.path.exists(query_out_dir):
        os.makedirs(query_out_dir)

    proto = args[df.PROTO]
    server_ip = args["server_ip"]
    time_sleep = float(args[df.TIME_SLEEP])
    buffer_query = int(args["update_db_at_once"])
    #time_sleep = 0.2
    queryBuffer = []


    '''
        Given the probing queries, send each to the server
        and record the AF 
        Store the results in  queryBuffer and flush when full 

    '''
    for i in range(len(chosen_queries)):

        field_values = chosen_queries[i]
        print("Chosen query field values ", field_values)
        af = BB.get_af_dict( server_ip,  field_values )
        insert_data = helper.gen_query_buffer_entry(field_values, af, server_ip, measurer_ip, phase ) 
        queryBuffer.append(insert_data)


        if len(queryBuffer) >= buffer_query:
            print("updating query buffer with len ", len(queryBuffer))
            query_out_filename = os.path.join(query_out_dir, server_ip)
            query_json.write_to_json(queryBuffer, query_out_filename)
            queryBuffer.clear()

        time.sleep(time_sleep)

    if len(queryBuffer) != 0 : 
        print("updating query buffer with len ", len(queryBuffer))
        query_out_filename = os.path.join(query_out_dir, server_ip)
        query_json.write_to_json(queryBuffer, query_out_filename)

        queryBuffer.clear()  

    return None 


'''
    Per-Field Search 
'''

def per_field_search(BB, proto_fields,  server_ip, starting_queries,  AF_THRESH,  args , connection=None, amp_found_map=None ):


    track_queries = OrderedDict()

    print("Args in PFS ", type(args))
    if isinstance(args, argparse.Namespace):
        args = vars(args) 
   

    #number of queries spent so far 
    query_random_and_probe = args[df.PER_SERVER_RANDOM_SAMPLE] + args[df.NUM_PROBE]
    
    # stores as the budget spent so far  
    num_total_query = budget.Budget(query_random_and_probe)  
    

    field_names = list(proto_fields.keys())
    per_server_budget = args[df.PER_SERVER_BUDGET]

    phase = "pfs"
    

    # Keeps track of patterns found so far 
    patterns_found = set()
    
    query_buffer_total = []

    # number of parameters if this was measurements (opposed to simulation)
    if args["measurement"] == True: 
        time_sleep = float(args[df.TIME_SLEEP])
        buffer_query = int(args["update_db_at_once"])
        
        #the measurer IP 
        node_ip = args["node_ip"]
        config = configparser.ConfigParser()
        config.read("common_path.ini")
        query_out_dir = os.path.join(config["common_path"]["query_out_dir"], node_ip)
        query_out_filename = os.path.join(query_out_dir, server_ip)

    '''
        Run PFS search for each starting query. 
        For the default AmpMap setting, we chose 1 starting query only  
    ''' 
    for q_k in starting_queries:  
        print("\nStartign query ", q_k)


        #in case, we have > 1 starting queries, we equally split the budget for each starting point 
        budget_per_starting = int( (per_server_budget - query_random_and_probe) / args["choose_K"])
        print(budget_per_starting )

        #number of queries spent so far for this starting query  
        num_query = budget.Budget(0)


        #initialize the current query 
        if args["measurement"] == True: 
            assert(server_ip == q_k["server_ip"])
            cur_query  = deepcopy(helper.align_fields(q_k["fields"], proto_fields)) 
        else: 
            cur_query  = deepcopy(list(q_k[2] )) 


        print("cur query ", cur_query)
        # initialize the maximum AF seen so far  
        max_amp_factor = BB.get_af(server_ip, field_names,  cur_query)   

        stack = list()
        pattern_id = deepcopy(cur_query)
   

        '''
            Qnode stores 
            (1) the query field values 
            (2) depth (in depth first search concept)
            (3) the amplification factor (AF )
            (4) the server IP 
            (5) pattern id where each field is a value or a range 
        '''
        q_node = qnode.QueryNode(deepcopy(cur_query), 0, max_amp_factor, server_ip, tuple(pattern_id) )
        stack.append(q_node) 

        # the high level idea is similar to the depth first searc h
        while len(stack) != 0 :
            #current element (query node ) to explore and search neighbors 
            v = stack.pop()
        
            # reaches the maximum depth (size of the fields) then skip 
            if v.depth >= len(field_names): 
                continue 

            pattern_id = v.cluster_id
            v.print()
            print("\tCurrent pattern found ", patterns_found)


            '''
                Checks if a new pattern 
                If not a new pattern, return the matched pattern 
            '''
            is_new, matched_pattern = is_new_pattern(pattern_id, patterns_found )


            ''' 
                If we want to disable checking new pattern (for analysis purposes), 
                assume its a new pattern always to force exploration 
            ''' 
            if  args["disable_check_new_pattern"] == True: 
                is_new = True 


            if is_new :  
                print("\t New pattern discovered " , patterns_found)
                patterns_found.add(pattern_id) 


                ''' 
                    Searches the neighboring queries 
                    Returns the neighbor list (a list of queries) and their corresponding AFs (amp_list )
                ''' 
                neighbor_list, amp_list = find_neighbors(v, server_ip,  BB, proto_fields, args, AF_THRESH, \
                     num_query, budget_per_starting, track_queries)
                

                if args["measurement"] == True: 
                    query_buffer  = helper.update_querybuffer_from_list( amp_list,  node_ip, phase, proto_fields)
                    query_buffer_total += query_buffer
                else: 
                    helper.update_map_from_list(amp_found_map, amp_list) 

                if num_query.budget >= budget_per_starting:
                    print("Exiting as the budget is all used up ", num_query.budget)
                    break

                for neighbor in neighbor_list: 
                    stack.append(neighbor)

            else: 
                print("\n\tIgnore a field val ", v.field_values , " with patternid ", pattern_id)
                print("\t Matched pattern : ", matched_pattern)
                

                # Update the pattern ID to the  one that is a SUPERSET 
                new_pattern_id = []
                for i in range(len(pattern_id)): 
                    orig = pattern_id[i]
                    matched = matched_pattern[i]
                    if is_contained(orig, matched): 
                        new_pattern_id.append(matched)
                    else:
                        new_pattern_id.append(orig)
                print("\tUpdated pattern is ", new_pattern_id)
                
                # Remove the old pattern and insert the UPDATED pattern 
                if matched_pattern in patterns_found: 
                    patterns_found.remove(matched_pattern)
                    patterns_found.add(tuple(new_pattern_id))
                print("\tUpdated pattern map  ", patterns_found)

            num_total_query.increase(num_query.budget)


    # Flush the buffer  
    if args["measurement"] == True and len(query_buffer_total) > 0:
        query_out_filename = os.path.join(query_out_dir, server_ip )
        query_json.write_to_json(query_buffer_total, query_out_filename)
        query_buffer_total.clear()

    return  

   

''' 
    Simple log sampling 
'''
def log_sample_from_range(list_search):

    # checking if this range is contiguous  
    if (list_search[-1] - list_search[0]) != (len(list_search) -1): 
        print("list is ", list_search)
        raise ValueError("The list (for long field) should be contiguous ") 

    # Number of sample is the number of bits 
    num_sample = int(np.log2(len(list_search)+1))
    if list_search[0] == 0: 
        sampled = [x-1 for x in np.geomspace(1, list_search[-1], num=num_sample, dtype=int)]
        #sampled = [x-1 for x in np.geomspace(1, list_search[-1], num=df.LARGE_FIELD_SAMPLE_SIZE, dtype=int)]
        sampled_int = [int(x) for x in sampled]
        return sampled
    sampled = [x for x in np.geomspace(list_search[0], list_search[-1], num=num_sample, dtype=int)]
    sampled_int = [int(x) for x in sampled]
    return sampled_int

def uniform_sample_from_range(list_search, num_sample):
    #Assume contiguous long range  
    if (list_search[-1] - list_search[0]) != (len(list_search) -1): 
        print("list is ", list_search)
        raise ValueError("The list (for long field) should be contiguous ") 
    return [ int(elem) for elem in np.linspace(list_search[0],list_search[-1], num_sample) ]

def __get_items_from_priority_queue(queue, minval=0):

    items_lst = [] 
    new_queue =  PriorityQueue() 
    while not queue.empty():  
        elem = queue.get() 
        new_queue.put( elem  )

        amp_factor = -1.0 * elem[0]
        #print(amp_factor, minval)
        if amp_factor >= minval:
            items_lst.append(elem)
    return new_queue, items_lst





def find_starting_queries_wrt_max(amp_found_map, server_ip, num_find):
    starting_queries = []
    found = 0 
    while found < num_find: 
        query = amp_found_map[server_ip].get() 
        starting_queries.append(query)
        found = found + 1
    print("Starting query : ", starting_queries , " For server ", server_ip)
    return starting_queries


def find_starting_queries_wrt_max_af_dist(amp_found_map, server_ip, num_find, minAF=0):
    starting_queries = []

    new_queue, items_lst = __get_items_from_priority_queue(amp_found_map[server_ip], minAF )
    print("items list : ",  items_lst)  
    print("length ", len(items_lst))
    percentile_index = np.linspace(0,1,num_find)*(len(items_lst)-1)
    print("perc index ", percentile_index)
    percentile_index = [int(p) for p in percentile_index]
    print("perc index ", percentile_index)

    starting_queries = []
    for index in percentile_index: 
        starting_queries.append( items_lst[index])
    amp_found_map[server_ip] = new_queue 

    print("Maximize the AF distance ", starting_queries )
    return starting_queries



# Above certain threshold. 
def find_starting_queries_wrt_random(amp_found_map, server_ip, num_find, minAF=0):

    starting_queries = []
    #found = 0 
    #print("get items from PQ ")
    new_queue, items_lst = __get_items_from_priority_queue(amp_found_map[server_ip], minAF )
        #print("items list : ",  items_lst)  
    starting_queries = random.choices(items_lst, k=min(num_find, len(items_lst)) )
    amp_found_map[server_ip] = new_queue 
    print("RANDOM - Starting queries ", starting_queries )
    return starting_queries





def get_field_names(AF): 
    return list(AF.all_fields.keys()) 





'''
    Picks the threshold given amp_found_map
    Just picks the top AF_THRESHOLD_RANK entry from the map
'''
def pick_server_threshold(amp_found_map, server_ip, minAF): 
    print("\n\n In Pick Server Threshold") 
    check_af = 0 

    if server_ip not in amp_found_map:
        print("Server" , server_ip , " does not exist in the map ")
        raise ValueError("Server" , server_ip , " does not exist in the map ")

    print("Server", server_ip, "exist in the map ")

    pq = amp_found_map[server_ip]

    # Deal with random failure
    if len(pq.queue) == 0:
        print("pq is empty!")
        return -1

    else:
        print("pq is not empty!")
        visited = OrderedDict() 

        #Obtain the thresholds ->    rank = 1
        tmp_pq = PriorityQueue()  
        
        q_max = pq.get() 
        
        fv_max = q_max[2]

        print("fv_max: ", fv_max)
        tmp_pq.put( q_max  )

        max_af = -1.0*q_max[0]
        #af_thresholds = max_af / 2.0 #:queries
        if max_af >= minAF*2: 
           print("Maximum af is greater than 20 ")
           af_thresholds = minAF
        else: 
           af_thresholds = max_af / 2.0  

        print("af_thresholds: ", af_thresholds)
        
        #Now, resotre into the map 
        while not tmp_pq.empty(): 
            restore = tmp_pq.get() 
            print("\t\tRestoring ", restore)
            amp_found_map[ restore[1] ].put(restore)
        print("\n\tThreshold Picked ", af_thresholds)
        return af_thresholds



def pick_random_query_all_fields(AF):

    query = []
    for f_id, f_metadata in AF.all_fields.items(): 
        
        val_picked = random.sample(f_metadata.accepted_range,1)[0]
        #val_picked = random.randint(0, f_metadata.size-1)
        query.append(val_picked)
    #print(query)
    return query
 




def get_new_cluster_id( parent_cid, field_index, range_for_field  ): 

    cid = list(parent_cid)
    cid[field_index] = range_for_field
    #print(" old cid ", parent_cid , " and new cid ", cid)
    return tuple(cid)




''' 
    Sweeps a field (either exhaustive or log sampling depending on the field type )
    (1) BB: an interface for the black-box server or the simulation server 
    (2) query to start exploring neighboring queries 
    (3) field_index: the index in a field list indicating which field to search 
    (4) field_metadata: metadata such as field size 
    (5) server : server IP 
    (6) num_query : stores the current queries spent so far 
    (7) per_q_budget: the maximum budget for a given server 
    (8) track_queries : Since log sampling is deterministic, we want to avoid 
    sending queries we already sent so track_queries stores the RAW queries sent so far

''' 
def sweep_field(BB, query, field_index, field_metadata, field_names, server, num_query, per_q_budget, track_queries, args):

    args_dict = args
    if not(type(args_dict) == dict or type(args_dict) == OrderedDict):
        args_dict = vars(args)

    is_measurement = args_dict["measurement"]
    proto = ""
    time_slee= ""

    if is_measurement == True: 
        proto = args_dict[df.PROTO]
        time_sleep = float(args_dict[df.TIME_SLEEP] )
    amp_list = [] 
    field_size = field_metadata.size

    field_values = [] 


    # For LARGE fields with size >= small field threshold (default 256)
    if field_size >= df.SMALL_FIELD_THRESHOLD:       


        # if sampling is disabled (for analysis purposes only)
        if  args_dict["disable_sampling"] == True: 
            field_values = field_metadata.accepted_range  
        else: 
            # Log sampling 
            field_values = log_sample_from_range(field_metadata.accepted_range )  
            print("field values" , field_values)
    else: 
        # just do exhaustive search 
        field_values = field_metadata.accepted_range  
        


    query_test = deepcopy(query)
    num_itr =0

    '''
        Having defined field values to search, 
        send the query to the server and obtain AF 
    '''
    for j in range(0, len(field_values)):
        query_test[field_index] = field_values[j]
        if len(field_names ) != len(query_test): 
            raise ValueError("field name and query test names should equal ", field_names, " " , query_test)
        amp_factor = 0 


        insert_track_queries = tuple(zip( field_names, query_test )) 

        # send this query if this EXACT raw query was not sent before 
        if insert_track_queries not in track_queries: 

            amp_factor = BB.get_af(server, field_names, query_test ) 
            
            #indicate that we spent one query  just now 
            num_query.increase(1)

            if args["measurement"] == True: 
                time.sleep(time_sleep)

            amp_list.append( (amp_factor, server, deepcopy(query_test)  ) )
            num_itr = num_itr + 1 
            track_queries[insert_track_queries] = 1 
        else: 
            print("Skip  ", query_test )

        # If we spent all the budget, just break and return what we have so far 
        if num_query.budget >= per_q_budget :
            print("Budget expleted  ", num_query.budget)
            break 


    return amp_list, num_itr, num_query, track_queries 
   




'''
    Simple sanity checking whether 
    field_search and field_name are consistent 
'''
def sanity_check_order(field_search, field_name ):
    i = 0 
    for key, value in field_search.items():
        if field_name[i] != key: 
            print("Field search  ", field_search, field_name )
            raise ValueError("Index of traversing the field is not consistent ")
        i = i + 1

def find_discrete_range(amp_list, field_index, cur_query, AF_THRESH ): 

    val_range = []
    candidates = OrderedDict()

    for amp_entry in amp_list: 
        curAF = amp_entry[df.AF_INDEX]
        field_val = amp_entry[df.FIELD_VAL_INDEX][field_index]
        print("\t\t\t",curAF , " -> " , field_val)
         
        if curAF >= AF_THRESH: 
            candidates[field_val] = [amp_entry]
            val_range.append(field_val )

    return val_range, candidates


def find_contiguous_range(amp_list, field_index, cur_query, AF_THRESH ): 
    val_range = []
    candidates = OrderedDict()
    candidate_entry = []
    r_index = 0 

    is_active = False
    start = None
    end = None
    for amp_entry in amp_list: 
        curAF = amp_entry[df.AF_INDEX]
        field_val = amp_entry[df.FIELD_VAL_INDEX][field_index]
        print("\t\t\t",curAF , " -> " , field_val)
        if is_active == True: 
            if curAF >= AF_THRESH: 
                candidate_entry.append(amp_entry)
                end = field_val 
            else:
                if start == end: 
                    val_range.append(start)
                    candidates[start] = candidate_entry
                else: 
                    val_range.append((start, end))
                    candidates[(start,end)] = candidate_entry
                is_active = False
                candidate_entry = []
        else: 
            if curAF >= AF_THRESH: 
                is_active = True
                candidate_entry.append(amp_entry)
                start = field_val 
                end = field_val 

    if is_active == True: 
        end = amp_list[-1][df.FIELD_VAL_INDEX][field_index]

        if start == end: 
            val_range.append(start)
            candidates[start] = candidate_entry
        else: 
            val_range.append((start, end))
            candidates[(start, end)] = candidate_entry

    return val_range, candidates


'''
   Searches neighbor by changing one field at a time  
'''
def find_neighbors(parent_node, server_id, BB, proto_fields, args, AF_THRESH, num_query, per_q_budget, track_queries): 

    neighbor_list_all = [] 

    '''
        Necessary setup 
    '''
    field_search = OrderedDict(list(proto_fields.items()))
    field_names = list(proto_fields.keys()) 
    cur_query = deepcopy(parent_node.field_values)
    num_per_search = 0         
    per_server_pq = PriorityQueue()
    field_index = 0 
    #sanity_check_order(field_search, field_names)


    amp_list_all = []


    # Iterate each field at a time 

    for field_name, field_metadata in field_search.items(): #AF.critical_fields.items():
        neighbors_list = []
        print("\n\n\tSweeping a field ", field_name)



        #amp list is a list 
        amp_list, numit, num_query, track_queries = \
            sweep_field(BB, cur_query, field_index, field_metadata, field_names, server_id, num_query, per_q_budget,  track_queries, args)              
        
        print("\t\t\t Amp list for neighbors ",  amp_list)

        if field_metadata.is_int == False: 
            high_ranges, high_candidates = find_discrete_range(amp_list, field_index, cur_query, AF_THRESH )

            print("\t\t\tis discrete : TRUE" )
        else: 
            high_ranges, high_candidates = find_contiguous_range(amp_list, field_index, cur_query, AF_THRESH )
            print("\t\t\tis discrete : FALSE" )
        print("\t\t\t High Candidates ", high_candidates)

        for ranges, candidates in high_candidates.items(): 
            cur_pattern_id = parent_node.cluster_id
            picked_af_entry = candidates[0]
            field_val = copy.deepcopy(picked_af_entry[df.FIELD_VAL_INDEX])
            curAF = picked_af_entry[df.AF_INDEX]



            assert(parent_node.server_id == picked_af_entry[df.SERVER_INDEX])
            new_pattern_id  =  get_new_cluster_id( cur_pattern_id, field_index, ranges  )
            node = qnode.QueryNode(field_val, parent_node.depth + 1, curAF, parent_node.server_id , new_pattern_id )
            node.print()
            neighbors_list.append(node)

        amp_list_all = amp_list_all + amp_list
        neighbor_list_all = neighbor_list_all + neighbors_list
        field_index = field_index + 1 

        if num_query.budget  >= per_q_budget : #args[df.PER_SERVER_BUDGET]:
            return neighbor_list_all, amp_list_all 




        #If budget exceeds then exit here .. 
    print("\t\tTotal number of neighbors ", len(neighbor_list_all))
    return neighbor_list_all, amp_list_all 


'''
    Check is a pattern c1 is contained in a pattern c2 
    For a range, we represent using a tuple (lower_bound, upper_bound )
'''
def is_contained(c1, c2): 

    #Type be of numpy type. Hence, just cast to int 
    if isinstance(c1, (int, np.integer)):
        c1 = int(c1)
    if isinstance(c2, (int, np.integer)):
        c2 = int(c2)

    #If INT and not the same then its not contained 
    if type(c1) == int and type(c2) == int : 
        if c1 != c2:
            return False 
    #IF both are not string 
    elif (type(c1) == str ) and (type(c2) == str ):
        if c1 != c2: 
            return False
    #    if c1 is just a single value and c2 is a range, 
    #    we check whether  c1 is contained in c2   
    elif type(c1) == int and type(c2) == tuple and len(c2) == 2: 
        if not(c1 >= c2[0] and c1 <= c2[1]):
            return False

    #    If c1 is a range andd c2 is a value, 
    #    we check whether c1 contained in c2   
    elif type(c2) == int and type(c1) == tuple and len(c1) == 2: 
        #then do something
        if not(c1[0] == c1[1] == c2):
            return False
    #    Case when both c1 and c2 are ranges 
    elif (type(c1) == type(c2) == tuple) and (len(c1) == len(c2) == 2): 
        #then do something
        if not(c1[0] >= c2[0] and c1[1] <= c2[1]):
            return False 
    else:
        print("type c1 ", type(c1))
        print("type c2 ", type(c2))

        raise ValueError("Wrong types: in determining subset for ", c1 , " and ", c2 )  
    return True


'''
    Given the patterns found so far, check if the current pattern is 
    contained in the pattern map 
'''

def is_new_pattern(pattern_check, pattern_map):


    # Check each pattern contained in the pattern_map 

    for pattern in pattern_map: 
        patterns_found = True
        print("Checking if ", pattern_check,  "is sub or superset of ", pattern )
        
        for i in range(len(pattern)): 
            c = pattern_check[i]
            m = pattern[i]
            if is_contained(c , m): 
                donothing = 1 
                print("\t", pattern_check, "is SUBSET of ", m)
            elif is_contained(m, c):
                donothing = 1 
                print("\t", pattern_check, "is SUPERSET of ", m)
            else: 
                # if not subset of superset, pattern is NOT foud 
                patterns_found = False; 
                break 
        if patterns_found :
            print("\n\t NOT a new pattern") 
            return False, pattern 
    print("\n\t Is a new pattern  ", pattern_check , " not contained in ", pattern_map )
    return True, None




def pick_random_query(fields): 
    field_vals = []
    query = OrderedDict() 
    for field_name, field_obj in fields.items():
        random_value = field_obj.accepted_range[random.randint(0, len(field_obj.accepted_range) - 1)]
        query[field_name] = random_value
    return query



