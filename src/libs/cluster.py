import json
import numpy as np
import random, csv, math 
from collections import OrderedDict
from queue import PriorityQueue
import argparse, os
import time
from textwrap import wrap
import subprocess
import os, sys
import libs.inputs as inputs
import shutil
import random
from shutil import copyfile
import libs.query_json as query_json
import configparser
import libs.definition as definition
from copy import deepcopy
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import random
import logging



import libs.search_common as search_common

from nltk.cluster import KMeansClusterer, euclidean_distance


def align_fields(dict_to_align, field_info): 
    lst = [] 
    for fid, f_metadata in field_info.items(): 
        lst.append(dict_to_align[fid] ) 
    return lst

def parse_json(f, field_info, minAF): 
    json_data = query_json.read_json(f)


    data_list = [] 
    amp_list = [] 
    for entry in json_data: 
        fv = align_fields(entry["fields"], field_info)
        amp_factor = round(float(entry["amp_factor"]), 4) 
        server_ip = entry["server_ip"]

        if amp_factor >= minAF: 
            data_list.append(fv)
            amp_list.append(amp_factor)
            #:qprint(fv)
    return data_list, amp_list

def iterate_files(base_dir, minAF, field_info ):
    lol = []
    feature_id = []

    print(base_dir)
    data_list = []
    amp_list = []
    for root, dirs, files in os.walk(base_dir):
        for sub_dir  in dirs: 
            path = os.path.join( root,  sub_dir )
            json_files = os.listdir(path)
            for f in json_files: 
                file_path = os.path.join( path, f )
                data, amp  = parse_json(file_path, field_info, minAF)
                data_list = data_list + data 
                amp_list = amp_list + amp 
    assert(len(amp_list) == len(data_list))

    return data_list, amp_list


def parse_data(ampdata , proto_fields, minAF):

    data_list = []
    amp_list = []
    for server_ip, server_data  in ampdata.items():  

        for entry in server_data: 
            amp_factor = round(float(entry["amp_factor"]), 4)  #server_data["amp_factor"]
            fv = align_fields(entry["fields"], proto_fields)

            if amp_factor >= minAF: 
                data_list.append(fv)
                amp_list.append(amp_factor)
    assert(len(amp_list) == len(data_list))
    return data_list, amp_list    


def convert_dataframe_to_list(df ): 
    data = [] 
    for index, row in df.iterrows():
        data.append(row.tolist() ) 

    return data



''' 
    Normalizing each field 
    as some are large vs. small, discrete vs. continuous

'''
def normalize_field(df, proto_fields, numBin = 10 ): 

    print("proto fields ", proto_fields)

    for c in df.columns: 
        print("c is " , c , " and df.columns is ", df.columns)
        logging.info("\nFIELD c is {} and  and df.columns is {}".format(c,df.columns)) 

        f_metadata = proto_fields[c]
        


        ''' 
            Field only takes one value so normalize to all 1             
        ''' 
        if len(f_metadata.accepted_range) == 1: 
            print("Field type 0 ", c , type(f_metadata.accepted_range))
            

            val_max = 1 
            logging.info("\tvalue max is {}".format(val_max))
            df[c] = pd.Series(df[c]).astype('category')
            df[c] = df[c].cat.codes

        #    Field takes a set of discrete values 
        #    Then it is a categorical field        
        elif f_metadata.is_int == False or type(f_metadata.accepted_range) != range :  
            print("Field type 1 ", c , type(f_metadata.accepted_range))
            logging.info("Field type 1 {} : {}".format(c, type(f_metadata.accepted_range)))


            val_max = len(f_metadata.accepted_range) - 1 

            df[c] = pd.Series(df[c]).astype('category')
            df[c] = df[c].cat.codes


        #    For field that takes a range of contiguous values, accordingly bin the values 
        elif len(f_metadata.accepted_range) > definition.SMALL_FIELD_THRESHOLD: 
            print("Large range is ", c)
            print("Field type 2 ", c , type(f_metadata.accepted_range))
            


            logging.info("Field type 2 {} : {}".format(c, type(f_metadata.accepted_range)))
            bin_size = math.ceil(f_metadata.accepted_range[-1] / numBin) 
            logging.info("Bin size is {}".format(bin_size))
            print("bin size ", bin_size)
            df[c] = df[c]//bin_size
            val_max = max(df[c])
            logging.info("\tvalue max is {}".format(val_max))
            print("\tvalue max is {}".format(val_max))
        #    For all others, 
        else: 
            print("Field type 3 ", c , type(f_metadata.accepted_range))
            

            logging.info("Field type 3 {} : {}".format(c, type(f_metadata.accepted_range)))
            val_max = max(f_metadata.accepted_range)

            #print(df[c])
        df[c] = df[c].astype(float)
        
        
        #    Corner case: a field takes a value of 0 only 
        if len(f_metadata.accepted_range) == 1 and f_metadata.accepted_range[0] == 0: 
            print("Field ", c , " is NOT normalized ") 
            continue            

        # Make the range betwen n 0 to 1         
        if val_max != 0:  
            df[c] = df[c]/val_max * 1
    return df
















def pick_representative_queries_weight_based(query_to_cluster, num_probe, num_cluster): 
    num_cluster = len(query_to_cluster)

    num_spent = 0 
    cluster_ids = list(query_to_cluster.keys())

    queries_final = []


    '''
        If size of cluster is greater than probe, 
        then make sure we pick one sampel from each cluster 
    ''' 
    while num_spent < num_probe and len(query_to_cluster) > 0 : 
        cluster_ids = list(query_to_cluster.keys()) 

        weights, scaled_weights = compute_weights(query_to_cluster)


        cid = random.choices( cluster_ids, weights=weights , k=1)[0]

        rand_index = random.randint(0, len(query_to_cluster[cid]) -1 )
        
        picked_query = query_to_cluster[cid][rand_index]
        query_to_cluster[cid].pop(rand_index)

        queries_final.append(picked_query)
        num_spent = num_spent + 1 

        if len(query_to_cluster[cid]) == 0: 
            del query_to_cluster[cid]
        if len(query_to_cluster) == 0: 
            print("Exising as all queries used")
            break 
    return queries_final







def pick_representative_queries_hybrid_weight(query_to_cluster, num_probe, num_cluster): 

    num_spent = 0 

    cluster_ids = list(query_to_cluster.keys())
    
    queries_final = []
    
    '''
        If size of cluster is greater than probe, then we pick X from the first NUM_PROBE clusters 
    '''
    if num_cluster >= num_probe: 
        print("Num cluster > Num Probe so pick random cluster ..  ")

        # sample WITHOUT replacement 
        chosen_cids = random.sample(cluster_ids, num_probe) 
        for cid in chosen_cids: 
            rand_index = random.randint(0, len(query_to_cluster[cid]) -1 )
            picked_query = query_to_cluster[cid][rand_index]
            queries_final.append(picked_query)

            query_to_cluster[cid].pop(rand_index)
        print("picked 1 from each cluster")
        return queries_final  
   

    '''
        Sample based on the weight (size) of each cluster 
    '''
    #WEIGHT BASED 
    print('WEIGHT based')
    while num_spent < num_probe and len(query_to_cluster) > 0 : 
        cluster_ids = list(query_to_cluster.keys()) 

        weights, scaled_weights = compute_weights(query_to_cluster)
        cid = random.choices( cluster_ids, weights=weights , k=1)[0]


        rand_index = random.randint(0, len(query_to_cluster[cid]) -1 )
        
        picked_query = query_to_cluster[cid][rand_index]
        query_to_cluster[cid].pop(rand_index)

        queries_final.append(picked_query)
        num_spent = num_spent + 1 

        if len(query_to_cluster[cid]) == 0: 
            del query_to_cluster[cid]
        if len(query_to_cluster) == 0: 
            print("Exising as all queries used")
            break 
    return queries_final





'''
    Pick samples and give equal weight to each cluster 
'''
def pick_representative_queries_equal_weight(query_to_cluster, num_probe, num_cluster): 

    num_spent = 0 

    cluster_ids = list(query_to_cluster.keys())
    
    queries_final = []
    

    '''
        If size of cluster is greater than probe, then we pick X from the first NUM_PROBE clusters 
    '''
    if num_cluster >= num_probe: 
        print("Num cluster > Num Probe so pick random cluster ..  ")

        #sample WITHOUT replacement 
        chosen_cids = random.sample(cluster_ids, num_probe) 
        for cid in chosen_cids: 
            rand_index = random.randint(0, len(query_to_cluster[cid]) -1 )
            picked_query = query_to_cluster[cid][rand_index]
            queries_final.append(picked_query)

            query_to_cluster[cid].pop(rand_index)
        return queries_final  
         
    '''
        Pick one from each other ..  
    '''

    del_keys = set() 
    for cid, queries in query_to_cluster.items() :
        rand_index = random.randint(0, len(queries) -1 )

        picked_query = query_to_cluster[cid][rand_index]
        queries_final.append(picked_query)

        query_to_cluster[cid].pop(rand_index)
        num_spent = num_spent + 1 
        
        if len(query_to_cluster[cid]) == 0: 
            del_keys.add(cid )


        if num_spent >= num_probe:
            break 
    print("picked {} so far ".format(num_spent) )
    print("Deleting cluster IDs that are empty {}  ".format(del_keys) )

    # Delete cluster ID if we picked all from that cluster  
    for cid in del_keys: 
        del query_to_cluster[cid]
    

    
    while num_spent < num_probe and len(query_to_cluster) > 0 : 
        cluster_ids = list(query_to_cluster.keys()) 
        
        cid =    random.choice( cluster_ids ) 
        rand_index = random.randint(0, len(query_to_cluster[cid]) -1 )
        picked_query = query_to_cluster[cid][rand_index]
        query_to_cluster[cid].pop(rand_index)

        queries_final.append(picked_query)
        num_spent = num_spent + 1 

        if len(query_to_cluster[cid]) == 0: 
            del query_to_cluster[cid]
        if len(query_to_cluster) == 0: 
            print("Exising as all queries used")
            break 
    return queries_final



def compute_weights(query_to_cluster): 
    weights = [] 
    total_weight = 0 
    for cid, queries in query_to_cluster.items(): 
        weights.append(len(queries))
        #print(len(queries))
        total_weight = total_weight + len(queries)
    scaled_weights = []
    for w in weights: 
        scaled_weights.append( w/total_weight )

    return weights, scaled_weights 






def pick_representative_queries_new(query_to_cluster, num_probe): 

    num_spent = 0 

    cluster_ids = list(query_to_cluster.keys())
    
    queries_final = []

    '''
        if size of cluster is greater than probe, then we pick X from the first NUM_PROBE clusters 
    '''
    if num_cluster >= num_probe: 
        print("Num cluster > Num Probe SO pick at least one sample ")
        
        
        chosen_cids = random.sample(cluster_ids, num_probe) 
        #print(sorted(chosen_cids) ) # len(chosen_cid), len(set(chosen_cid)))
        for cid in chosen_cids: 
            rand_index = random.randint(0, len(query_to_cluster[cid]) -1 )

            picked_query = query_to_cluster[cid][rand_index]

            queries_final.append(picked_query)
            #Debugged this problem -- May 29, SOOJIN MOON 
            #queries_final.append(picked_query[0])
            #Delete the picked queries 
            query_to_cluster[cid].pop(rand_index)
        return queries_final  
         

    '''
        If num_cluster < num_probe : 
        Pick one from each other .. 
    '''
    for cid, queries in query_to_cluster.items() :
        rand_index = random.randint(0, len(query_to_cluster[cid]) -1 )

        picked_query = query_to_cluster[cid][rand_index]
        queries_final.append(picked_query)

        #Delete the chosen  queries 
        query_to_cluster[cid].pop(rand_index)
        num_spent = num_spent + 1 
        if num_spent >= num_probe:
            break 
    print("picked {} so far ".format(num_spent) )
    

    while num_spent < num_probe: 
        cluster_ids = list(query_to_cluster.keys()) 

        weights, scaled_weights = compute_weights(query_to_cluster)
        cid = np.random.choice( cluster_ids, 1, p=scaled_weights)[0]
        
        rand_index = random.randint(0, len(query_to_cluster[cid]) -1 )
        picked_query = query_to_cluster[cid][rand_index]
        query_to_cluster[cid].pop(rand_index)

        queries_final.append(picked_query)
        num_spent = num_spent + 1 

        if len(query_to_cluster[cid]) == 0: 
            del query_to_cluster[cid]
        if len(query_to_cluster) == 0: 
            print("Exising as all queries used")
            break 
    return queries_final








def pick_representative_queries(query_to_cluster, num_probe): 
    num_cluster = len(query_to_cluster)

    num_spent = 0 


    queries_final = []
    

    '''
        If size of cluster is greater than probe, then 
        we pick one sampel from each cluster 
    '''

    if num_cluster >= num_probe: 
        print("Num cluster > Num Probe SO pick at least one sample ")
        #pick at least 
        print(query_to_cluster)
        for cid, queries in query_to_cluster.items() :
            #print(query_to_cluster)
            rand_index = random.randint(0, len(query_to_cluster[cid]) -1 )
        
            picked_query = query_to_cluster[cid][rand_index]


            queries_final.append(picked_query)
            #Delete the picked queries 
            query_to_cluster[cid].pop(rand_index)
            print(" Picked id ", num_spent  ,  ":" ,picked_query[0], picked_query[2],  "with cid ", cid )
            num_spent = num_spent + 1 




    while num_spent < num_probe: 
        cluster_ids = list(query_to_cluster.keys()) 

        weights, scaled_weights = compute_weights(query_to_cluster)
        cid = np.random.choice( cluster_ids, 1, p=scaled_weights)[0]

        rand_index = random.randint(0, len(query_to_cluster[cid]) -1 )
        
        picked_query = query_to_cluster[cid][rand_index]
        print(" ID " , num_spent , ":" , picked_query[0] , picked_query[2], "with cid ", cid )
        query_to_cluster[cid].pop(rand_index)

        queries_final.append(picked_query)
        num_spent = num_spent + 1 

        if len(query_to_cluster[cid]) == 0: 
            del query_to_cluster[cid]
        if len(query_to_cluster) == 0: 
            print("Exising as all queries used")
            break 
    return queries_final












def convert_to_dict(chosen_queries, proto_fields):

    field_names = list(proto_fields.keys())
    queries_dict = [] 



    for entry in chosen_queries: 
        #print("entry " ,entry)
        q = deepcopy(entry)
        #print("q is ", q)
        #print("field name is ", field_names)
        q_dict = OrderedDict() 
        for i in range(len(field_names)):
            # Yucheng: 2/4/2019, maybe buggy
            #print("q is ", q)
            q_dict[field_names[i]] = q[0][i]

            # q_dict[field_names[i]] = q[i]
        queries_dict.append(q_dict)
    return queries_dict



'''
    Clustering queries with above minAF AFs 
    Returns the chosen queries used for probing 

''' 
def cluster(ampdata, proto_fields, num_cluster, minAF, num_queries, is_measurement, args, config, log_file=None):
    minAF = float(minAF)


    '''
        Logging features 
    '''
    print("In clustering")
    if log_file == None:  
        log_file = os.path.join( config["common_path"]["log_out_dir"], "cluster.log") 

    logging.basicConfig(filename=log_file,format='%(levelname)s  :  %(asctime)s \t %(message)s', \
        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)

    logging.info("In clustering")



    '''
        Only fetch those queries with AF >= minAF 
    '''

    data_list, amp_list = parse_data(ampdata , proto_fields, minAF)
    assert(len(data_list) == len(amp_list))

    print("Length of data is {}".format(len(data_list)))
    
    logging.info("Length of data is ", len(data_list))
    if len(data_list) == 0: 
        return [] 

    
    amp_data_frame = pd.DataFrame(data_list)
    amp_data_frame.columns = list(proto_fields.keys())
    
    logging.info("Generated data frame")

    '''
        Normalize each field before running the clustering 
    '''
    amp_data_norm = normalize_field( amp_data_frame, proto_fields, numBin=10 )
    amp_data_norm_list = convert_dataframe_to_list(amp_data_norm )
    print("\n\nLength of data is ", len(amp_data_norm_list))
    logging.info("\n\nLength of data is {}".format(len(amp_data_norm_list)))

    
    # Just for debugging purposes 
    for i in amp_data_norm_list: 
        logging.info(" in amp data norm list {}".format(i) )
        print(i)



    logging.info("Original num cluster is {} ".format( num_cluster))
    num_cluster = min(num_cluster, len(data_list))
    logging.info("Num cluster is {} ".format( num_cluster))



    kmeans = KMeans(n_clusters=num_cluster).fit(amp_data_norm_list)
    logging.info("generated KMEANS CLUSTER")
    logging.info("Inertia: {}".format(kmeans.inertia_))


    '''
        Assign cluster ID (label) to each query 
    '''
    query_to_cluster = OrderedDict() 
    for l in set(kmeans.labels_): 
        query_to_cluster[l] = []

    for i in range(len(amp_data_norm_list)): 
        c = kmeans.labels_[i] 
        query_to_cluster[c].append((data_list[i], amp_data_norm_list[i], amp_list[i]))
    


    
    if args.cluster_weight_hybrid: 

        logging.info("Cluster : Picking Queries with HYBRID strategy (equal weight and nOT equal weight ) ")
        chosen_queries = pick_representative_queries_hybrid_weight(query_to_cluster, num_queries, num_cluster)

    elif args.cluster_equal_weight: 
        logging.info("Cluster : Picking Queries with Equal Weight ")
        chosen_queries = pick_representative_queries_equal_weight(query_to_cluster, num_queries, num_cluster )
    elif args.cluster_weight_based: 
        logging.info("Cluster : Picking Queries with Weight Based ")
        chosen_queries = pick_representative_queries_weight_based(query_to_cluster, num_queries, num_cluster)

    else: 
        logging.info("Cluster : (default) Picking Queries with HYBRID strategy ")
        chosen_queries = pick_representative_queries_hybrid_weight(query_to_cluster, num_queries, num_cluster)

    
    chosen_queries_dict = convert_to_dict(chosen_queries, proto_fields) 
    logging.info("Chose {} probing queries ".format( len(chosen_queries_dict)))



    for i in chosen_queries_dict:   
        print("chosen queries ", i )
        logging.info( "Chosen queries {}".format(i)  )

    return chosen_queries_dict




