from itertools import permutations
from copy import deepcopy
import numpy as np
import argparse, sys, random, heapq
from queue import PriorityQueue 
import networkx as nx
import os, time, math
import libs.master as master
import libs
import configparser


def __make_out_dirs(out_dir):
    dirs = [ "searchout","logs", "command" , "query", "pcap", "gt", "pcap_temp"]
    paths = [os.path.join(out_dir, d) for d in dirs]

    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)

def main(args): 
	# below is for sim 
    model_inputs_path = os.path.join(os.getcwd(), args.in_dir)
    inputs_path = os.path.join(os.getcwd(), args.in_dir)
    __make_out_dirs(args.out_dir)


    # Running ampmap algorithm 
    if args.measurement: 
        libs.master.start_measurement_batch(args, inputs_path)

    # Running strawman solution 
    elif args.simulated :
        libs.master.start_simulated_batch(args,inputs_path)
    else: 
        msg = 'One of --measurement, --simuated, or --simulated_random needs to be set' 
        raise ValueError(msg) 

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="available --coordinate, --custom, --random , --random_accepted")
    parser.add_argument('--out_dir', type=str, default='out') 



    parser.add_argument('--in_dir', type=str, default='field_inputs_dns') 


    parser.add_argument('--proto', type=str, default='DNS') 
    parser.add_argument('--measurement' , action='store_true', default=True)


    parser.add_argument('--title', type=str, default='test_run') 


    '''
        Flags related to strawman solutions 
    '''
    parser.add_argument('--random_accepted', action='store_true' , default=False) 
    parser.add_argument('--simulated', action='store_true', default=False)
    parser.add_argument('--simulated_random', action='store_true', default=False)



    '''
        Budget-related parameters  
    '''
    parser.add_argument('--per_server_budget', type=int, default=100) 
    parser.add_argument('--per_server_random_sample', type=int, default=1000) 
    parser.add_argument('--num_probe', type=int, default=75 )


    '''
        Flags related to how to choose starting points for the per-field search 
    '''
    parser.add_argument('--choose_K', type=int , default=1)
    parser.add_argument('--choose_K_max_val', action='store_true', default=False)
    parser.add_argument('--choose_K_max_dist', action='store_true', default=False)
    parser.add_argument('--choose_K_random', action='store_true', default=False)
    # We always use default 20 for this value  
    parser.add_argument('--probe_numcluster', type=int, default=20)


    '''
        Time wait for query and the size of block to update intermediate logging  
    '''  
    parser.add_argument('--query_wait', default=5)
    parser.add_argument('--update_db_at_once', default=200)


    # server timeout 
    parser.add_argument('--server_timeout', type=float, default=2)


    # the minimum threshold AF to denote high AF queries 
    parser.add_argument('--minAF', default=10)



    '''
        Timeout for different stages 
    '''
    parser.add_argument('--random_timeout', default=2)
    parser.add_argument('--probe_timeout', default=1)
    parser.add_argument('--pfs_timeout', default=2)


    # clustering choices 
    parser.add_argument('--cluster_equal_weight', action='store_true', default = False)
    parser.add_argument('--cluster_weight_based', action='store_true', default = False)
    parser.add_argument('--cluster_weight_hybrid', action='store_true', default = False)


    # batch processing
    parser.add_argument('--block_size', default=5)


    # Parameters below are rarely used (used for testing/analysis purposes)
    parser.add_argument('--multiply_random_factor', type=int, default=None)
    parser.add_argument('--server_ip', type=str)
    parser.add_argument('--url', type=str, default=None)
    parser.add_argument('--disable_check_new_pattern', action='store_true', default = False)
    parser.add_argument('--disable_sampling', action='store_true', default = False)
    parser.add_argument('--pcap_dump', default=False)


    # For cmu local exp. Yucheng 5/14/2020
    parser.add_argument('--local_exp_cmu', default=False, action="store_true")
    

    args = parser.parse_args()


    main(args) 
