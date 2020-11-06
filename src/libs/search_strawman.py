import numpy as np
import time
import itertools
from queue import PriorityQueue
import argparse
from copy import deepcopy
from collections import OrderedDict
import string, random, math 
from datetime import datetime
import sys, json, os
import struct
import configparser
import logging
import collections
import queue
import platform


import libs.definition as df
import libs.budget as budget
import libs.query_node as qnode
import libs.inputs as inputs
import libs.query_json as query_json
import libs.helper as helper


from hyperopt import fmin, tpe, hp, STATUS_OK,Trials,anneal
import hyperopt 



def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def f1( space):
    global time_sleep 

    space1 = collections.OrderedDict(space)


    newdict={}
    for k1,v1 in space1.items():

        if isinstance(v1, str):
            newdict[k1]=v1
        else:
            newdict[k1]=int(v1)

    amp_factor=simAF.get_af_dict( server_ip,  newdict)
    time.sleep(time_sleep)


    print(newdict,amp_factor)
    return {'loss':(-1.0*amp_factor),
             'status': STATUS_OK,
        'String': {'value': space
                  }
           }


simAF={}
server_ip={}

def generate_simulated_queries(BB, proto_fields, measurer_ip, args):
    global simAF
    simAF=BB
    global server_ip

    global time_sleep 

    print(proto_fields)
    phase = "random"
    config = configparser.ConfigParser()
    config.read("common_path.ini")
    query_out_dir = os.path.join(config["common_path"]["query_out_dir"], measurer_ip)

    print(query_out_dir)

    if not os.path.exists(query_out_dir):
        os.makedirs(query_out_dir)

    proto = args[df.PROTO]
    num_rand_queries = args[df.PER_SERVER_RANDOM_SAMPLE]
    print("num random queries ", num_rand_queries)
    server_ip = args["server_ip"]
    time_sleep = float(args[df.TIME_SLEEP])
    buffer_query = int(args["update_db_at_once"])


    queryBuffer = []
    simulated_space={}
    for f, finfo in proto_fields.items(): 
        ar=finfo.accepted_range
        e_str=ar[0]
        e_end=ar[len(ar)-1]
        print(e_end,e_str)
        if type(e_end) is str:
            print("STRING",ar)
            list_ap=ar
            simulated_space[f]=hp.choice(f,ar)
        else:

            len1=e_end-e_str

            if(len1 >=  df.SMALL_FIELD_THRESHOLD):
                print("VERY LARGE")
                simulated_space[f]=hp.quniform(f,e_str,e_end,100)
            else:
                simulated_space[f]=hp.choice(f,finfo.accepted_range)


        print(f,vars(finfo),ar,e_str,e_end)
        print(simulated_space[f])

    print(simulated_space,len(simulated_space))
    per_server_budget=args["per_server_budget"]



    if ('init_trials' in args):
        points_to_evaluate = args['init_trials']
        print("OLD Trials: \n\n",points_to_evaluate,len(points_to_evaluate),"\n\n"," DONE")
        lp=len(points_to_evaluate)

    else:
        points_to_evaluate=None
        lp=0
    rand_budget= 0
    trials1 = Trials()

    best = fmin(fn=f1,space=simulated_space,points_to_evaluate=points_to_evaluate,algo=hyperopt.anneal.suggest,\
        max_evals=per_server_budget ,trials=trials1)

    pq = PriorityQueue()  


    for ind_ele in trials1.results:
        loss=ind_ele['loss']*-1
        print(ind_ele['String']['value'],loss)
        ll=[]

        field_values = ind_ele['String']['value']
        af =  loss
        insert_data = gen_query_buffer_entry(field_values, af, server_ip, measurer_ip , 'SA' )
        print("Insert data ", insert_data) 
        queryBuffer.append(insert_data)


    if len(queryBuffer) != 0 : 
        print("updating query buffer with len ", len(queryBuffer))
        query_out_filename = os.path.join(query_out_dir, server_ip)
        query_json.write_to_json(queryBuffer, query_out_filename)

        queryBuffer.clear()  

    return None 
 


