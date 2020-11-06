import pandas as pd
import numpy as np
import scipy.stats as ss
import os
import seaborn as sns
import argparse 
import json


from collections import OrderedDict
import itertools
from copy import deepcopy 


parser = argparse.ArgumentParser()#help="--fields_path , --data_folder_name  --proto  ")
#parser.add_argument('--proto', type=str, default="dns")#, required=True)
#parser.add_argument('--plot_root_dir', type=str, default="./qp_plots")#, required=True)
parser.add_argument('--qp_dir', type=str, default="./qps_june/out_DNS_10k_may29/" )#, required=True)
parser.add_argument('--out_dir', type=str, default="./risk_quantification/out_DNS_10k_may29/")#, required=True)
parser.add_argument('--sig_input_file', type=str, default="./known_patterns/dns.json")#, required=True)
parser.add_argument('--proto', type=str, default="dns")#, required=True)
parser.add_argument('--match_all_data', default=False, action='store_true')


args = parser.parse_args()

proto = args.proto
sig_dir=args.sig_input_file


if args.match_all_data: 
    sig_input = {}
else:
    with open(sig_dir , 'r') as f:
        sig_input = json.load(f)
        

complete_filename = os.path.join( args.qp_dir,  "complete_info.csv") 


qp_dir =  args.qp_dir #  "./qps_june/out_DNS_10k_may29/"
out_dir = args.out_dir 
#out_dir = "./risk_quantification/out_DNS_10k_may29/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

complete_queries = pd.read_csv(complete_filename)

print("done loading complete queries")
uniq_servers = np.unique(complete_queries["server_id"])



# sig_input =  dict() 

# sig_input["edns"] =  [0 ]
# sig_input["rdatatype"] =  [255]


# if proto.lower() == "memcached":
#     #ALL queries 
#     q = complete_queries.query("command == 'x'" )    #["start_line"].
#     q.to_csv(os.path.join(out_dir, "removed_data.csv") )
#     #print(q.shape)
#     complete_queries = complete_queries.drop(q.index.values ).reset_index()

#     print("Removed certain data ", q.shape ) 

# elif proto.lower() == "ssdp":
#     #ALL queries 
#     q = complete_queries.query("start_line == 'M'" )    #["start_line"].
#     q.to_csv(os.path.join(out_dir, "removed_data.csv") )
#     #print(q.shape)
#     complete_queries = complete_queries.drop(q.index.values ).reset_index()

#     print("Removed certain data ", q.shape ) 

    
df = complete_queries 







## translate into possible match signature

if args.match_all_data:
    sig = []
else: 
    if len(sig_input) != 0: 
        keys, values = zip(*sig_input.items())
        sig =  [dict(zip(keys, v)) for v in itertools.product(*values)]    
        print(sig)








print("DF shape" , df.shape)
#For each sig .. filter the data 
index_store =  [] 
for i,s in enumerate(sig):
    if i % 1000 == 0 : 
        print("processed ", i , " servers")
    #Get the data that match signature 
    print("s is ", s)
    df_temp = df.loc[(df[list(s)] == pd.Series(s)).all(axis=1)]
    print("df temp shape ", df_temp.shape)
    index_store.extend( df_temp.index.values )
    
#Should be uniq index 
assert(len(index_store) == len(np.unique(index_store)))
df_match = df.iloc[index_store].reset_index() 

if args.match_all_data: # "snmp"  in proto: 
    print("match all data")
    df_match = df


df_match.to_csv( os.path.join( out_dir, "known_pattern_data.csv" ))


print("DF shape ", df.shape )
print("DF match shape ", df_match.shape )


#For each serveer, find the the risk 
orig_risk = 0   
server_risk_known =  []#  dict()


risk = df_match.groupby(["server_id"])["amp_fac"].max().to_dict()

#print(risk)

#print("Orig risk" , orig_risk)
#print(server_risk_orig_map)
server_risk_known_pd = pd.DataFrame.from_dict(risk, orient='index',  columns=['amp_fac'])
server_risk_known_pd.reset_index( inplace=True)
server_risk_known_pd.columns = ["server_id", "amp_fac"]

orig_risk = server_risk_known_pd["amp_fac"].sum() 
    
server_risk_known_pd.to_csv( os.path.join( out_dir, "known_pattern_server_specific_risk.csv" ), index=True)

known_pattern = sig_input 

np.save(os.path.join( out_dir, "known_pattern.npy") , known_pattern)

print(server_risk_known_pd)
#a = np.load(os.path.join( out_dir, "known_pattern.np")) 
#print(a.item())


summary = dict() 
summary["known_pattern_total_risk"] = orig_risk 
summary["num_total_server"] = len(np.unique(server_risk_known_pd["server_id"])) 
np.save( os.path.join( out_dir , "known_pattern_total_risk.npy") , summary)








def generate_all_possible_sig(): 
    possible_sig = dict()
    for field, val in sig_input.items():
        #print("field ",  field, "val " , val)
        possible_sig[field] = list(np.unique( df[field] ))
    keys, values = zip(*possible_sig.items())
    a =  [tuple(zip(keys, v)) for v in itertools.product(*values)]    
    #print(a )
    return a 
    

def construct_all_sig(df, all_possible,  sig_match ):
    
    removed = deepcopy(all_possible) 
    for sig in sig_match: 
        print("sig ", sig)
        sig_tuple = tuple(sig.items()  )  
        print("sig tuple ", sig_tuple)

        removed.remove(sig_tuple )
        #a.remove(sig_tuple) 
    return removed
    
    


#sig exclude match stores all signatures that are NOT matched 


if args.match_all_data: 
    print("Finished job for matching all data")
    sys.exit()


    
all_possible_sig = generate_all_possible_sig(  )
print(len(all_possible_sig))
sig_exclude_match = construct_all_sig(df, all_possible_sig, sig  )
print(len(sig_exclude_match))





lol = []

sig_alias = dict() 
for i, sig in enumerate(sig_exclude_match):
    sig_dict = dict(sig)
    sig_alias[i] = sig_dict 
    sig_df = df.loc[(df[list(sig_dict)] == pd.Series(sig_dict)).all(axis=1)]

    server_to_risk = sig_df.groupby(["server_id"])['amp_fac'].max().to_dict() 

    lol.append(server_to_risk )

new_pattern_pd = pd.DataFrame(lol)
new_pattern_pd 
    

summary = dict() 
summary["new_pattern_total_risk"] = new_pattern_pd.sum().sum() 
summary["num_total_server"] = len(new_pattern_pd.columns) 
np.save( os.path.join(out_dir , "new_pattern_total_risk.npy"), summary)


print(out_dir)

new_pattern_pd.to_csv( os.path.join(out_dir , "new_pattern_data.csv") , index=True )


#shows the server specific risk 
server_specific_risk = new_pattern_pd.sum().reset_index()
server_specific_risk.columns = ["server_id", "risk"]
server_specific_risk.to_csv(os.path.join(out_dir , "new_pattern_server_specific_risk.csv")  )



np.save( os.path.join(out_dir , "new_pattern_sig_alias.npy"), sig_alias)




