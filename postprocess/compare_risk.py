import pandas as pd
import numpy as np
import scipy.stats as ss
import os
import seaborn as sns
import itertools
from collections import OrderedDict
import json, glob 
import argparse




parser = argparse.ArgumentParser()#help="--fields_path , --data_folder_name  --proto  ")
parser.add_argument('--risk_dir', type=str, default="risk_out")#, required=True)

args = parser.parse_args()
print(args)





root_dir=args.risk_dir #"risk_quantification_20200524"

folders = glob.glob(root_dir + "/*")

known_risk = {}
known_risk["dns"] = 28.7 
# known_risk["dns-nodnssec"] = 28.7 
# known_risk["ntpprivate"] = 556.9 
# known_risk["ntpprivate-and"] = 556.9 


# known_risk["snmpnext"] = 0 
# known_risk["snmpget"] = 0 
# known_risk["snmpbulk"] = 6.4

# known_risk["snmpnext-and"] = 0 
# known_risk["snmpget-and"] = 0 
# known_risk["snmpbulk-and"] = 6.4


# known_risk["ssdp"] = 30.8

# known_risk["memcached"] = 10000  
# known_risk["chargen"] = 358.8 
num_servers = 10000 

known_key = 'known_pattern_total_risk'
new_key = "new_pattern_total_risk"
lol = [] 
for folder in folders: 

    basefile = os.path.basename(folder)
    proto = basefile.split("_")[1]

    entry = OrderedDict() 
    entry["file"] = basefile 
    entry["proto"] = proto 
    if proto in known_risk: 

        #if proto == "ntpprivate-and": 
        #    known_pat_risk = known_risk[proto] *  num_servers / 3083
        #else: 
        known_pat_risk = known_risk[proto] * num_servers
            
        entry["known_risk"] = known_pat_risk 

        known_risk_f = os.path.join(folder, "known_pattern_total_risk.npy")
        if  os.path.exists( known_risk_f ):
            data = np.load(known_risk_f, allow_pickle=True).item()
            known_real_risk = data[known_key]
            #known_real_risk = known_real_risk * num_servers 
            entry["known_real_risk"] = known_real_risk 
        
            if proto == "ntpprivate-and": 
                entry["known_real_risk"] = entry["known_real_risk"] * num_servers / 3083 
                
    
        
        new_risk_f = os.path.join(folder, "new_pattern_total_risk.npy")

        if os.path.exists(new_risk_f): 
            new_risk = np.load(new_risk_f, allow_pickle=True).item()[new_key]
            #new_risk = new_risk * num_servers 
            entry["new_real_risk"] = new_risk   
            if proto == "ntpprivate-and": 
                entry["new_real_risk"] = entry["new_real_risk"] * num_servers / 3083 

        lol.append(entry)

risk_df = pd.DataFrame(lol)
#
risk_df.to_csv(os.path.join(root_dir, "summary_comparison.csv ")) 
