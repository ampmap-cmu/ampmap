
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import scipy.stats as ss
import os
#import matplotlib.pyplot as plt


import matplotlib
#matplotlib.get_backend()
from matplotlib import pyplot as plt
import seaborn as sns

#import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt 
import json
import os
import ast
from scipy.spatial import distance
import argparse
from collections import Counter, OrderedDict
from operator import itemgetter
#from sklearn.metrics import jaccard_similarity_score
#from sklearn.metrics import jaccard_score






parser = argparse.ArgumentParser()#help="--fields_path , --data_folder_name  --proto  ")
parser.add_argument('--proto', type=str, default="dns")#, required=True)
parser.add_argument('--proto_folder', default=None)#, required=True)

parser.add_argument('--plot_root_dir', type=str, default="./qp_plots")#, required=True)
parser.add_argument('--qp_dir', type=str, default="./qps/out_DNS_10k_query_searchout_May11dns_sec")#, required=True)
parser.add_argument('--depth_pq_file', type=str, default="dnssec_DAG_QPS_depth_median.npy")#, required=True)
#parser.add_argument('--depths', type=str, default="dnssec_DAG_QPS_depth_median.npy")#, required=True)

parser.add_argument('--depth', nargs='+',  type=int, help='<Required> depth flag', required=True)
parser.add_argument('--max_plot_similarity',  action='store_true',  default=False)
parser.add_argument('--all_plot_similarity',  action='store_true',  default=False)

parser.add_argument('--DNSSEC_True', action='store_true',  default=False)

#parser.add_argument('--intermediate_data_folder', type=str, default="./intermediate_data")
#parser.add_argument('--aggregate_summary',  default=False, action='store_true')
args = parser.parse_args()

print(args)


# CHANGE these names when generating new data/protocol/signature
# Queries_filename = 'out_dns1kdns_sec-1.csv'

plot_root_dir = args.plot_root_dir# "./qp_plots"
#qp_dir = "."#"/Users/soojin/Google Drive/Research/AmpMap/Eval_Current/MeasurementOut/QueryPattern/out_DNS_10k_query_searchout_May11dns_sec"
qp_dir = args.qp_dir # "./qps/out_DNS_10k_query_searchout_May11dns_sec"
#qp_dir =   "./qps/out_dns1kdns_sec/"#   "./qps/out_DNS_10k_query_searchout_May11dns_sec"




# PERCENTILE = 98

#######Relevant file 
Queries_filename = os.path.join( qp_dir,  "ALL_Queries.csv") 
sig_filename = os.path.join(qp_dir, 'sigs.npz') 
#depth_pq_file = os.path.join(qp_dir,"dnssec_DAG_QPS_depth_median.npy" )
depth_pq_file = os.path.join(qp_dir, args.depth_pq_file )

domain_dnssec = ['berkeley.edu', 'energy.gov', 'aetna.com', 'Nairaland.com']
depth_minus1_file = os.path.join( qp_dir,  "Hamming_2.csv") 


# Queries_filename = os.path.join( qp_dir, 'ALL_Queries.csv') 
# sig_filename = os.path.join(qp_dir, 'sigs.npz') 
# depth_pq_file = os.path.join(qp_dir,"dnssec_DAG_QPS_depth_median.npy" )
# domain_dnssec = ['berkeley.edu', 'energy.gov', 'aetna.com', 'Nairaland.com']
# depth_minus1_file = os.path.join( qp_dir,  "Hamming_2.csv") 


topK = 10

####flag 

all_plot_similarity = args.all_plot_similarity
max_plot_similarity = args.max_plot_similarity


DNSSEC_True= args.DNSSEC_True # True

PROTO = args.proto 

# if PROTO.lower() == "dns" and DNSSEC_True == True: 
#     PROTO = "dns-dnssec"
# elif PROTO.lower() == "dns" and DNSSEC_True == False: 
#     PROTO = "dns-nodnssec"




if args.proto_folder == None:
    args.proto_folder = PROTO

print(" ", args.proto_folder )

proto_dir = os.path.join(plot_root_dir, args.proto_folder ) #"yucheng_plots/"+PROTO





SetCover_True = True


# load QPs

# depths = [-1] #6,5,4,3,2,1] #[-1]
depths =  args.depth # [0,1,2,3,4,5,6,7,8,9]

########### Hamming QP ######################################
# out_dir = "yucheng_plots/"+PROTO+"/hamming"
# if not os.path.exists(out_dir):
#     os.makedirs(out_dir)
    
# QPs = pd.read_csv(QP_filename)
# QPs.sort_values(by=['amp_fac'], ascending=False,inplace=True)
#############################################################





#plt.clf() 

#sys.exit(1)



def compute_percentile(QP_AFs, PERCENTILE, outfile):
    QP_AFs_percentile = {}
    for key, value in QP_AFs.items():
        QP_AFs_percentile[key] = np.percentile(value, PERCENTILE)
    
    QP_AFs_percentile = OrderedDict(sorted(QP_AFs_percentile.items(), key = itemgetter(1), reverse = True))
    
    fp = open(os.path.join(out_dir, outfile+"_PERCENTILE_"+str(PERCENTILE)+".csv"), 'w')
    for key, value in QP_AFs_percentile.items():
        fp.write("{},{}\n".format(key, value))
    fp.close()
    
    return QP_AFs_percentile


# In[6]:

# map one query (currow) to a list of QPs
# Input: one query
# Output: a list of matching QPs index
def map_query_to_QP(currow):
    newrow=[]
    for row_sig in signatures:
        e1=row_sig[0]
        if e1 in ['url']:
            continue
        e2=hc[e1]

#         print(e1,e2)
        if(e2!=2):
            newrow.append(currow[e1])
        else:
            sigvals=col_info[e1]
#             print(sigvals)
            curval = currow[e1]
            for i,rr in enumerate(sigvals):
                if curval in rr[0]:
                    newrow.append(i)
                    break
#     print(currow,newrow,new_sigs[0,:])
    newrow=np.array(newrow)
#     print(newrow.shape)
    curmatches=[]
    for signum,sig in enumerate(new_sigs):
        match = 1
        for k in range(0,newrow.shape[0]):
            sigin=k
            v1=newrow[k]
            cur_col=sig[sigin]
#             print(v1,cur_col)
            if '-1' in cur_col:
                continue
#                 match=1
#             else:  
            if str(v1) not in cur_col:
                match=0
                break
            
        if(match==1):
            curmatches.append(signum)
    #             aux_array.
    curAF=currow[0]
#     print("ROW : ",newrow)
#     print("curmatches: ", curmatches)
#     for matches in new_sigs[curmatches]:
#         print("matches: ", matches)
    return curmatches


# In[7]:

# convert a list of tuples to dict
def merge_tuples(tuple_list):
    results = {}
    for item in tuple_list:
        key = item[0]
        value = item[1]
        
        if key not in results:
            results[key] = []
        results[key].append(value)
        
    return results


def read_from_json(filename):    
    with open(filename, 'r') as fp:
        dict_ = json.load( fp )
    return dict_


def output_dict_to_json(dict_, filename):
    results = {}
    results["children"] = []
    for key, value in dict_.items():
        result = {"Name": "QP "+str(key), "Count": round(value, 2)}
        results["children"].append(result)
        
    with open(filename, 'w') as fp:
        json.dump(results, fp)

def output_dict_to_json_v2(dict_, filename):    
    with open(filename, 'w') as fp:
        json.dump(dict_, fp)

# AForCount : 0: AF, 1: Count
def output_dict_to_csv(dict_, filename, AForCount):
    with open(filename, 'w') as fp:
        if AForCount == 0:
            fp.write("QP_index,meanAF\n")
        elif AForCount == 1:
            fp.write("QP_index,count\n")
        elif AForCount == 2:
            fp.write("QP_index,medianAF\n")
        for key, value in dict_.items():
            fp.write("{},{}\n".format(key, value))




def output_json_to_html(infile, outfile, AForCount):
    fr = open("bubble_plot.html", 'r')
    fw = open(os.path.join(proto_dir, "depth_"+str(DEPTH)+"_"+outfile), 'w')
    print(os.path.join(proto_dir, "depth_"+str(DEPTH)+"_"+outfile))
    
    infile = "depth_"+str(DEPTH)+"/"+infile
    
    for line in fr:
        if (line.strip().startswith("d3.json")):
            fw.write("\t\td3.json(\"%s\", function(dataset) {\n"%infile)
        elif (line.strip().startswith("var diameter")):
            if AForCount == 0:
                fw.write("\t\t\tvar diameter = 800\n")
            elif AForCount == 1:
                fw.write("\t\t\tvar diameter = 600\n")
        else:
            fw.write(line)
    fr.close()
    fw.close()


def output_QP_stats(QP_AFs):
    QP_mean_AF = {}
    QP_occ = {}
    QP_percent = {}
    
    total_len = 0
    
    for key, value in QP_AFs.items():
        QP_mean_AF[key] = np.mean(value)
        QP_occ[key] = len(value)
        total_len += len(value)
    
    for key, value in QP_occ.items():
        QP_percent[key] = float(value)/float(total_len)
    
    QP_mean_AF = OrderedDict(sorted(QP_mean_AF.items(), key = itemgetter(1), reverse = True))
    QP_occ = OrderedDict(sorted(QP_occ.items(), key = itemgetter(1), reverse = True))
    QP_percent = OrderedDict(sorted(QP_percent.items(), key = itemgetter(1), reverse = True))
    
    return QP_mean_AF, QP_occ, QP_percent

# In[24]:

# box plot for top FIVE QPs
# pick TOP by MEAN AF
# box plot for top FIVE QPs
# pick TOP by MEAN AF
def QP_boxplot(QP_AFs, QP_mean_AF, topK, outfile, title, rank_by):
    assert(len(QP_AFs) == len(QP_mean_AF))
    top_index_num = min(len(QP_mean_AF), topK)
    #print("top index num" , top_index_num)
    #print("list ",list(QP_mean_AF.keys()))
    top_index = list(QP_mean_AF.keys())[:top_index_num]
    #print(top_index)
    data = []
    xlabels = []
    nll=[]
    plt.style.use(['seaborn-whitegrid', 'seaborn-paper'])
    df = pd.DataFrame(columns=['QPs', 'value'])
    rowlist=[]
#     dict={}
    for index in top_index:
        values=QP_AFs[index]
        for e1 in values:
            curd={}
            curd['QP']="QP"+str(index)
            curd['AF'] = e1
            rowlist.append(curd)
#     print(rowlist)
    df = pd.DataFrame(rowlist)               

#     print(df.head())
#     ()+1
#         data.append(QP_AFs[index])
#         xlabels.append("QP "+str(index))
#         curd
#         nll.append
#     print(xlabels)
#     print(data)
    plt.clf()
    plt.figure(figsize=(20, 5))
    ax = sns.boxplot(x="QP", y="AF", data=df,  linewidth=4, palette="Set2",dodge=True,showmeans=True ) #  figsize=(15,6))
    #ax = sns.boxplot(x='mode',y='count',hue='design',data=df1,linewidth=4, palette="Set2",dodge=True,showmeans=True )

#     plt.boxplot(data)
    ax.set_xticks([i for i in range(top_index_num)], xlabels) #, fontsize=18)
    #ax.set_xticklabels([i for i in range(top_index_num)], xlabels, fontsize=18)
    ax.set_ylabel("Amplification Factor", fontsize=24)
    ax.set_xlabel("Query Patterns (QP) ranked by {}".format(rank_by), fontsize=25, labelpad=20)
    ax.tick_params(axis='x', labelsize=21)
    ax.tick_params(axis='y', labelsize=23)
    
    #plt.title(title)
    
    
    plt.savefig(outfile,bbox_inches='tight')
    
    





for DEPTH in depths:
    print("DEPTH: ", DEPTH)

    ########### DEPTH QP ######################################
    proto_dir =    os.path.join(plot_root_dir , args.proto_folder) #   "yucheng_plots/"+ args.proto
    if not os.path.exists(proto_dir):
        os.makedirs(proto_dir)
    #out_dir =  plot_root_dir + "/" +  PROTO +  "/depth_"+str(DEPTH)   #  "yucheng_plots/"+ args.proto +"/depth_"+str(DEPTH)

    out_dir = proto_dir +  "/depth_"+str(DEPTH)   #  "yucheng_plots/"+ args.proto +"/depth_"+str(DEPTH)
    #out_dir =  plot_root_dir + "/" +  args.proto_folder +  "/depth_"+str(DEPTH) 
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    if SetCover_True == True: 
        print(depth_pq_file)
        depth_QPs_dict = np.load( depth_pq_file, allow_pickle=True )
        QPs_set = depth_QPs_dict.item()
        QPs = QPs_set[DEPTH]
        QPs = QPs.applymap(str)
    else: 
        depth_QPs_dict = np.load( depth_pq_file , allow_pickle=True )
        QPs = pd.read_csv(depth_minus1_file)
        QPs.sort_values(by=['amp_fac'], ascending=False,inplace=True)
        print(QPs)

    #depth_QPs_dict = np.load("DAG_QPS_depth.npy")

    ##############################################################
     

    # load queries
    all_queries=pd.read_csv(Queries_filename)

    # load signature files
    sig = np.load(sig_filename, allow_pickle=True )

    hc = sig['hc'].item()
    signatures=sig['sigs']


    # In[2]:

    # group dataframe by server id
    # Note that groupby() preserves order
    # dict = {server_id : query}
    dict_serverid_query = {}
    for item in all_queries.groupby('server_id'):
        serverid = item[0]
        queries = item[1]
        dict_serverid_query[serverid] = queries

    print(len(dict_serverid_query))


    # In[3]:

    counter=0
    col_info={}
    
    #hc = sorted(hc.items(), key = lambda kv:(kv[1], kv[0]))
    from collections import OrderedDict 
    hc_new = OrderedDict() 

    # to preserve sig and hc ordering 
    for s in signatures: 
        print(s )
        field_name = s[0]
        hc_new[field_name] = hc[field_name] 
    hc = hc_new 

    print("hc is ", hc)
    print("sig is ", signatures)
    for col,tp in hc.items():
        counter+=1
        if tp!=2:
            continue
        curcnt= counter-1
        vals= signatures[curcnt]
        scores=[]
        print("col is ", col) 
        print(" vals ", vals)

        #start from 1 to skip the field name 
        for k in range(1,len(vals)):
            rr=vals[k]
            print("rr is ", rr)
            curscore=0
            for eentry in rr:
                curscore+=len(eentry)
            scores.append(curscore)
                    # scores.append(len(rr[0]))
        #()+1
        rank=ss.rankdata(scores)
        values=vals[1:]
        print("rank", rank)
        print("values ", values)
        sortedsigvals = [x for _,x in sorted(zip(rank,values))]
        col_info[vals[0]]=sortedsigvals


    # In[4]:
    # store QP index to QP
    print(col_info,hc)
    simplifiedQP=[]

    for index,currow in QPs.iterrows():
        newrow=currow.copy()
        for k,v in col_info.items():
            if '-1'  in newrow[k] or '-2'  in newrow[k]:
                continue
            curvalues=ast.literal_eval(currow[k])
            newval=[]
            #print("cur values ", curvalues)
            for eachval in curvalues:
                 #print([int(float(eachval))])
                 newval.append(str(v[int(float(eachval))][0]))

            newrow[k] =newval
        simplifiedQP.append(newrow.values)
    print(simplifiedQP[0])
    simpDF=pd.DataFrame(simplifiedQP,columns=QPs.columns)
    simpDF.to_csv(os.path.join(out_dir, "QPs_depth_"+str(DEPTH)+".csv"))
    simpDF



    #Writing some stuff 
    out_f = os.path.join(out_dir, "field_values_depth_"+str(DEPTH)+".csv")
    fields_f = open(out_f, "w")
    for column in simpDF.columns:
        #print(list(set([a for b in simpDF[column].tolist() for a in b])))
        uniq_vals = np.unique(np.hstack(simpDF[column])).tolist()
        print(column)
        parsed_unique_vals = []

        if "amp_fac" in column: 
            continue
        for i in uniq_vals:
            print(i ) 
            new_val = i.replace('[','')
            new_val = new_val.replace(']','')
            parsed_unique_vals.append(new_val)
        print(column, parsed_unique_vals)
        print("{}, {}".format(column,parsed_unique_vals))
        #for i in uniq_vals:
        fields_f.write("{}, {}\n".format(column,parsed_unique_vals)) 

    fields_f.close()


    # In[5]:

    new_sigs=QPs.values
    print("Curr QP size: ", new_sigs.shape)


    if "dns" in PROTO.lower(): 
        if DNSSEC_True == True: 
            all_queries=all_queries.query('url in @domain_dnssec')
            del all_queries['url']
        else: 
            all_queries=all_queries.query('url not in @domain_dnssec')
            del all_queries['url']

    inter_out=[]



    # In[8]:

    # Task 1: Measurement similarity between servers

    # CHANGE AF_threshold if needed
    AF_threshold = 10

    # ONLY count the maximum query

    # ONLY count the maximum query
    cnt = 0
    max_hit = 0
    num_total_servers = 0
    total_largest_queries = []
    max_serverid_QPs = {}
    max_QP_AF_tuples = []

    serverid_to_serverIndex = {}


    file_variables_misc = os.path.join(out_dir, "misc_variables.json")
    exists_file_var = os.path.isfile(file_variables_misc )



    max_serverid_QPs_files = os.path.join(out_dir, "max_serverid_QPs.json")
    exists_max_serverid_QPs =  os.path.isfile(max_serverid_QPs_files)


    sid_to_sindex_file = os.path.join(out_dir, "serverid_to_serverIndex.json")
    exists_sid_to_sindex_file = os.path.isfile(sid_to_sindex_file)


    max_QP_AF_tuples_file = os.path.join(out_dir, "max_QP_AF_tuples.npy")
    exists_max_QP_AF_tuples_file = os.path.isfile(max_QP_AF_tuples_file)


    if exists_file_var and exists_max_serverid_QPs and exists_sid_to_sindex_file and \
        exists_max_QP_AF_tuples_file:

        print("Files exist")
        max_serverid_QPs = read_from_json(max_serverid_QPs_files)
        serverid_to_serverIndex = read_from_json(sid_to_sindex_file)
        d = read_from_json(file_variables_misc)
        max_QP_AF_tuples = np.load( max_QP_AF_tuples_file, allow_pickle=True)
        cnt = d["cnt"]
        max_hit = d["max_hit"] 
        num_total_servers = d["num_total_servers"]# = num_total_servers
        #continue
    else: 
        print(" HERE   ")
        for serverid, queries in dict_serverid_query.items():
            largest_query = queries.iloc[0]
            serverid_to_serverIndex[serverid] = num_total_servers
            num_total_servers += 1
            #print(serverid, largest_query["amp_fac"])

            if (largest_query["amp_fac"] >= AF_threshold):
                cnt += 1
                total_largest_queries.append(largest_query)
                matched_QPs = map_query_to_QP(largest_query)
                max_serverid_QPs[serverid] = matched_QPs

                for i in matched_QPs:
                    max_QP_AF_tuples.append((i, largest_query["amp_fac"]))

                #print(max_QP_AF_tuples)
                #()+1

                if len(max_serverid_QPs[serverid]) != 0:
                    max_hit += 1
        print(" finished processing ")
            #Store some variables 
        d = {}
        d["cnt"] = cnt
        d["max_hit"] = max_hit
        d["num_total_servers"] = num_total_servers
        output_dict_to_json_v2(d, file_variables_misc)
        print(len(max_QP_AF_tuples ), max_QP_AF_tuples_file  )
        np.save( max_QP_AF_tuples_file,  max_QP_AF_tuples )
        print("SAVED")
        output_dict_to_json_v2(max_serverid_QPs, max_serverid_QPs_files )
        output_dict_to_json_v2(serverid_to_serverIndex, sid_to_sindex_file )





    if max_plot_similarity : # == True: 
        print("ONLY maximum query from each server >= THRESHOLD, HIT RATE = {}/{} = {}%"
              .format(max_hit, cnt, round(float(max_hit)/cnt*100, 2)))
        
        # max_feature_vector
        max_feature_vec = np.zeros((num_total_servers, len(QPs)))

        for serverid, QP in max_serverid_QPs.items():
            serverIndex = serverid_to_serverIndex[serverid]
            
            for Q in QP:
                max_feature_vec[serverIndex][Q] = 1

        print("Here")
        # In[10]:

        # only plot for valid servers
        max_valid_serverIndex = []
        for serverid in max_serverid_QPs.keys():
            max_valid_serverIndex.append(serverid_to_serverIndex[serverid])
        max_feature_vec = max_feature_vec[max_valid_serverIndex]


        # In[11]:

        # compute similarities
        #if plot_similarity == True: 
        print("Plot max similarity ")
        max_similarity_file = os.path.join(out_dir, "max_jaccard_similarity_matrix.npy")
        exist_max_similarity = os.path.isfile(max_similarity_file)

        max_similarity_matrix = np.zeros((cnt, cnt))

        if exist_max_similarity: 
            max_similarity_matrix = np.load(max_similarity_file) 
        else:
            for i in range(cnt):
                for j in range(cnt):
                    #print(i,j )
                    max_similarity_matrix[i][j] = jaccard_similarity_score(max_feature_vec[i], max_feature_vec[j])
                    #print(max_similarity_matrix[i][j])
                    #print(len(max_feature_vec[i]), len(max_feature_vec[j]))
                    #()+1
                    #max_similarity_matrix[i][j] = distance.jaccard(max_feature_vec[i], max_feature_vec[j])

            np.save(max_similarity_file,max_similarity_matrix  )


        plt.clf()
        plt.figure(figsize=(15, 15))
        
        ax = sns.heatmap(max_similarity_matrix, xticklabels=cnt-1, yticklabels=cnt-1)
        ax.set_title("Heatmap for maximum queries ONLY, HIT RATE = {}/{} = {}%"
              .format(max_hit, cnt, round(float(max_hit)/cnt*100, 2)), fontsize=8)
        # plt.show()
        plt.savefig(os.path.join(out_dir, "heatmap_depth_{}_max.pdf".format(DEPTH)),  bbox_inches='tight')
        print("done plot max similarity ")
    # In[13]:
    # count ALL the queries >= THRESHOLD


    print("Count all queries >= Thresh")
  
    all_hit = 0
    all_serverid_QPs = {}
    all_QP_AF_tuples = []


    all_QP_AF_tuples_files = os.path.join(out_dir, "all_QP_AF_tuples.npy")
    exist_f1 = os.path.isfile(all_QP_AF_tuples_files)

    all_serverid_QPs_file = os.path.join(out_dir, "all_serverid_QPs.json")
    exist_f2 = os.path.isfile(all_serverid_QPs_file)


    if exist_f1 and exist_f2: 
        print("Files exist ")
        #Stores all_serverid_QPs 
        with open( all_serverid_QPs_file, 'r') as fin: 
            all_serverid_QPs = json.load( fin)
            
        #Stroes all_QP_AF_tuples 
        all_QP_AF_tuples = np.load(all_QP_AF_tuples_files , allow_pickle=True )
        
    else: 
        #Compute data 
        total_len_servers = len(dict_serverid_query)
        print("Computing data ", total_len_servers)
        s_count = 0  
        for serverid, queries in dict_serverid_query.items():
            if s_count % 100 == 0: 
                print("server count :", s_count , " left : ", total_len_servers - s_count)
            for index, query in queries.iterrows():
                if (query["amp_fac"] >= AF_threshold):
                    if serverid not in all_serverid_QPs:
                        all_serverid_QPs[serverid] = []
                    matched_QPs = map_query_to_QP(query)
                    all_serverid_QPs[serverid] += matched_QPs

                    for i in matched_QPs:
                        all_QP_AF_tuples.append((i, query["amp_fac"]))
            s_count = s_count + 1
            if (serverid in all_serverid_QPs) and (len(all_serverid_QPs[serverid]) != 0):
                all_hit += 1
        #Stores files 
        np.array(all_QP_AF_tuples).dump(open(all_QP_AF_tuples_files, 'wb'))
        with open(all_serverid_QPs_file, 'w') as f:
            json.dump(all_serverid_QPs, f)




    # In[14]:

    print("ALL queries from each server >= THRESHOLD, HIT RATE = {}/{} = {}%"
          .format(all_hit, len(all_serverid_QPs), round(float(all_hit)/len(all_serverid_QPs)*100, 2)))


    print("Here  before all feature  ")
    # all_feature_vector
    all_feature_vec = np.zeros((num_total_servers, len(QPs)))

    for serverid, QP in all_serverid_QPs.items():
        serverIndex = serverid_to_serverIndex[serverid]
        
        for Q in QP:
            all_feature_vec[serverIndex][Q] = 1
    print("done all feature   ")


    # In[15]:

    if all_plot_similarity  : #== True: 
        # only plot for valid servers
        all_valid_serverIndex = []
        for serverid in all_serverid_QPs.keys():
            all_valid_serverIndex.append(serverid_to_serverIndex[serverid])
        all_feature_vec = all_feature_vec[all_valid_serverIndex]


        # In[16]:
        all_similarity_matrix = np.zeros((len(all_serverid_QPs), len(all_serverid_QPs)))


        all_similarity_file = os.path.join(out_dir, "all_jaccard_similarity_matrix.npy")
        exist_all_similarity = os.path.isfile(all_similarity_file)


        if exist_all_similarity: 
            #load
            all_similarity_matrix = np.load(all_similarity_file )
        else: 
            for i in range(len(all_serverid_QPs)):
                for j in range(len(all_serverid_QPs)):
                    #all_similarity_matrix[i][j] = distance.jaccard(all_feature_vec[i], all_feature_vec[j])
                    all_similarity_matrix[i][j] = jaccard_similarity_score(all_feature_vec[i], all_feature_vec[j])
                    if DEPTH == 0: 
                        print(all_similarity_matrix[i][j]) 
            np.save(all_similarity_file,all_similarity_matrix  )
        # In[17]:

        #plt.clf()
        #if plot_similarity == True: 

        print("Print all similarity")
    
        ax = sns.heatmap(all_similarity_matrix, xticklabels=cnt-1, yticklabels=cnt-1)
        # ax.set_title("Heatmap for ALL queries, HIT RATE = {}/{} = {}%"
        #       .format(all_hit, len(all_serverid_QPs), round(float(all_hit)/len(all_serverid_QPs)*100, 2)), fontsize=20)
        # plt.show()
        # ax.set(xlabel='Server ID', ylabel='Server ID')
        print("Done plot  all similarity")

        ax.set_xlabel('Server ID', fontsize=15)
        ax.set_ylabel('Server ID', fontsize=15)
        plt.savefig(os.path.join(out_dir, "heatmap_depth_{}_all.pdf".format(DEPTH)),bbox_inches='tight')
# In[18]:
    print("Saved fig ")

    max_QP_AFs_file = os.path.join(out_dir, "max_QP_AFs.json")
    exists_max_QP = os.path.isfile(max_QP_AFs_file)
    print("checkpoint ")
    all_QP_AFs_file = os.path.join(out_dir, "all_QP_AFs.json")
    exists_all_QP = os.path.isfile(all_QP_AFs_file)
    print("chk point 2 ")
    #print(max_QP_AF_tuples )
    #()+1

    #if exists_max_QP: 
    #    print(" exists max QP")
    #    max_QP_AFs = read_from_json(max_QP_AFs_file)
    #else: 
    print("max qp af tules length ", len(max_QP_AF_tuples) )
    max_QP_AFs = merge_tuples(max_QP_AF_tuples)
    print("max QP AF s ", len(max_QP_AFs))
        
    #if exists_all_QP: 
    #    all_QP_AFs = read_from_json(all_QP_AFs_file)
    #else: 
    all_QP_AFs = merge_tuples(all_QP_AF_tuples)
    print("all QP AF s ", len(all_QP_AFs))
    max_QP_mean_AF, max_QP_occ, max_QP_percent = output_QP_stats(max_QP_AFs)
    all_QP_mean_AF, all_QP_occ, all_QP_percent = output_QP_stats(all_QP_AFs)




    # In[20]:

    # In[21]:

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # json format
    output_dict_to_json(max_QP_mean_AF, os.path.join(out_dir, "max_QP_mean_AF.json"))
    output_dict_to_json(all_QP_mean_AF, os.path.join(out_dir, "all_QP_mean_AF.json"))
    output_dict_to_json(max_QP_occ, os.path.join(out_dir, "max_QP_occ.json"))
    output_dict_to_json(all_QP_occ, os.path.join(out_dir, "all_QP_occ.json"))
    output_dict_to_json_v2(max_QP_AFs, os.path.join(out_dir, "max_QP_AFs.json"))
    output_dict_to_json_v2(all_QP_AFs, os.path.join(out_dir, "all_QP_AFs.json"))

    # csv format

    output_dict_to_csv(max_QP_mean_AF, os.path.join(out_dir, "max_QP_mean_AF.csv"), 0)
    output_dict_to_csv(all_QP_mean_AF, os.path.join(out_dir, "all_QP_mean_AF.csv"), 0)
    output_dict_to_csv(max_QP_occ, os.path.join(out_dir, "max_QP_occ.csv"), 1)
    output_dict_to_csv(all_QP_occ, os.path.join(out_dir, "all_QP_occ.csv"), 1)


    # In[22]:

    # compute 98% percentile for QPs
    max_QP_AFs_percentile_98 = compute_percentile(max_QP_AFs, 98, "max_QP_AFs")
    all_QP_AFs_percentile_98 = compute_percentile(all_QP_AFs, 98, "all_QP_AFs")

    # compute median (50% percentile) for QPs
    max_QP_AFs_median = compute_percentile(max_QP_AFs, 50, "max_QP_AFs")
    all_QP_AFs_median = compute_percentile(all_QP_AFs, 50, "all_QP_AFs")

    output_dict_to_csv(max_QP_AFs_median, os.path.join(out_dir, "max_QP_AFs_median.csv"), 2)
    output_dict_to_csv(all_QP_AFs_median, os.path.join(out_dir, "all_QP_AFs_median.csv"), 2)



    # In[23]:

    # output html files for bubble plot
    # AForCount : 0: AF, 1: Count   

    #output_json_to_html("max_QP_mean_AF.json", "max_QP_mean_AF.html", 0)
    #output_json_to_html("all_QP_mean_AF.json", "all_QP_mean_AF.html", 0)
    #output_json_to_html("max_QP_occ.json", "max_QP_occ.html", 1)
    #output_json_to_html("all_QP_occ.json", "all_QP_occ.html", 1)
    
    
    print(len(all_QP_AFs), len(all_QP_mean_AF))

    # box plot based on mean AF
    QP_boxplot(max_QP_AFs, max_QP_mean_AF, topK, 
               os.path.join(out_dir, "boxplot_depth_"+str(DEPTH)+"_max_rank_by_mean.pdf"), 
               "Depth {}: Max Query based on mean AF".format(DEPTH), "mean AF")
    # ()+1
    print("plot 1 done ")
    QP_boxplot(all_QP_AFs, all_QP_mean_AF, topK, 
               os.path.join(out_dir, "boxplot_depth_"+str(DEPTH)+"_all_rank_by_mean.pdf"), 
               "Depth {}: All Queries based on mean AF".format(DEPTH), "mean AF")
    print("plot 2 done ")

    # box plot based on occurence
    QP_boxplot(max_QP_AFs, max_QP_occ, topK, 
               os.path.join(out_dir, "boxplot_depth_"+str(DEPTH)+"_max_rank_by_occ.pdf"), 
               "Depth {}: Max Query based on occurence".format(DEPTH), "NumOccurence")

    print("plot 3 done ")


    QP_boxplot(all_QP_AFs, all_QP_occ, topK, 
               os.path.join(out_dir, "boxplot_depth_"+str(DEPTH)+"_all_rank_by_occ.pdf"), 
               "Depth {}: All Queries based on occurence".format(DEPTH), "NumOccurence")

    print("plot 4 done ")

    # box plot based on 98% percentile AF
    QP_boxplot(max_QP_AFs, max_QP_AFs_percentile_98, topK, 
               os.path.join(out_dir, "boxplot_depth_"+str(DEPTH)+"_max_rank_by_98.pdf"), 
               "Depth {}: Max Query based on 98% percentile AF".format(DEPTH), "98th percentile")

    print("plot 5 done ")


    QP_boxplot(all_QP_AFs, all_QP_AFs_percentile_98, topK, 
               os.path.join(out_dir, "boxplot_depth_"+str(DEPTH)+"_all_rank_by_98.pdf"), 
               "Depth {}: All Queries based on 98% percentile AF".format(DEPTH), "98th percentile")

    print("plot 5 done ")

    # box plot based on median (50% percentile) AF
    QP_boxplot(max_QP_AFs, max_QP_AFs_median, topK, 
               os.path.join(out_dir, "boxplot_depth_"+str(DEPTH)+"_max_rank_by_median.pdf"), 
               "Depth {}: Max Query based on median AF".format(DEPTH), "median AF")


    print("plot 6 done ")

    QP_boxplot(all_QP_AFs, all_QP_AFs_median, topK, 
               os.path.join(out_dir, "boxplot_depth_"+str(DEPTH)+"_all_rank_by_median.pdf"), 
               "Depth {}: All Queries based on median AF".format(DEPTH), "median AF")
    print("Done generating plots ")
print("Done running depths ", depths)
