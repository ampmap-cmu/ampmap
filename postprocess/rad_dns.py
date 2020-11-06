import numpy as np
import pandas as pd
import ntpath
import glob,json,os,sys,time
import pandas as pd
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import numpy as np
# import jenkspy
from scipy.signal import argrelextrema
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN,AffinityPropagation
from sklearn import metrics
from convert import *
from plot import fig_gen

import configparser



gpath ='./new-src/'
sys.path.append(os.path.abspath(gpath))


config = configparser.RawConfigParser()   

to_include ='dns'
configFilePath ="config/params.py"
print(configFilePath)
config.read(configFilePath)
details_dict = dict(config.items(to_include))
print(details_dict)
inputs_path = os.path.join( os.getcwd(), gpath, details_dict['ipp'])


og_dict = dict(config.items('all'))
print(og_dict)
proto_name =  details_dict['proto_name']

folder_name = og_dict['data_path']+str(proto_name)+"/"

KL_thresh=float(og_dict['kl_thresh'])
high_AF_thresh=float(og_dict['af_thresh'])

proto_name=og_dict['sv_path']+proto_name+'_out_KL_'+str(KL_thresh)+'_AF_'+str(high_AF_thresh)

# inputs_path = os.path.join( os.getcwd(), gpath, "field_inputs_dns_db2")

import libs 
import libs.inputs  as inputs
import operator
from scipy import stats
import scipy.stats as ss
from itertools import groupby
from bitstring import BitArray
from scipy.stats import chisquare
from merged_QP import reduced_QPspace,DAG_setcover
from common import *

from collections import OrderedDict
import libs.definition as lib_df 
import scipy.stats as ss
def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))
def signature_matching(df,ignored_cols,range_info,bin_arr,sigs_fname,high_AF_thresh,proto_name):
    if 1 == 0:
        print("DO NOTHING")

    else:
        range_contcols={}
        priority={}
        rank={}
        for column in cont_cols:
            col_sig=range_info[column]
            col_sig=np.array(col_sig)
            col_bin=np.array(bin_arr[column])
            for j1 in range(col_sig.shape[0]):
                current_sig=col_sig[j1,:]
                print("o1: ",j1,column,current_sig)
                valid_inds=np.argwhere(current_sig==1)
                valid_ranges=col_bin[valid_inds]

                ll=[]
                for rr in valid_ranges:
                    for pt in rr[0]:
                        ll.append(pt)

                data= list(ranges(ll))
                print("p1: ",len(data),data,len(data[0]))
                curscore=len(data[0])
                curscore=0
                for eentry in data:
                    curscore += len(eentry)
                if column in range_contcols:
                    range_contcols[column].append(data)
                    priority[column].append(curscore)
                else:
                    range_contcols[column]=[data]
                    priority[column]= [curscore]
                print("HERE: ",priority[column])
            rank[column]=ss.rankdata(priority[column])

        print("RC ",range_contcols," PP ",priority," RR ",rank)



        # ()+1

        # qdf=df
        # ur= 'berkeley.edu'
        # df=df.query('url in @ur')
        qdf= df.drop(columns=ignored_cols)
        signatures=[]
        handle_columns={}
        for i,col in enumerate(qdf.columns):
            print(i,col)
            if col  in['amp_fac','server_id']:
                continue
            curinfo=[col]
            if col not in cont_cols:
                curdata=np.unique(qdf[col].values)
                try:
                    curnewdata=np.sort(curdata.astype(np.float)).tolist()
                    handle_columns[col]=1
                except:
                    curnewdata=curdata.tolist()
                    handle_columns[col]=-1

                curinfo+=curnewdata
            else:

                breaks=range_contcols[col]
                handle_columns[col]=2

                curinfo+=breaks
            signatures.append(curinfo)
        #         print(col)
        signatures=np.array(signatures)
        print(signatures)
        # ()+1
        curfac=1
        for eachrow in signatures:
            values=eachrow[1:]
            curfac*=len(values)
            print(eachrow, len(values))
        print(curfac)
        print(qdf.shape,ignored_cols)
        final_dict={}
        count=1
        start=time.time()

        arr_ind=[]
        newdf=qdf.copy()
       
        collist=newdf.columns.values
        for sig in signatures:
            col=sig[0]

            current_row=qdf[col].values
            curcolindex=np.where(collist==col)[0]
            print(col,handle_columns[col],curcolindex)

            # ()+1
            # current_rowval=getattr(row, col)
                # print(col,current_rowval)
                # ()+1
            if(handle_columns[col]==1):
                curval=current_row.astype(float)
                sigvals=sig[1:]
                print("unq",np.unique(curval))
                newdf.iloc[:,curcolindex] = curval
                # i = np.where(curval == sigvals)
                # print(i)
                # ()+1
                # ind=sigvals.index(curval)
                # arr_ind.append(ind)
            elif(handle_columns[col]==-1):
                curval=current_row
                print("unq",np.unique(curval))
                newdf.iloc[:,curcolindex] = curval
            else:

                curval=current_row.astype(float)
                sigvals=np.array(sig[1:])
                curd=np.array([])
                cur_ranks=rank[col] 
                print("HH: ",sig," KK ",cur_ranks,"TT:\t\n",sigvals)

                kaka=zip(cur_ranks,sigvals)
                pkp = np.argsort(cur_ranks)
                print(kaka,pkp)
                sortedsigvals = [x for _,x in sorted(zip(cur_ranks,sigvals))]

                print(sig,cur_ranks,sigvals,sigvals,sortedsigvals,curval.shape)
                newscore=np.ones(curval.shape[0])*-2
                for i,rr in enumerate(sortedsigvals):
                    score=np.isin(curval,rr[0])
                    valid_inds=np.where(newscore==-2)[0]
                    truce = np.where(score==True)[0]
                    subset=np.intersect1d(valid_inds,truce)
                    print("VALID: ",valid_inds.shape,truce.shape,subset.shape)
                    newscore[subset]= i 
                print(newscore,np.unique(newscore),np.min(curval),np.max(curval))                
                curd=newscore
                newdf.iloc[:,curcolindex] = curd
        cdf=newdf.copy()
        cdf.to_csv(proto_name+'/ALL_Queries.csv',index=False)

        newdf.drop(['amp_fac','server_id'], inplace=True, axis=1)


        print(newdf.shape,newdf.head())
        col_grp=list(newdf.columns.values)
        print(col_grp,cdf.shape,cdf.head())

        domain_dnssec = ['berkeley.edu', 'energy.gov', 'aetna.com', 'Nairaland.com']
        domain_no_dnssec = ['chase.com', 'google.com', 'Alibaba.com', 'Cambridge.org', 'Alarabiya.net', 'Bnamericas.com']

        subcdf=cdf.copy()
        print("Sub",subcdf.shape)
        # cdf=subcdf
        res = subcdf.groupby(col_grp)['amp_fac'].median().reset_index()
        minres = subcdf.groupby(col_grp)['amp_fac'].min().reset_index()
        maxres = subcdf.groupby(col_grp)['amp_fac'].max().reset_index()
        countres = subcdf.groupby(col_grp)['amp_fac'].count().reset_index()
        print(minres.shape,res.shape,maxres.shape)
        res['min_amp'] = minres['amp_fac']
        res['max_amp'] = maxres['amp_fac']
        res['count_amp'] = countres['amp_fac']
        res.to_csv(proto_name+'/all_QPS.csv',index=False)

        subcdf=cdf.query('url in @domain_dnssec')
        print("Sub",subcdf.shape)
        res = subcdf.groupby(col_grp)['amp_fac'].median().reset_index()
        minres = subcdf.groupby(col_grp)['amp_fac'].min().reset_index()
        maxres = subcdf.groupby(col_grp)['amp_fac'].max().reset_index()
        countres = subcdf.groupby(col_grp)['amp_fac'].count().reset_index()
        print(minres.shape,res.shape,maxres.shape)
        res['min_amp'] = minres['amp_fac']
        res['max_amp'] = maxres['amp_fac']
        res['count_amp'] = countres['amp_fac']
        res.to_csv(proto_name+'/dnssec_QPs.csv',index=False)

        subcdf=cdf.query('url in @domain_no_dnssec')
        print("Sub",subcdf.shape)
        res = subcdf.groupby(col_grp)['amp_fac'].median().reset_index()
        minres = subcdf.groupby(col_grp)['amp_fac'].min().reset_index()
        maxres = subcdf.groupby(col_grp)['amp_fac'].max().reset_index()
        countres = subcdf.groupby(col_grp)['amp_fac'].count().reset_index()
        print(minres.shape,res.shape,maxres.shape)
        res['min_amp'] = minres['amp_fac']
        res['max_amp'] = maxres['amp_fac']
        res['count_amp'] = countres['amp_fac']
        res.to_csv(proto_name+'/nodnssec_QPs.csv',index=False)

        res = cdf.groupby(col_grp)['amp_fac'].median().reset_index()
        minres = cdf.groupby(col_grp)['amp_fac'].min().reset_index()
        maxres = cdf.groupby(col_grp)['amp_fac'].max().reset_index()
        countres = cdf.groupby(col_grp)['amp_fac'].count().reset_index()

        print(minres.shape,res.shape,maxres.shape)
        res['min_amp'] = minres['amp_fac']
        res['max_amp'] = maxres['amp_fac']
        res['count_amp'] = countres['amp_fac']

        amp_scores=res['amp_fac'].values
        print(amp_scores,res.shape)
        ins=np.where(amp_scores>=high_AF_thresh)[0]
        print(ins,ins.shape)
        outs=np.where(amp_scores<high_AF_thresh)[0]
        print(outs,outs.shape)
        res.to_csv('Rel_info1.csv',index=False)
        res=res.drop(res.index[outs])
        print(res.shape,res.head())
        res.to_csv(sigs_fname+'.csv', encoding='utf-8', index=False)
        res=pd.read_csv(sigs_fname+'.csv')                  
        np.savez(sigs_fname, hc =handle_columns,sigs = signatures)
    return res,handle_columns,signatures
def log_sample_from_range(list_search): 
    if (list_search[-1] - list_search[0]) != (len(list_search) -1): 
        print("list is ", list_search)
        raise ValueError("The list (for long field) should be contiguous ") 
    print("LIST SEARCH",list_search)
    if list_search[0] == 0: 
        sampled = [x for x in np.geomspace(1, list_search[-1] + 1, num=int(np.log2(len(list_search)+1)), dtype=int)]
        print("samM M pled",sampled)
        sampled[0]-=1
        sampled_int = [int(x) for x in sampled]
        return sampled
    num_samples= np.log2(len(list_search)+1)
    print(num_samples)

    sampled = [x for x in np.geomspace(list_search[0], list_search[-1]+1, num=np.log2(len(list_search)+1), dtype=int)]
    sampled_int = [int(x) for x in sampled]
    print("sampled",sampled_int)
    ()+1
    return sampled_int
print(inputs_path)
proto_fields =  inputs.generate_proto_fields(inputs_path)
def ranges(lst):
    pos = (j - i for i, j in enumerate(lst))
    t = 0
    for i, els in groupby(pos):
        l = len(list(els))
        el = lst[t]
        t += l
        yield range(el, el+l)
        

def get_freq_count(AF_THRESH,selected,proto_fields,df,KL_thresh,freq_fname,mass_thresh):

    binarr={}

    if os.path.isfile(freq_fname):
        ff=np.load(freq_fname,allow_pickle=True)
        # print(ff.files)
        field_to_server_freq=ff['arr_0'].item()
        binarr=ff['arr_1'].item()
    else:
        list_search = range(0,65536)
        num_bin = 10 
        print("selected one ", len(selected))
        server_list = selected 
        metric = "rangeExist"#"countfreq"
        dist = "jsDistance"
        #This 
        field_to_server_freq = OrderedDict() 
        #Dict of field to set 
        field_to_unique_freq = OrderedDict()
        print("Start building ")
        # Iterate over fields 
        for field_name, f_metadata  in proto_fields.items() :  #For all fields .. 
            
            print("Field name " , field_name , f_metadata)
        #     break 
            field_to_unique_freq[field_name] = set() 
            server_freq = OrderedDict() 
            
            if f_metadata.is_int == True: 
                df[field_name] = df[field_name].astype('int64') 
            
            
            flag=-1
            if f_metadata.size >= lib_df.SMALL_FIELD_THRESHOLD: #larger than 256 
                sampled =  log_sample_from_range( f_metadata.accepted_range)
                print("sampled : ", sampled, len(sampled))
                bins = [] 
                #construct bins in form of ranges 
                for i in range(len(sampled)-1): 
                    bins.append( range( sampled[i], sampled[i+1] ))
                print("Bins : " ,  bins)
                bin_str = ["{}-{}".format(b[0],b[-1]) for b in bins ] 

                query_str = '{} in @b '.format(field_name, field_name) 
                #num_sample = num_bin + 1
                #sampled = [x-1 for x in np.geomspace(1, list_search[-1], num=num_sample, dtype=int)]
                flag=0
            else: 
                bins = f_metadata.accepted_range 
                
                query_str = '{} == @b '.format(field_name, field_name) 
                flag=1
            binarr[field_name] = bins
            print("Field name ", field_name )
            print(" bins " , bins  )
            print("query str ", query_str,metric)
            # ()+1
            for slo1,server_id in enumerate(selected): 
                query=df.query('server_id in @server_id and amp_fac >= @AF_THRESH')
                if slo1%100==0:
                    print("Field name ", field_name )
                    print(" bins " , bins  )
                    print("query str ", query_str,metric)
                    print(slo1,server_id,query.shape,len(selected))
                # tempdf = df
                # ()+1

                freq = []             
                for b in bins: 
                    filtered = query.query( query_str )

                    if "rangeExist" in metric: 
                        if flag==0:
                            if len(filtered) > 0: 
                                freq.append(1)
                            else: 
                                freq.append(0)
                        else:
                            freq.append(len(filtered))
                    
                    elif "countfreq" in metric:  
                        freq.append( len(filtered))  
                    else: 
                        raise ValueError("Metric has to be either rangeExist or countFrequency! ")
                    
                
                server_freq[server_id] = freq #prob_vector 
                field_to_unique_freq[field_name].add(tuple(freq))
            
            field_to_server_freq[field_name] = server_freq

        # print("Finished computing the freq for field ", field_name,flag,field_to_server_freq[field_name] )
        np.savez(freq_fname, field_to_server_freq,binarr)
        # np.save(freq_fname+'', field_to_server_freq)


    IGNORE_LIST = getKL(proto_fields,lib_df,field_to_server_freq,proto_name,KL_thresh)


    field_to_freq_vector = OrderedDict() 
 
    for field, f_metadata  in proto_fields.items() :  #For all fields .. 
        if f_metadata.size >= lib_df.SMALL_FIELD_THRESHOLD:               
            server_to_freq_vectors = field_to_server_freq[field]
            # print("For field ", field )
            field_to_freq_vector[field] = OrderedDict()     
            #total_servers = 0
            tmp_vector = {} 
            for s in server_to_freq_vectors:     
                #print("\tFor server ", s)
                f_v = server_to_freq_vectors[s]
                f_v_tuple = tuple(f_v)
                if f_v_tuple not in tmp_vector: 
                    tmp_vector[f_v_tuple] = 1 
                else: 
                    cur_val = tmp_vector[f_v_tuple]
                    tmp_vector[f_v_tuple] = cur_val + 1 

            reversed_freq_vector = OrderedDict(sorted(tmp_vector.items(), key=lambda x: x[1]))
            sorted_freq_vector = OrderedDict(reversed(list(reversed_freq_vector.items())))
            field_to_freq_vector[field] = sorted_freq_vector
            # print("sorted freq vector ", sorted_freq_vector)


    server_list=selected
    num_server = len(server_list)

    range_info={}
    print(field_to_freq_vector)
    mass_thresh = 0.1
    # ()+1
    for f in field_to_freq_vector: 
        range_info[f]=[]
        vectors = field_to_freq_vector[f]
        # print("For field ", f, )
        total_server = 0 #len(vectors )
        for sig in vectors: 
            # print(" Sig {} : percentage {} ".format(sig, vectors[sig]/num_server ) )
            if (vectors[sig]/num_server > mass_thresh):
                # if f in range_info:
                range_info[f].append(sig)
                # else:
                #     range_info[f]=sig
    print("RANGES",range_info)
    # ()+1
    # print(bins)
    return IGNORE_LIST,range_info,binarr

    
def read_protodata(folder_name):
    fields=['amp_fac','server_id']
    folder_dir=folder_name+'query/*/*'
    folders=glob.glob(folder_dir)
    eachfile=folders[0]
    with open(eachfile) as f:
        data = json.load(f)
        cur_server=[]
        for ind_data in data:
            for attribute, value in ind_data['fields'].items():
                fields.append(attribute)
            break
    df = pd.DataFrame(columns=fields)
    print(df.head())
    rows_list = []
    stringlist=df.columns.values
    total_data=[]
    for cnt,eachfile in enumerate(folders):
        print(cnt,len(folders),len(total_data))
        statinfo = os.stat(eachfile)
        if(statinfo.st_size==0):
            continue
        with open(eachfile) as f:
            data = json.load(f)
            cur_server=[]
            for ind_data in data:
                current_value=[float(ind_data['amp_factor']),ntpath.basename(eachfile)]
                for i in range(2,len(stringlist)):
                    eachval=stringlist[i]
                    val1=ind_data['fields'][eachval]
                    current_value.append(val1)
                total_data.append(current_value)
                # print(ind_data)
                # ()+1
    total_data=np.array(total_data)
    df = pd.DataFrame(total_data)
    df.columns =fields
    
    df['amp_fac'] = pd.to_numeric(df['amp_fac'], errors='coerce').fillna(0)
    
    # print(df)
    df.sort_values(by=['amp_fac'], ascending=False,inplace=True)
    print(df.head())
    return df

'sig_KL'+str(KL_thresh)+'_AF_'+str(high_AF_thresh)+dname

if not os.path.exists(proto_name):
    os.makedirs(proto_name)
print(proto_name)

print(folder_name)


all_df_fname=folder_name+'/complete_info.csv'
print(all_df_fname)
exists = os.path.exists(all_df_fname)
print(exists)
if exists == True :
    df=pd.read_csv(all_df_fname)
else:
    raise Exception("File Not Found")
print(df.shape)
print(df)

# ()+1

srv_ids=np.unique(df['server_id'].values)
selected_server=-1

cur_num= 0 
selected =  [] 

newdf= df.groupby(['server_id'])['amp_fac'].max()
print(newdf.shape)

for i,row in newdf.iteritems():
    if(row>=high_AF_thresh):
        selected.append(i)
    print(i,"-->",row)
print("Number of Selected ", len(selected))

df=df.query('server_id in @selected')

print("PRE" , df.shape,"GHHH")
field_explore = ['payload', 'id']
mass_thresh = 0.1

freq_fname=proto_name+'/freqcount.npz'

[ignored_cols,range_info,bin_arr]= get_freq_count(high_AF_thresh,selected,proto_fields,df,KL_thresh,freq_fname,mass_thresh)

np.save(proto_name+'/ignored', ignored_cols)
np.save(proto_name+'/range',range_info)

print("IGNORED",ignored_cols)
print("RANGES",range_info)
print("BINS",bin_arr)


print(df.head())

cont_cols=[*range_info.keys()]

print("CONTS",cont_cols,bin_arr.keys())
sigs_fname=proto_name+'/sigs.npz'

print("DF",df.shape,"Num",len(selected))


[tempDF,handle_columns,signatures] = signature_matching(df,ignored_cols,range_info,bin_arr,sigs_fname,high_AF_thresh,proto_name)
tempDF.sort_values(by=['amp_fac'], ascending=False,inplace=True)


DAG_setcover(proto_name,1,0,high_AF_thresh)
fig_gen("dnssec_",0,proto_name)

DAG_setcover(proto_name,0,0,high_AF_thresh)
fig_gen("nondnssec_",0,proto_name)

DAG_setcover(proto_name,1,1,high_AF_thresh)
fig_gen("dnssec_",1,proto_name)


DAG_setcover(proto_name,0,1,high_AF_thresh)
fig_gen("nondnssec_",1,proto_name)


# fig_gen("dnssec",0,proto_name)
