import numpy as np
import pandas as pd
import ntpath
import glob,json,os,sys,time
import pandas as pd
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import argrelextrema
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN,AffinityPropagation
from sklearn import metrics
from importlib import import_module
# sys.path.append(os.path.abspath("./config/"))
import configparser
from common import *

ac_list=['snmp_next','snmp_get','snmp_bulk','snmp_next_or','snmp_get_or','snmp_bulk_or','ntp_normal_or']
gpath ='./new-src/'
sys.path.append(os.path.abspath(gpath))
config = configparser.RawConfigParser()   

to_include =sys.argv[1]

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



import libs 
import libs.inputs  as inputs
import operator
from scipy import stats
import scipy.stats as ss
from itertools import groupby
from bitstring import BitArray
from scipy.stats import chisquare

from collections import OrderedDict
import libs.definition as lib_df 
import scipy.stats as ss
from merged_QP_others import reduced_QPspace,DAG_setcover
from plot_data import fig_gen
def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def get_freq_count1(AF_THRESH,selected,proto_fields,df,KL_thresh,freq_fname,mass_thresh):

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
                df[field_name] = df[field_name].astype('int') 
            
            
            flag=-1
            if f_metadata.size >= lib_df.SMALL_FIELD_THRESHOLD: #larger than 256 
                sampled =  log_sample_from_range( f_metadata.accepted_range)
                print("sampled : ", sampled, len(sampled),f_metadata.size,f_metadata.accepted_range)
                new_sampled=[]
                for x in sampled:
                    if x ==0:
                        x=1
                    cval = int(np.log2(x))
                    new_sampled.append(cval)

                # new_sampled = [ for x in sampled]
                sampled=new_sampled
                # ()+1
                bins = [] 
                #construct bins in form of ranges 
                for i in range(len(sampled)-1): 
                    bins.append( range( sampled[i], sampled[i+1] ))
                    
                    #  bins.append(list(np.linspace(sampled[i],sampled[i+1],num=100)))
                print("Bins : " ,  bins)
                bin_str = ["{}-{}".format(b[0],b[-1]) for b in bins ] 

                query_str = '{} in @b '.format(field_name, field_name) 
                # toval = field
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
            # ()+1
            print("query str ", query_str,metric,flag)
            # ()+1
            newdf = df.loc[df['amp_fac'] >= AF_THRESH]
            # print(df.shape,newdf.shape)
            # ()+1
            counter=0

            for server_id in selected: 

                counter+=1
                if counter%100==0:
                    print("iterative over bin: => ",counter,server_id,bins,field_name)

                query = newdf.loc[newdf['server_id'] == server_id]
                # if field_name =="id":
                #     print(query.shape)
                if flag==0:
                    values_arr = query[field_name]

                freq = []        
                # print(field_name,query_str,f_metadata)
                cnt=0
                for b in bins: 
                    # ll = list(b)
                    if flag==0:

                        ll = list(b)
 
                        filtered = values_arr.isin(ll)


                    else:
                        filtered=query.loc[query[field_name] == b] 

     

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
        print(field,f_metadata.size,lib_df.SMALL_FIELD_THRESHOLD)

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
    # print("RANGES",range_info)
    # print(bins)
    return IGNORE_LIST,range_info,binarr

    

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
                # continue
                bins = f_metadata.accepted_range 
                
                query_str = '{} == @b '.format(field_name, field_name) 
                flag=1
            binarr[field_name] = bins
            print("Field name ", field_name )
            print(" bins " , bins  )
            print("query str ", query_str,metric)

            
            # ()+1
            for server_id in selected: 
                query=df.query('server_id in @server_id and amp_fac >= @AF_THRESH')
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
    # print("RANGES",range_info)
    # print(bins)
    return IGNORE_LIST,range_info,binarr

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
            # print(col_bin,col_sig.shape)
            for j1 in range(col_sig.shape[0]):
                current_sig=col_sig[j1,:]
                # print(current_sig)
                valid_inds=np.argwhere(current_sig==1)
                valid_ranges=col_bin[valid_inds]
                # print(col_sig,"BREA",valid_inds)
                # print("VALID",valid_ranges)
                ll=[]
                for rr in valid_ranges:
                    for pt in rr[0]:
                        ll.append(pt)

                data= list(ranges(ll))
                print(len(data),data[0],len(data[0]))
                curscore=len(data[0])
                if column in range_contcols:
                    range_contcols[column].append(data)
                    priority[column].append(curscore)
                else:
                    range_contcols[column]=[data]
                    priority[column]= [curscore]

                rank[column]=ss.rankdata(priority[column])

        print(range_contcols,priority,rank)


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
                lsd = np.array(qdf[col].values,dtype = str)
                print(lsd)
                curdata=np.unique(lsd)

                # curdata=np.unique(qdf[col].values)
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
        # for row in qdf.itertuples(index=False):
        #     # print(row)
        #     # ()+1
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
                # print("unq",np.unique(curval))
                newdf.iloc[:,curcolindex] = curval

                # sigvals=sig[1:]
                # ind=sigvals.index(curval)
                # arr_ind.append(ind)
            else:

                curval=current_row.astype(float)

                sigvals=np.array(sig[1:])
                # curd=np.zeros(len(sigvals))
                curd=np.array([])
                cur_ranks=rank[col] 
                sortedsigvals = [x for _,x in sorted(zip(cur_ranks,sigvals))]

                # sortedsigvals=sigvals.sort(key = cur_ranks)
                print(sigvals,sortedsigvals,curval.shape)
                # ()+1
                newscore=np.ones(curval.shape[0])*-1
                for i,rr in enumerate(sortedsigvals):
                    score=np.isin(curval,rr[0])
                    valid_inds=np.where(newscore==-1)[0]
                    truce = np.where(score==True)[0]
                    subset=np.intersect1d(valid_inds,truce)
                    print("VALID: ",valid_inds.shape,truce.shape,subset.shape)
                    newscore[subset]= i 

                print(newscore,np.unique(newscore),np.min(curval),np.max(curval))
                
                curd=newscore

                newdf.iloc[:,curcolindex] = curd

        # ()+1
        cdf=newdf.copy()
        cdf.sort_values(by=['amp_fac'], ascending=False,inplace=True)

        cdf.to_csv(proto_name+'/ALL_Queries.csv',index=False)

        # cdfback=cdf.copy()
        newdf.drop(['amp_fac','server_id'], inplace=True, axis=1)


        print(newdf.shape,newdf.head())
        # cdf.drop_duplicates(inplace=True)
        col_grp=list(newdf.columns.values)
        print(col_grp,cdf.shape,cdf.head())

        subcdf=cdf.copy()
        print("\n",subcdf.head())

        print("Sub",col_grp,subcdf.shape)
        # cdf=subcdf
        res = subcdf.groupby(col_grp)['amp_fac'].median().reset_index()
        minres = subcdf.groupby(col_grp)['amp_fac'].min().reset_index()
        maxres = subcdf.groupby(col_grp)['amp_fac'].max().reset_index()
        countres = subcdf.groupby(col_grp)['amp_fac'].count().reset_index()
        print(minres.shape,res.shape,maxres.shape)
        res['min_amp'] = minres['amp_fac']
        res['max_amp'] = maxres['amp_fac']
        res['count_amp'] = countres['amp_fac']
        res.sort_values(by=['amp_fac'], ascending=False,inplace=True)

        res.to_csv(proto_name+'/all_QPS.csv',index=False)
        print(signatures)
        print(range_contcols,priority,rank)

        print(handle_columns)
        # ()+1
        np.savez(sigs_fname, hc =handle_columns,sigs = signatures)
    return res,handle_columns,signatures
def log_sample_from_range(list_search): 
    if (list_search[-1] - list_search[0]) != (len(list_search) -1): 
        print("list is ", list_search)
        raise ValueError("The list (for long field) should be contiguous ") 
    print("LIST SEARCH",list_search)
    if list_search[0] == 0: 
        # sampled = [x for x in np.geomspace(1, list_search[-1] + 1, num=np.log2(len(list_search)+1), dtype=int)]
        sampled = [x for x in np.geomspace(1, list_search[-1] + 1, num=int(np.log2(len(list_search)+1)), dtype=int)]

        print("samM M pled",sampled)
        sampled[0]-=1
        sampled[-1]+=1
        sampled_int = [int(x) for x in sampled]
        print("sampled",sampled_int,sampled,list_search)

        # ()+1
        return sampled
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
    total_data=np.array(total_data)
    df = pd.DataFrame(total_data)
    df.columns =fields
    df['amp_fac'] = pd.to_numeric(df['amp_fac'], errors='coerce').fillna(0)
    df.sort_values(by=['amp_fac'], ascending=False,inplace=True)
    return df



# proto_name+='ntp_pvt'


if not os.path.exists(proto_name):
    os.makedirs(proto_name)

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


srv_ids=np.unique(df['server_id'].values)
selected_server=-1

# high_AF_thresh=10

cur_num= 0 
selected =  [] 

newdf= df.groupby(['server_id'])['amp_fac'].max()
print(newdf.shape)

for i,row in newdf.iteritems():
    if(row>high_AF_thresh):
        selected.append(i)
    print(i,"-->",row)
print("Number of Selected ", len(selected))

df=df.query('server_id in @selected')



print("PRE" , df.shape,"GHHH")


mass_thresh = 0.1

freq_fname=proto_name+'/freqcount.npz'
if to_include in ac_list:
    logfields=[]
    for field_name, f_metadata  in proto_fields.items() :  #For all fields .. 
                
        print("Field name " , field_name , f_metadata,f_metadata.size ,lib_df.SMALL_FIELD_THRESHOLD)
        if f_metadata.size >= lib_df.SMALL_FIELD_THRESHOLD: 
            logfields.append(field_name)
                # if f_metadata.size > lib_df.SMALL_FIELD_THRESHOLD: 
            print(field_name)
            ovals= df[field_name].values
            ovals+=1
            newvals = np.log2(ovals).astype(int)
            df[field_name] = newvals
    print("LOG FIELDS",logfields)
    [ignored_cols,range_info,bin_arr]= get_freq_count1(high_AF_thresh,selected,proto_fields,df,KL_thresh,freq_fname,mass_thresh)
else:
    [ignored_cols,range_info,bin_arr]= get_freq_count(high_AF_thresh,selected,proto_fields,df,KL_thresh,freq_fname,mass_thresh)


print(df.head())
# ()+1
np.save(proto_name+'/ignored', ignored_cols)

print("PR :",range_info,bin_arr)

print("IGNORED",ignored_cols)
print("RANGES",range_info)
print("BINS",bin_arr)


print(df.head())

cont_cols=[*range_info.keys()]

print("CONTS",cont_cols,bin_arr.keys())
sigs_fname=proto_name+'/sigs.npz'

print("DF",df.shape,"Num",len(selected))

print("DF",df.shape,"Num",len(selected))

if df.shape[0] ==0:
    print("NO Queries with high AF")
else:

    [tempDF,handle_columns,signatures] = signature_matching(df,ignored_cols,range_info,bin_arr,sigs_fname,high_AF_thresh,proto_name)
    tempDF.sort_values(by=['amp_fac'], ascending=False,inplace=True)
    print(signatures)
    DAG_setcover(proto_name,0)
    DAG_setcover(proto_name,1)
fig_gen("",1,proto_name)
fig_gen("",0,proto_name)

