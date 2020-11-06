import pandas as pd

import sys
from common import *
import configparser,json,os
import glob as glob
import ntpath
import pandas as pd

def read_protodata(folder_name,mdd=0):
    fields=['amp_fac','server_id']
    folder_dir=folder_name+'query/*/*'
    folders=glob.glob(folder_dir)
    eachfile=folders[0]
    if mdd==1:
        print("NOTH")
    else:
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
        print(cnt,len(folders),len(total_data),fields)
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
    if mdd==1:
        # df.drop(df.columns.difference(['amp_fac','server_id']), 1, inplace=True)
        df = df.filter(['amp_fac','server_id'])

    return df
# print(df.shape)
# print(df.head())

def conv(df):
    did= df.groupby(['server_id'])['amp_fac'].max().reset_index()
    did.sort_values(by=['amp_fac'], ascending=False,inplace=True)
    did.reset_index(drop=True,inplace=True)
    did['new_IP'] = did.index
    did['new_IP'] = 'IP_' + did['new_IP'].astype(str)
    print(did)
    # servers=np.unique(df['server_id'].values
    del did['amp_fac']
    di = did.to_dict('list')
    print(di,"Done")

    # new_dict={}

    serverid_list = di['server_id']
    mapname_list=di['new_IP']

    dictionary = dict(zip(serverid_list, mapname_list))

    print(dictionary)

    df['server_id'] = df['server_id'].map(dictionary)
    print(df)
    return df


config = configparser.RawConfigParser()   



# l1=['ntp_cnt_or','ntp_pvt_or','ntp_normal_or']

# l1=['ntp_cnt','ntp_pvt','ntp_normal']
# l1=['snmp_next_or','snmp_bulk_or','snmp_get_or']
# l1=['snmp_next','snmp_bulk','snmp_get']
l1=['memcached']

configFilePath ="config/params.py"
print(configFilePath)
config.read(configFilePath)
did = pd.DataFrame()
for e1 in l1:
    print(e1)
    to_include =e1
    details_dict = dict(config.items(to_include))
    print(details_dict)
    proto_name = details_dict['proto_name']

    folder_name="/Users/rahulans/Google Drive (rahulans@andrew.cmu.edu)/cmu/ampmap-code/analysis/generate_signatures/zips/"+proto_name+"/"
    print(folder_name)
    df=read_protodata(folder_name,1)
    print(df.shape)
    print(df)

    # ()+1
    did = did.append(df)
    # ()+1

did= df.groupby(['server_id'])['amp_fac'].max().reset_index()
did.sort_values(by=['amp_fac'], ascending=False,inplace=True)
did.reset_index(drop=True,inplace=True)
did['new_IP'] = did.index
did['new_IP'] = 'IP_' + did['new_IP'].astype(str)
print(did)
# servers=np.unique(df['server_id'].values
del did['amp_fac']
di = did.to_dict('list')
print(di,"Done")

# new_dict={}

serverid_list = di['server_id']
mapname_list=di['new_IP']

dictionary = dict(zip(serverid_list, mapname_list))

print(dictionary)
# mode = int(sys.argv[1])



# if mode ==1:
#     to_include ='group1'
# elif mode==2:
#     to_include='group2'
# elif mode==3:
#     to_include='group3'
# else:
#     to_include='group4'
# configFilePath ="config/params.py"
# print(configFilePath)
# config.read(configFilePath)
# details_dict = dict(config.items(to_include))
# print(details_dict)

# ()+1
# ppath=sys.argv[1]
# df = pd.read_csv(ppath)
# ndf = conv(df)
# print(ndf)

# # ()+1
# ndf.to_csv(ppath,index=False)
for e1 in l1:
    print(e1)
    to_include =e1
    details_dict = dict(config.items(to_include))
    print(details_dict)
    proto_name = details_dict['proto_name']

    folder_name="/Users/rahulans/Google Drive (rahulans@andrew.cmu.edu)/cmu/ampmap-code/analysis/generate_signatures/zips/"+proto_name+"/"
    print(folder_name)
    df=read_protodata(folder_name)
    print(df.shape)
    print(df)
    df['server_id'] = df['server_id'].map(dictionary)

    # ()+1
    newfolder_name='data_dir/'+proto_name
    if not os.path.exists(newfolder_name):
        os.makedirs(newfolder_name)
    newfpath=newfolder_name+  "/complete_info.csv"
    df.to_csv(newfpath,index=False)
