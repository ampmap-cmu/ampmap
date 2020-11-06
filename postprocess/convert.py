import pandas as pd

import sys

# ppath=sys.argv[1]
# df = pd.read_csv(ppath)

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
    # df.to_csv(ppath,index=False)
