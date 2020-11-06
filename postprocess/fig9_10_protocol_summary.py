## need to do NTP and 
import os 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns 
import numpy as np
from collections import OrderedDict 

from collections import defaultdict 


# Protocol maps 

root_dir = "data_dir"
intermediate_dir = "intermediate_files"
fig_dir = "figs"




if not os.path.exists(intermediate_dir):
    os.makedirs(intermediate_dir)
    



if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
    
    


# stores the mapping of  proto name to actual directory   

proto_alias = defaultdict(dict)

proto_alias["dns"] = { "dns" : "sig_KL0.08_AF_30query_searchout_DNS_20200516"} 
proto_alias["ntp-or"] = { "ntpprivate-or" : "query_searchout_NTPPrivate_20200518/", 
                         "ntpcontrol-or": "query_searchout_NTPControl_AND_20200608", 
                         "ntpnormal-or": "query_searchout_NTPNormal_OR_20200529/"} 

proto_alias["ntp-and"] = { "ntpprivate-and" : "query_searchout_NTPPrivate_AND_20200526", 
                         "ntpcontrol-and": "query_searchout_NTPControl_AND_20200608", 
                         "ntpnormal-and": "query_searchout_NTPNormal_AND_20200608"} 

proto_alias["snmp-or"] = { "snmpbulk-or": "query_searchout_SNMPBulk_20200521", 
                           "snmpnext-or": "query_searchout_SNMPNext_OR_20200608", 
                           "snmpget-or": "query_searchout_SNMPGet_OR_20200531" }


proto_alias["snmp-and"] = { "snmpbulk-and": "query_searchout_SNMPBulk_AND_20200528", 
                           "snmpnext-and": "query_searchout_SNMPNext_AND_20200601", 
                           "snmpget-and": "query_searchout_SNMPGet_AND_20200529"} 

proto_alias["chargen"] = { "chargen": "query_searchout_chargen_20200524" }
proto_alias["ssdp"] = { "ssdp": "query_searchout_SSDP_20200523" }

proto_alias["memcached"] = { "memcached": "query_searchout_memcached_20200526" }

                           
proto_agg_list = ["dns", "ntp-and", "snmp-and", "ssdp" , "chargen", "memcached"] 







def process_cdf(data): 
    maxAF = pd.Series(data.groupby(["server_id"])["amp_fac"].max() )
    return pd.Series(maxAF)


'''
    This one generates the intermediate file that is find the maximum AF for each server 
     writes to an intermediate file 
'''
for proto, proto_lst in proto_alias.items(): 
    
    lol = []   
    cdf_df_fname = os.path.join(intermediate_dir , "{}-cdf-list-1.csv".format(proto)  )
   
    if  os.path.exists(cdf_df_fname):
        print(" File exists  ", cdf_df_fname )
        continue 
    
    
    for sub_proto, proto_dir in proto_lst.items(): 
    
        c_file = os.path.join(root_dir, proto_dir, "complete_info.csv" )
        sdf = pd.read_csv(c_file) 

        print("Read " , proto, sdf.shape )
        lol.append(sdf)
        
    df = pd.concat(lol )
    print("df shape ", df.shape )

    
    cdf_df = process_cdf(df)
    cdf_df.to_csv(cdf_df_fname,index=False)


        
        
lol = [] 

for proto, _ in proto_alias.items(): 
    #print(proto, alias )
    cdf_df_fname = os.path.join(intermediate_dir, proto +"-cdf-list-1.csv" )

    cdf_df=pd.read_csv(cdf_df_fname) 
    cdf_df = cdf_df.astype(float)
    cdf_df.columns = [proto]
    lol.append(cdf_df )

df = pd.concat(lol, axis=1)
print("Read all DF " , df.shape  )





''' 
Generate Fig 9 
'''

plt.style.use(['seaborn-whitegrid', 'seaborn-paper'])
             

plt.clf()
plt.figure(figsize=(25, 5))
ax = sns.boxplot( data=df,  linewidth=5, palette="Set3", dodge=True,width=0.8,\
                 showmeans=True,fliersize=5  ) #  figsize=(15,6))

#ax.set_xticks([i for i in range(top_index_num)], xlabels) #, fontsize=18)
ax.set_ylabel("Max Amplification Factor", fontsize=30)
ax.set_xlabel("Protocols", fontsize=30, labelpad=5)

#ax.set_xlabel("Query Patterns (QP) ranked by {}".format(rank_by), fontsize=25, labelpad=20)
ax.tick_params(axis='x', labelsize=29)
ax.tick_params(axis='y', labelsize=25)
plt.yscale("log")


medians = df.median()
median_labels = [str(np.round(s, 2)) for s in medians]

print(median_labels)
pos = range(len(medians))
for tick,label in zip(pos,ax.get_xticklabels()):
    ax.text(pos[tick], medians[tick] , median_labels[tick], 
            horizontalalignment='center', size=22 , color='white', va="center", ha="center", weight='bold', \
            bbox=dict(facecolor='#445A64'))


outfile = os.path.join(fig_dir, "Fig9_max_AF_protocols.pdf" )
plt.savefig(outfile,bbox_inches='tight')






''' 
Generate Fig 10a 
'''




''' 
Generate Fig 10a  : Aggregate Summary
'''
def construct_aggregate_summary(  ): 

    d = {}
    
    af_ranges = [(None,10), (10,30), (30,50), (50,100), (100,None)]
    af_legends = ["<10 (low risk)", "[10,30)", "[30,50)", "[50,100)", ">=100"]
    
    tmp_d = OrderedDict()  
    for proto in proto_agg_list: #proto_list: 
        
        # Just need to plot 
        if "or" in proto: 
            continue 
        
        
        cdf_df_fname=os.path.join( intermediate_dir, proto +'-cdf-list-1.csv')
        exists = os.path.isfile(cdf_df_fname)
        if exists:
            print("Skipping reading the parsed file")
            cdf_df=pd.read_csv(cdf_df_fname)
            print("read the csv")
        else:
        
            print("proto ", proto , cdf_df_fname)
            cdf_df=process_cdf(df)
        cdf_df.columns = ["amp_fac"]
        print(cdf_df.head())

        af_values = [] 
        for r in af_ranges: 
            r_left = r[0]
            r_right = r[1]

            if r_left == None: 
                query = cdf_df.query("amp_fac < @r_right ")
            elif r_right == None: 
                query = cdf_df.query("amp_fac >= @r_left" ) 
            else: 
                query = cdf_df.query("amp_fac >= @r_left and amp_fac < @r_right" )
            af_values.append(len(query))            

        norm = [float(i)/sum(af_values)*100 for i in af_values]
        d[proto] = norm

    print(d)
    df=pd.DataFrame(data=d).transpose()
    df.columns = af_legends
    return df

aggregate_df =construct_aggregate_summary() 
print("aggregate df ", aggregate_df.shape  )
#aggregate_df = aggregate_df.reindex(index=aggregate_df.index[::-1])

df = aggregate_df 



df.rename(columns={'<10 (low risk)':'< 10'}, inplace=True)


ax=df.plot.bar(stacked=True,fontsize=26,width=0.8, cmap="Set3", figsize=(6,14), \
    edgecolor='black', linewidth=1) 


ax.set_ylabel("% of Servers",fontsize=28)

ax.set_ylim([0,100])

ax.set_xlabel('Protocols',fontsize=28,labelpad=-6)





n_bars = int(df.shape[0])

hatches = ['dummy', '.', '\\', 'x', '+', '*', '+', 'O', '.']

h_index = 0 

for i, patch in enumerate(ax.patches):
    if i % n_bars == 0: 
        h_index += 1 
    hatch = hatches[h_index]
    patch.set_hatch(hatch)


plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
plot_margin = 0.3
plt.grid(True, axis='y')
x0, x1, y0, y1 = plt.axis()
plt.axis((x0 - 0,
          x1 + 0,
          y0 - 0,
          y1 + plot_margin))

ax.legend(loc='upper center', fontsize=26, columnspacing=1,  ncol=2, bbox_to_anchor=(0.48, 1.3))

plt.tight_layout()

fig_name = os.path.join( fig_dir , "Fig10_AF_Distribution_yr2020.pdf")
plt.savefig(fig_name,  bbox_inches='tight',pad_inches=0)




