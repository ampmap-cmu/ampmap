import numpy as np
import glob
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.pylab as pl
import seaborn as sns
from itertools import cycle, islice
import matplotlib.pyplot as plt
import argparse 
import os 

parser = argparse.ArgumentParser()#help="--fields_path , --data_folder_name  --proto  ")
parser.add_argument('--out_dir', type=str, default="figs",  help = "figs is an example ")
parser.add_argument('--parsed_data', type=str, default="figs/dns/",  help = "figs is an example ")
args = parser.parse_args()

dnstypes = {
    0: "ANY",
    1: "A", 2: "NS", 3: "MD", 4: "MF", 5: "CNAME", 6: "SOA", 7: "MB", 8: "MG",
    9: "MR", 10: "NULL", 11: "WKS", 12: "PTR", 13: "HINFO", 14: "MINFO",
    15: "MX", 16: "TXT", 17: "RP", 18: "AFSDB", 19: "X25", 20: "ISDN", 21: "RT",  # noqa: E501
    22: "NSAP", 23: "NSAP-PTR", 24: "SIG", 25: "KEY", 26: "PX", 27: "GPOS",
    28: "AAAA", 29: "LOC", 30: "NXT", 31: "EID", 32: "NIMLOC", 33: "SRV",
    34: "ATMA", 35: "NAPTR", 36: "KX", 37: "CERT", 38: "A6", 39: "DNAME",
    40: "SINK", 41: "OPT", 42: "APL", 43: "DS", 44: "SSHFP", 45: "IPSECKEY",
    46: "RRSIG", 47: "NSEC", 48: "DNSKEY", 49: "DHCID", 50: "NSEC3",
    51: "NSEC3PARAM", 52: "TLSA", 53: "SMIMEA", 55: "HIP", 56: "NINFO", 57: "RKEY",  # noqa: E501
    58: "TALINK", 59: "CDS", 60: "CDNSKEY", 61: "OPENPGPKEY", 62: "CSYNC",
    99: "SPF", 100: "UINFO", 101: "UID", 102: "GID", 103: "UNSPEC", 104: "NID",
    105: "L32", 106: "L64", 107: "LP", 108: "EUI48", 109: "EUI64",
    249: "TKEY", 250: "TSIG", 251: "IXFR", 252: "AXFR", 
    255:"ANY",  256: "URI", 257: "CAA", 258: "AVC",
    32768: "TA", 32769: "DLV", 65535: "RESERVED"
}



col_name = "rdatatype"

plt.rcParams["font.family"] = "Verdana"
fname= args.parsed_data # 'figs_20200524/dns_10k/'

d0=pd.read_csv(fname+'d1_0')


d1=pd.read_csv(fname+'d1_1')
d2=pd.read_csv(fname+'d1_2')
d3=pd.read_csv(fname+'d1_3')

d0 = d0.rename(columns={'Unnamed: 0': 'rdatatype'} ) 
d1 = d1.rename(columns={'Unnamed: 0': 'rdatatype'} ) 
d2 = d2.rename(columns={'Unnamed: 0': 'rdatatype'} ) 
d3 = d3.rename(columns={'Unnamed: 0': 'rdatatype'} ) 


#d0["test"] = d0["rdatatype"]
d0["rdatatype"]  = d0.apply(lambda x: pd.Series(dnstypes[x["rdatatype"]] , index=["rdatatype"])   ,axis=1,  result_type = 'expand'  )  
d1["rdatatype"]  = d1.apply(lambda x: pd.Series(dnstypes[x["rdatatype"]] , index=["rdatatype"])   ,axis=1,  result_type = 'expand'  )  
d2["rdatatype"]  = d2.apply(lambda x: pd.Series(dnstypes[x["rdatatype"]] , index=["rdatatype"])   ,axis=1,  result_type = 'expand'  )  
d3["rdatatype"]  = d3.apply(lambda x: pd.Series(dnstypes[x["rdatatype"]] , index=["rdatatype"])   ,axis=1,  result_type = 'expand'  )  

d0.set_index("rdatatype",inplace= True)
d1.set_index("rdatatype",inplace= True)
d2.set_index("rdatatype",inplace= True)
d3.set_index("rdatatype",inplace= True)


interval = ["AF 10-30", "AF 30-50", "AF 50-100"]
# bad = np.load(fname+'d3_.npy')
# print(bad)
ll=[]

for rtype ,r in d0.iterrows():
    curd={}
    curd[interval[0]]= r['NumVulnerableServers'] -  d1.loc[rtype]['NumVulnerableServers']  
    curd[interval[1]]= d1.loc[rtype]['NumVulnerableServers']  - d2.loc[rtype]['NumVulnerableServers']
    curd[interval[2]]= d2.loc[rtype]['NumVulnerableServers'] - d3.loc[rtype]['NumVulnerableServers'] 
    curd["Total"] = curd[interval[0]] +  curd[interval[1]] + curd[interval[2]]
    curd['AF > 100']= d3.loc[rtype]['NumVulnerableServers'] 
    #curd['rdatatype'] = 
    #assert( r["rdatatype-code"] == d1["rdatatype-code"] == d2[""])
    curd["recordtype"] = rtype 
    ll.append(curd)


df=pd.DataFrame(ll)
df.sort_values("Total", inplace=True, ascending=False)
 





f, ax1 = plt.subplots(1, 1)
# my_colors = ['g', 'b']*5 # <-- this concatenates the list to itself 5 times.
# my_colors = [(0.5,0.1,0.5), (0.75, 0.75, 0.25)]*5 # <-- make two custom RGBs and repeat/alternate them over all the bar elements.
# my_colors = [(x/10.0, x/20.0, 0.75) for x in range(len(df))]
# print(my_colors)

cmap = pl.cm.PRGn

# # Get the colormap colors
my_cmap = cmap(np.arange(cmap.N))

my_cmap[0,-1]=0.5
# print(my_cmap[:,-1])
# Create new colormap
my_cmap = ListedColormap(my_cmap)



df.plot.bar(x = "recordtype", y = interval,  stacked=True,figsize= (20,8), fontsize=10,ax=ax1, cmap="Paired", \
            edgecolor='black', linewidth=2) #color=palette)

       
ax1.tick_params(axis='y',direction='out', length=10, labelsize=16)
ax1.tick_params(axis='y',which='minor', length=8, labelsize=15)

# ax1.tick_params(axis='y',direction='out', length=8, which='minor')
ax1.grid(linestyle='-', which='major')
ax1.set_axisbelow(True)
f1=30
#  ax1.set_xlabel( x_label , fontsize=23)
f2=30
f3=30
plt.setp(ax1.get_xticklabels(), rotation=90,fontsize=f2);
#labels=[1,2,5,6,12,15,16,17,18,24,25,28,29,33,35,36,37,39,41,42,43,44,45,46,47,48,49,50,51,52,55,59,60,61,249,250,251,252,255,256,257,32768,32769]
#print(len(labels))
#ax1.set_xticklabels(labels)

ylabel = list(range(0, 7000, 1000))
ylabel_write = [ int(i/10000 * 100) for i in ylabel]

#ax1.set_yticklabels(labels)
ax1.set_yticklabels(ylabel_write, fontsize=30)




ax1.legend(loc='upper center', fancybox=True, shadow=True, ncol=4,fontsize=f3,labelspacing=0)
plt.xlabel("Record Types",fontsize=f1 ) # fontweight="semio"  )
plt.ylabel("% Of Vunlerable Servers",fontsize=f1+ 3 )

plt.gca().xaxis.set_label_coords(0.8, -0.4 ) 


plt.tight_layout()


ax1.set_ylim([0, 5500])

plt.gca().get_xticklabels()[0].set_color("red")
plt.gca().get_xticklabels()[1].set_color("red")
plt.gca().get_xticklabels()[0].set_weight("semibold")
plt.gca().get_xticklabels()[1].set_weight("semibold")

plt.gca().get_xticklabels()[13].set_size(f1-3)
plt.gca().get_xticklabels()[18].set_size(f1-3)
plt.gca().get_xticklabels()[22].set_size(f1-3)


#print(a )

leg = ax1.get_legend()
fname = os.path.join(args.out_dir, "Figure13_dns_stacked_bar.pdf") 
plt.savefig(fname , bbox_inches='tight')

