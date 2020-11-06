import numpy as np
import pandas as pd
import ntpath
import glob,json,os,sys,time
import pandas as pd
#from sklearn.neighbors import KernelDensity


import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 

#import matplotlib.pyplot as plt

# from sys import platform as sys_pf
# if sys_pf == 'darwin':
#     import matplotlib
#     matplotlib.use("TkAgg")

plt.clf()

import numpy as np

#sys.path.append(os.path.abspath('../new-src/'))
import libs 

import operator
from scipy import stats
import scipy.stats as ss
from itertools import groupby

from collections import OrderedDict
import argparse
from collections import OrderedDict
import itertools, math
from textwrap import wrap
import seaborn as sns
from libs.prior_work_sig import *


from pylab import *
from matplotlib.colors import ListedColormap


NUM_DIVISION_AF = 3 



RAW_DATA_STR = "No Filter" #"Raw Data" 
width=0.3
FIG_SIZE = (15,20)
fontsize = 18
label_fontsize=15
second_c = 'orange'  








def read_protodata(folder_name):
    fields=['amp_fac','server_id']
    folder_dir=folder_name+'query/*/*'
    folders=glob.glob(folder_dir)
    print(folders)
    eachfile=folders[0]
    count = 0 
    with open(eachfile) as f:
        data = json.load(f)
        cur_server=[]
        for ind_data in data:
            for attribute, value in ind_data['fields'].items():
                fields.append(attribute)
            break
        count = count + 1 
        if count % 1000 == 0 : 
            print("Read {} files ".format(count))
    df = pd.DataFrame(columns=fields)
    print(df.head())
    rows_list = []
    stringlist=df.columns.values
    total_data=[]
    for count, eachfile in enumerate(folders):
        if count % 1000 == 0: 
            print("Stored {} files ".format(count) )
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


def make_dir(directory): 
    if not os.path.exists(directory):
        os.makedirs(directory)

    

def compute_num_distinct_server(df):
    #print("Distinct Servers : ", len(query['server_id'].unique()) 
    return len(df['server_id'].unique()) 



def compute_num_distinct_fields(df, field_name): 
    return len(df[field_name].unique())

# def compute_num_distinct_datatypes(df): 
#     return len(df['rdatatype'].unique())

#Directly count the frequency 
def compute_heavy_hitter_raw_queries(rdatatype, df, field_name): 
    d = OrderedDict() 
    count = df.groupby(field_name).size() 
    count.index = count.index.astype(int)
    count.sort_index(inplace=True)
    #for i, v in count.items():
        #print('index: ', i, 'value: ', v)    
    return count.to_frame() #to_dict(OrderedDict)

def compute_heavy_hitter_num_distinct_server(rdatatype, data, field_name): 
    #Initialize the dictionary 
    d = dict()#OrderedDict() 
    for i in rdatatype: 
        d[i] = 0 
    
    #For each server, count the distinct Rdatatype 
    c = 0 
    for s in data['server_id'].unique(): 
        #First filter based on the server id 
        df_server = data.query("server_id == @s") 

        #if type(df_server[field_name][0]  
        try: 
            unique_rdatatype = [int(i) for i in df_server[field_name].unique()] 
        except: 
            unique_rdatatype = df_server[field_name].unique()

        #print("unique rdatatype ", unique_rdatatype)
        for rd in unique_rdatatype: 
            d[rd] = d[rd] + 1 
        #c = c + 1 
    series = pd.Series(d)
    return series.to_frame()




def get_AF_interval(df): 
    maxAF = df['amp_fac'].max() 
    print("Max AF is ", maxAF )

    #Need to eliminate the max b/c we are searching for AF > x where x should not be the max AF
    #AF_list = [int(i) for i in np.linspace(10,maxAF,NUM_DIVISION_AF + 1 )[:-1]]
    AF_list = [10,30,50,100]

    if maxAF <= 52: 
        AF_list = [10,30] 
    elif maxAF <= 100: 
        AF_list = [10,30, 50]
    else: 
        AF_list = [10,30,50,100]
    #elif maxAF <= 100: 
    #    AF_list = 

    #AF_list.append( maxAF)
    AF_interval = []
    for a in range(0,len(AF_list)): 
        if a == len(AF_list) - 1: 
            AF_end  = math.ceil( maxAF )
        else:  
            AF_end = AF_list[a + 1 ]
            
        #AF_interval.append( "AF {}-{}".format(AF_list[a], AF_end) )  
        AF_interval.append( "AF ≥ {}".format(AF_list[a]) )  
    
    print(AF_list )
    print(AF_interval)
    #sys.exit(1)
    return AF_list, AF_interval 



def plot_num_distinct(df, AF_list,  title, x_label, figname, color): 

    fig, ax1 = plt.subplots()

    columns = df.columns
    width_new  = 0.6
    #print(df )
    df[columns[0]].plot(kind='bar',color=color ,width=0.5,   figsize=(10,7), fontsize=20)

    #ax1.legend(loc='upper left', fontsize=label_fontsize, bbox_to_anchor=(0.1, 1.0)) 
    
    ax1.set_ylabel( df.columns[0] , fontsize=20  )
    
    #fig.suptitle(title , fontsize=25)
    fig.suptitle("\n".join(wrap(title,60))  , fontsize=18, y=0.94)

    ax1.set_yscale('log')
    #ax1.tick_params(axis='x' , rotation=0)
    
    ax1.tick_params(axis='y',direction='out', length=10, labelsize=18)
    ax1.tick_params(axis='y',direction='out', length=7, which='minor', labelsize=18)
    ax1.grid(linestyle='-', which='major')
    ax1.set_axisbelow(True)
        
    ax1.set_xlabel( x_label , fontsize=20)

    #plt.show()
    fig.savefig(figname, bbox_inches="tight")
    plt.close()
    return 

    

def plot_hh(hh_list, AF_interval_list, title, x_label, figname, bad_values, legends,fig_folder_path, \
    output_count=False): 
    #f = os.path.join(fig_folder_path,'d1_{}'.format(len(hh_list)) )
    #np.save(f)

    if output_count: 
        
        for i in range(0, len(hh_list)): 
            f = os.path.join(   fig_folder_path, 'd1_'+str(i)   )
            print(hh_list[i] )
            hh_list[i].to_csv(f,index=True)
    fig = plt.figure()
    
    len_total = len(hh_list)


    colors = sns.color_palette("muted")
    good_color = colors[0]
    bad_color = colors[1]

  
    #print("AF list is ", AF_list)
    for i in range(0, len_total):
        plot_id =  "{}{}{}".format(len_total, "1" , i+1) 
        print("Plot id is ", plot_id )
        ax1 = fig.add_subplot( int(plot_id) )
        #ax1_2 = ax1.twinx() 
        
        colors_bar = []
        for j, r in  hh_list[i]['NumVulnerableServers'].iteritems(): 

            if j in bad_values: 
                colors_bar.append(bad_color)     #[i] = 'darkred'
            else: 
                colors_bar.append(good_color)
                #colors[i] = 'green'
        colors_bar = tuple(colors_bar)
        #hatches = tuple(hatches)
        print( len(hh_list[i]['NumVulnerableServers']) )
        custom_lines = [Line2D([0], [0], color=good_color, lw=7),
                        Line2D([0], [0], color=bad_color, lw=7)]
        #hh_list[i]['NumRawQueries'].plot(kind='bar', rot=90,  ax=ax1, figsize=FIG_SIZE, fontsize=fontsize, position=1, width=width)
        hh_list[i]['NumVulnerableServers'].plot(kind='bar',rot=90, color=colors_bar, figsize= (15,18), fontsize=fontsize,  width=0.6)

        ax1.legend(custom_lines, legends, loc='upper left', ncol=1, fontsize=13, bbox_to_anchor = (0, 1.05))
        ax1.set_ylabel( hh_list[i].columns[0] , fontsize=20 )      
        ax1.set_yscale('log')
       
        ax1.tick_params(axis='y',direction='out', length=10, labelsize=16)
        ax1.tick_params(axis='y',which='minor', length=8, labelsize=15)

        ax1.tick_params(axis='y',direction='out', length=8, which='minor')
        ax1.grid(linestyle='-', which='major')
        ax1.set_axisbelow(True)

        #ax.tick_params(axis='both', which='major', labelsize=18)
        #ax.tick_params(axis='both', which='minor',labelsize=16)

        #ax.tick_params(axis='y',direction='out', length=10, labelsize=18
       
        ax1.set_title("AF Range : " +  str(AF_interval_list[i]), fontsize=23 )
        ax1.tick_params(axis='x', labelbottom=True)
        if i == (len_total-1):
            ax1.set_xlabel( x_label , fontsize=23)
    fig.suptitle(title ,y = 0.93, fontsize=25)
    #plt.show()
    fig.savefig(figname,  bbox_inches="tight")
    plt.close()
    return     


def plot_hh_individual(hh_list, AF_interval_list, title, x_label, figname, bad_values,  legends, proto ): 

    
    len_total = len(hh_list)

    #colors = sns.color_palette("muted")
    colors = sns.color_palette("Set1", n_colors=8)# desat=.5)

    good_color = colors[1]
    bad_color = colors[0]


    #legends = ["Not highlighted in prior work", "Known to cause high AF by prior work"]

  
    #print("AF list is ", AF_list)
    for i in range(0, len_total):
        fig = plt.figure()
        #plot_id =  "{}{}{}".format(len_total, "1" , i+1) 
        #print("Plot id is ", plot_id )
        ax1 = fig.add_subplot( 111 )
        #hh_list[i]['NumRawQueries'].plot(kind='bar', rot=90,  ax=ax1, figsize=FIG_SIZE, fontsize=fontsize, position=1, width=width)
        
        #colors = tuple(np.where(hh_list[i]["a"]>2, 'g', 'r'))

        #print(hh_list[i]['NumVulnerableServers'])
        #colors = tuple( np.where  )

        colors_bar = []
        hatches = [] 
        #colors2 = {} 
        count = 0 

        good_val = []
        #bad_color = 'b'
        #good_color = 'orange'

        for j, r in  hh_list[i]['NumVulnerableServers'].iteritems(): 

            if j in bad_values: 
                colors_bar.append(bad_color)     #[i] = 'darkred'
            else: 
                colors_bar.append(good_color)
 
                #good_val.append(True)

                #colors[i] = 'green'
        colors_bar = tuple(colors_bar)
        hatches = tuple(hatches)
        print(colors_bar, len(colors_bar))
        print( hh_list[i]['NumVulnerableServers'] )

        custom_lines = [Line2D([0], [0], color=good_color, lw=7),
                        Line2D([0], [0], color=bad_color, lw=7)]

        print(hh_list[i]['NumVulnerableServers'])
        hh_list[i]['NumVulnerableServers'].plot(kind='bar',rot=90, figsize= (15,3.8), fontsize=20,  width=0.6,  label=colors_bar, color=colors_bar)

        ax1.legend(custom_lines, legends, loc='upper left', ncol=1, fontsize=16, bbox_to_anchor = (0, 1.10))
        # ax1.legend(custom_lines, legends, loc='upper left', ncol=1, fontsize=13)
        # ax1.set_ylabel( hh_list[i].columns[0] , fontsize=17)      
        # change ylabel name
        ax1.set_ylabel( "# of Vulnerable Servers" , fontsize=19)
        ax1.set_yscale('log')
        #ax1.set_hatch(hatches)
        ax1.tick_params(axis='y',direction='out', length=10, labelsize=18)
        ax1.tick_params(axis='y',which='minor', length=8, labelsize=18)

        ax1.tick_params(axis='y',direction='out', length=8, which='minor')
        ax1.grid(linestyle='--', which='major', axis='y')
        ax1.set_axisbelow(True)

       
        ax1.tick_params(axis='x', labelbottom=True, labelsize=16)
        #if i == (len_total-1):
        if proto == "DNS":
            ax1.set_xlabel( x_label , fontsize=29, labelpad=-10)
        elif proto == "NTPPrivate":
            ax1.set_xlabel( x_label , fontsize=29)
        fig.suptitle("{} (AF Range : {}) ".format(title, AF_interval_list[i]) ,fontsize=19)
        
        fig.savefig(figname.replace(".pdf", "_{}.pdf".format(i) ) ,  bbox_inches="tight") # dpi=800)
    #()+1
    return     



def generate_combinations(conditions, num_elem ): 
    lst = [] 
    lst = lst + list(itertools.combinations(conditions, num_elem)) 
    return lst 




def construct_cond_str(cond , combined_conds): 
    print(cond )
    print(combined_conds)
    if len(cond) == 0: 
        return "True"
    if len(cond) == len(combined_conds): 
        join_cond = " and ".join(cond)
        return " not ({})".format(join_cond)
    not_included = [x for x in combined_conds if x not in cond]
    print("not included ", not_included)
    part_cond1 = " and ".join(not_included) 
    part_cond2 = " and ".join(cond)
    return "not ({}) and {}".format( part_cond2, part_cond1 )


def construct_cond_str_v2(cond ): 
    print(cond )
    if len(cond) == 0: 
        return "True"
    join_cond = " and ".join(cond)
    return " not ({})".format(join_cond)
    

    

def __make_out_dirs():
    #dirs = ["configs", "inifiles", "pcaps", "graphviz", "logs", "intermediate_out", "click_logs"]
    paths = [os.path.join(out_dir, d) for d in dirs]

    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)


def get_fig_name(fig_folder_path, num_itr, header ): 
    
    fig_name = "itr{}_{}.pdf".format( num_itr, header)
    print("FIG NAME " , fig_name)
    return os.path.join(fig_folder_path, fig_name) 



def join_fig_name(figname): 
    
    return os.path.join(fig_folder_path, fig_name) 



def plot_group_bar( df,  title, y_label, x_label, figname, num_column): 

    # Setting the positions and width for the bars
    pos = list(range(len(df.index))) 
    
    if num_column < 3: 
        width = 0.25
    else: 
        width = 0.1 
    print("pos is ", pos)
    # Plotting the bars
    fig, ax = plt.subplots(figsize=(10,7))

    print("COLUMNS " , df.columns)
    #colors = ['#EE3224', '#F78F1E', '#F78F1E', '#4169E1', '#2E8B57', ] 
    
    colors = sns.color_palette("Set1", n_colors=8)# desat=.5)
    for i  in range(num_column):# len(df.columns)): 
        # Create a bar with pre_score data,
        # in position pos,
        pos_new = [p + width * i for p in pos]
        data = df[df.columns[i]]
        print("pos new ", pos_new)
        print("plot data ", data)

        plt.bar( pos_new, 
                #using df['pre_score'] data,
                data, 
                # of width
                width, 
                color = colors[i],
                # with alpha 0.5
                alpha=0.5, 
                #index 
                label=df.columns[i]) 

    # Set the y axis label
    ax.set_ylabel(y_label, fontsize=20)
    ax.set_xlabel(x_label, fontsize=20)
    ax.set_position([0.1,0.4,0.8,0.5])
    
    if num_column < 3 : 
        ax.legend(loc='lower center', fontsize=15, bbox_to_anchor=(0.5, -0.45))     
    else: 
        ax.legend(loc='lower center', fontsize=15, bbox_to_anchor=(0.5, -0.8))     
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor',labelsize=16)

    ax.tick_params(axis='y',direction='out', length=10, labelsize=18)
    
    ax.set_yscale('log')
    ax.grid(linestyle='-', which='both')

    ax.set_axisbelow(True)
        
    
    # Set the chart's title
    #ax.set_title(title, fontsize=fontsize)

    # Set the position of the x ticks
    #factor = width * num_column # len(df.columns)
    #ax.set_xticks([p + (1/num_column) * width for p in pos])
    i = float((num_column )/2.0)
    pos_new = [p + width * i for p in pos] 
    ax.set_xticks(pos_new)   
    
    print("axis post ", pos_new)

    # Set the labels for the x ticks
    ax.set_xticklabels(df.index, fontsize=fontsize)

    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos)-width, max(pos)+width*num_column)
    #plt.ylim([0, max(df['pre_score'] + df['mid_score'] + df['post_score'])] )

    if title != None: 
        fig.suptitle("\n".join(wrap(title,60))  , fontsize=20)
    #plt.show()
    fig.savefig(figname)
    


#YUCHENG this function plots FIG 15
def compute_percent_retention( data,  title, y_label, x_label, figname, num_column, base_column): 

    # Setting the positions and width for the bars
    pos = list(range(len(data.index))) 
    output_file = figname.replace(".pdf", ".txt")
    out_f = open(output_file, "w")

    if num_column <- 3: 
        width = 0.25
    else: 
        width = 0.2
    print("pos is ", pos)
    # Plotting the bars
    fig, ax = plt.subplots(figsize=(10,4))

    print("df data is ", data, base_column)
    #colors = ['#EE3224', '#F78F1E', '#F78F1E', '#4169E1', '#2E8B57', ] 
    base_data = data[base_column] 
    print("base data is ", base_data )
    
    hatches = ['||', '//', '\\\\', '/', 'x', '+', '\\', '.']
    df = data.drop([base_column], axis=1)
    
    print([base_column])
    
    
    print("\n\n\nDF droppped ", df )
    
    colors = sns.color_palette("Set1", n_colors=8) # desat=.5)
    for i  in range(0, num_column  ):# len(df.columns)): 
        # Create a bar with pre_score data,
        # in position pos,
        pos_new = [p + width * i for p in pos] 
        print("For column ",df.columns[i])
        data = []# (base_data -  df[df.columns[i]]) / base_data * 100.0
        
        for j in range(len(df[df.columns[i]])):
            cur_data = df[df.columns[i]][j]
            print( "( {} / {} )".format(cur_data, base_data[j]) )
            data.append(  cur_data / base_data[j] * 100  )
        
        print("pos new for i ", pos_new , i )
        print("plot data ", data)

        plt.bar( pos_new, 
                #using df['pre_score'] data,
                data, 
                # of width
                width, 
                color = colors[i],
                # with alpha 0.5
                alpha=0.5, 
                #index 
                label=df.columns[i], 
                hatch=hatches[i]) 
        out_f.write("{}: {}\n".format("Index ", df.index)) 
        out_f.write("{}: {}\n".format(  df.columns[i], data )) 

        print("Index is " , df.index)
        print("Data is " , df.columns[i], data)
    # Set the y axis label
    ax.set_ylabel(y_label, fontsize=23, labelpad=-5)
    ax.set_xlabel(x_label, fontsize=26)
    #ax.set_position([0.1,0.4,0.8,0.5])

    
    #print(x_label, y_label)
    ax.set_position([0.1,0.15,0.87,0.65])
    out_f.close() 
    # if num_column < 3 : 
    #     ax.legend(loc='lower center', fontsize=15, bbox_to_anchor=(0.5, -0.45))     
    # else: 
    #     ax.legend(loc='lower center', fontsize=13, bbox_to_anchor=(0.5, -0.85))   
    ax.legend(loc='upper center', fontsize=20,ncol=3,  bbox_to_anchor=(0.5, 1.3))      
    ax.tick_params(axis='both', which='major', labelsize=27)
    ax.tick_params(axis='both', which='minor',labelsize=27)

    ax.tick_params(axis='y',direction='out', length=10, labelsize=18)
    
    #ax.set_yscale('log')
    #ax
    ax.grid(linestyle='-', which='both')

    ax.set_axisbelow(True)
        
    
    # Set the chart's title
    #ax.set_title(title, fontsize=fontsize)

    # Set the position of the x ticks
    #factor = width * num_column # len(df.columns)
    #if num_column == 1: 
    
    i = float((num_column )/2)
    pos_new = [p + width * i for p in pos] 
    print("axis post ", pos_new)
    ax.set_xticks(pos_new)   
    #    ax.set_xticks([p + 0.25 * width for p in pos])
    #elseif num_column == 1: 
    #else: 
    #    ax.set_xticks([p + (1/(num_column+1)) * width for p in pos])
    

    # Set the labels for the x ticks
    ax.set_xticklabels(df.index, fontsize=22)

    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos)-width, max(pos)+width*num_column)
    #plt.ylim([0, max(df['pre_score'] + df['mid_score'] + df['post_score'])] )

    if title != None: 
        fig.suptitle("\n".join(wrap(title,60))  , fontsize=20)
    #plt.show()
    fig.savefig(figname, bbox_inches='tight')



#Data in a list of list 
def convert_to_dataframe(data, index, column): 
    df = pd.DataFrame(data)
    df.columns = column
    df.index = index
    return df 



#Calculate the distinct server that has that 
def plot_query_analysis_v2(df, AF_list, AF_interval_list, signature, fig_folder_path) :
    #cond = [] 
    itr = 0
    
    server_color = 'green'
    other_color = 'orange'
    
    
    print(AF_list)
    print(AF_interval_list)
    #()+1


    conditions_list =  signature.conditions    
    #num_elements = signature.num_elements
    field_check = signature.feature_fields
    field_label = signature.feature_label    

    #if type(df[field_check][0]) == int: 
    try: 
        field_complete = sorted([int(i) for i in  df[field_check].unique()] )
    except: 
        field_complete = df[field_check].unique()

    print("\t\t", field_complete)
    print("field check ", field_check)
    print("field label ", field_label) 

    filter_eff_server = [[] for x in range(0, len(AF_list) )]
    #filter_eff_rdata =  [[] for x in range(0, len(AF_list) )]
    filter_title = []

    
    hh_info_files = os.path.join(fig_folder_path, "signature_info.txt") 

    hh_info = []

    print(conditions_list )
    for itr, conditions  in enumerate(conditions_list): 
        #cond = combined_conds[itr]
        fields_cond_str = construct_cond_str_v2(conditions)
        print("\n\nfieids_cond_str ", fields_cond_str)
        filter_title.append(signature.legends[itr])
        #continue
        num_dist_server_list = [] 
        num_dist_other_list = [] 
        #return

        hh = [] 
        hh_info.append( "ITR  {} : {}".format(itr, fields_cond_str) )
        for a in range(0,len(AF_list)): #AF in AF_list: 
            AF = AF_list[a]
            #amp_cond = ""
            #if a == len(AF_list) - 1 : 
            amp_cond = "amp_fac >= @AF "
            #else: 
                #AF_next = AF_list[a+1]
            #    amp_cond = "amp_fac >= @AF" # and amp_fac <= @AF_next "


            cond_str  = "{} and  {} ".format(amp_cond, fields_cond_str) 
            print("COND STR ", cond_str)

            query=df.query( cond_str )
            query.sort_values(by=['amp_fac'], ascending=False,inplace=True)
            print("done querying ")

            num_raw_queries = query.shape[0] 
            #Figure otu num 
            num_server =  compute_num_distinct_server(query) 
            num_field_check =  compute_num_distinct_fields(query,  field_check) #compute_num_distinct_datatypes(query) 

            num_dist_server_list.append(num_server )
            num_dist_other_list.append(num_field_check )

            print("\tFIELD COMPLETE " , field_complete)
            hh_distinct_servers =  compute_heavy_hitter_num_distinct_server(field_complete, query, field_check)     
            hh_distinct_servers.columns = ["NumVulnerableServers"]
            hh.append(hh_distinct_servers)
            
            
            filter_eff_server[a].append( num_server )    

        print(filter_eff_server)
        #()+1
        color = server_color 
        df_distinct_s = pd.DataFrame(num_dist_server_list)
        df_distinct_s.columns = ['NumDistinctServer' ]
        df_distinct_s.index = AF_interval_list 
        figname =get_fig_name(fig_folder_path, itr, "NumDistinctServer" )
        plot_num_distinct(df_distinct_s, AF_list, "Number of Distinct Servers" , "AF Range" , figname, color)   

        color = other_color
        df_distinct_other = pd.DataFrame(num_dist_other_list)
        df_distinct_other.columns = ['Num Distinct ' + field_label.replace(" " , "" ) ]
        df_distinct_other.index = AF_interval_list 
        figname =get_fig_name(fig_folder_path,  itr, "NumDistinct" + field_label.replace(" " , "" ))
        plot_num_distinct(df_distinct_other, AF_list, "Number of Distinct " + field_label , "AF Range" , figname, color )   


        color = other_color
        figname =get_fig_name(fig_folder_path, itr, "HeavyHitter" )
        title = fields_cond_str
        if "True" in title: 
            title = RAW_DATA_STR
            field_cond_str = RAW_DATA_STR

        if args.proto == "DNS": 
            bad_values = [255]
            legends = ["Others", "ANY record (highlighted by prior work)"]

        elif args.proto == "NTPPrivate": 
            bad_values = [20, 42]
            legends = ["Others", "MONLIST (highlighted by prior work)"]
        else: 
            bad_values = []
            legends = []
        plot_hh_individual(hh, AF_interval_list,  fields_cond_str , field_label, figname, bad_values, legends , args.proto)

        output_count = False 
        if len(conditions) == 0: 
            output_count = True

        plot_hh(hh, AF_interval_list,  fields_cond_str , field_label, figname, \
             bad_values, legends , fig_folder_path, output_count)
        #compute_percent_reduction(  )

    print("Plot done ")
    
    with open(hh_info_files, "w") as hh_f: 

        for i in hh_info: 
            hh_f.write( "{}\n".format(i) )



    legends = filter_title 
    
    for i in range(len(legends)):
        if "True" in legends[i]: 
            legends[i] = RAW_DATA_STR
    print("Legends " , legends , len(legends))
    print("Filter eff server" , filter_eff_server )
    df_eff_server = convert_to_dataframe(filter_eff_server, AF_interval_list, legends)  
    print(df_eff_server) 
 
    signature_eff_server_f = os.path.join(fig_folder_path, "signature_effective_server_full.txt")

    with open(signature_eff_server_f, "w") as sig_f: 
        sig_f.write("{}".format(df_eff_server) )


    num_column = len(df_eff_server.columns )
    fig_full_path = os.path.join(fig_folder_path,"signature_effective_server_full.pdf" )
    
    plot_group_bar(df_eff_server,  None, \
                   "Number of Distinct Servers" , "AF Range", fig_full_path , num_column)
    
    fig_reduction = os.path.join(fig_folder_path,"signature_effective_server_percent_reduction_full.pdf" )
    compute_percent_retention(df_eff_server,  None, \
                   "% of Remaining\nVulnerable Servers" , "Range of Amplification Factors", fig_reduction , num_column-1, RAW_DATA_STR )

    
    
    return df_eff_server




#Calculate the distinct server that has that 
def plot_query_analysis_yucheng(df, AF_list, AF_interval_list, signature, fig_folder_path) :
    #cond = [] 
    itr = 0
    
    server_color = 'green'
    other_color = 'orange'
    
    #conditions = ["edns in @edns_on", "rdatatype in @rdatatype_any" , "url in @domain_dnssec"]
    


    conditions_list =  signature.conditions    
    #num_elements = signature.num_elements
    field_check = signature.feature_fields
    field_label = signature.feature_label    

    #if type(df[field_check][0]) == int: 
    try: 
        field_complete = sorted([int(i) for i in  df[field_check].unique()] )
    except: 
        field_complete = df[field_check].unique()

        #field_complete = df[field_check].unique()
    print(field_complete)
    #print(conditions )
    #print("num elem ", num_elements)
    print("field check ", field_check)
    print("field label ", field_label) 

    #return 
    #num_elements =  [i for i in range(len(conditions))] # [0, 3, 1]  #  1]# 2]

    
    #Calculate the effect of the server ..  
    filter_eff_server = [[] for x in range(0, len(AF_list) )]
    #filter_eff_rdata =  [[] for x in range(0, len(AF_list) )]
    filter_title = []

    
    #For each conditions ... 
    print(conditions_list)
    for itr, conditions  in enumerate(conditions_list): 
        #cond = combined_conds[itr]
        fields_cond_str = construct_cond_str_v2(conditions)
        print("fieids_cond_str ", fields_cond_str)
        filter_title.append(signature.legends[itr])
        #continue
        num_dist_server_list = [] 
        num_dist_other_list = [] 
        #return

        hh = [] 

        for a in range(0,len(AF_list)): #AF in AF_list: 
            AF = AF_list[a]
            amp_cond = ""
            if a == len(AF_list) - 1 : 
                amp_cond = "amp_fac > @AF "
            else: 
                AF_next = AF_list[a+1]
                amp_cond = "amp_fac > @AF " #" and amp_fac <= @AF_next "

            cond_str  = "{} and  {} ".format(amp_cond, fields_cond_str) 
            print("COND STR ", cond_str)

            query=df.query( cond_str )
            query.sort_values(by=['amp_fac'], ascending=False,inplace=True)
            print("done querying ")

            num_raw_queries = query.shape[0] 
            #Figure otu num 
            num_server =  compute_num_distinct_server(query) 
            # num_field_check =  compute_num_distinct_fields(query,  field_check) #compute_num_distinct_datatypes(query) 

            # num_dist_server_list.append(num_server )
            # num_dist_other_list.append(num_field_check )

            hh_distinct_servers =  compute_heavy_hitter_num_distinct_server(field_complete, query, field_check)     
            hh_distinct_servers.columns = ["NumVulnerableServers"]
            hh.append(hh_distinct_servers)
            
            
            filter_eff_server[a].append( num_server )    


        # color = server_color 
        # df_distinct_s = pd.DataFrame(num_dist_server_list)
        # df_distinct_s.columns = ['NumDistinctServer' ]
        # df_distinct_s.index = AF_interval_list 
        # figname =get_fig_name(fig_folder_path, itr, "NumDistinctServer" )
        # plot_num_distinct(df_distinct_s, AF_list, "Number of Distinct Servers" , "AF Range" , figname, color)   

        # color = other_color
        # df_distinct_other = pd.DataFrame(num_dist_other_list)
        # df_distinct_other.columns = ['Num Distinct ' + field_label.replace(" " , "" ) ]
        # df_distinct_other.index = AF_interval_list 
        # figname =get_fig_name(fig_folder_path,  itr, "NumDistinct" + field_label.replace(" " , "" ))
        # plot_num_distinct(df_distinct_other, AF_list, "Number of Distinct " + field_label , "AF Range" , figname, color )   


        color = other_color
        figname =get_fig_name(fig_folder_path, itr, "HeavyHitter" )
        title = fields_cond_str
        if "True" in title: 
            title = RAW_DATA_STR
            field_cond_str = RAW_DATA_STR

        if args.proto == "DNS": 
            bad_values = [255]
            legends = ["Others", "ANY record (highlighted by prior work)"]
        elif args.proto == "NTPPrivate": 
            bad_values = [20, 42]
            legends = ["Others", "MONLIST (highlighted by prior work)"]
        else: 
            bad_values = []
            legends = []
        plot_hh_individual(hh, AF_interval_list,  fields_cond_str , field_label, figname, bad_values, legends, args.proto)

        plot_hh(hh, AF_interval_list,  fields_cond_str , field_label, figname, bad_values, legends )
        #compute_percent_reduction(  )

    print("Plot done ")
    
    legends = filter_title 
    
    for i in range(len(legends)):
        if "True" in legends[i]: 
            legends[i] = RAW_DATA_STR
    print("Legends " , legends , len(legends))
    print("Filter eff server" , filter_eff_server )
    df_eff_server = convert_to_dataframe(filter_eff_server, AF_interval_list, legends)  
    print(df_eff_server) 
    #raw_input()
    #fig_full_path = os.path.join(fig_folder_path,"signature_effective_server_partial.png" )
    # num_column = 2
    # plot_group_bar(df_eff_server,  "Effectiveness of Each Signature", \
    #                "Number of Distinct Servers" , "AF Range", fig_full_path , num_column)
    
    # fig_reduction = os.path.join(fig_folder_path,"signature_effective_server_percent_reduction_partial.png" )
    # compute_percent_retention(df_eff_server,  "Effectiveness of Each Filter", \
    #                "% of Reduction in NumServers" , "PLAF Range", fig_full_path , num_column-1, RAW_DATA_STR )

    num_column = len(df_eff_server.columns )
    fig_full_path = os.path.join(fig_folder_path,"signature_effective_server_full.pdf" )
    
    plot_group_bar(df_eff_server,  None, \
                   "Number of Distinct Servers" , "AF Range", fig_full_path , num_column)
    
    fig_reduction = os.path.join(fig_folder_path,"signature_effective_server_percent_reduction_full.pdf" )
    compute_percent_retention(df_eff_server,  None, \
                   "% of Remaining\nvulnerable Servers" , "AF Range", fig_reduction , num_column-1, RAW_DATA_STR )

    
    
    return df_eff_server






#Calculate the distinct server that has that 
def plot_query_analysis(df, AF_list, AF_interval_list, signature, fig_folder_path) :
    #cond = [] 
    itr = 0
    
    server_color = 'green'
    other_color = 'orange'
    
    #conditions = ["edns in @edns_on", "rdatatype in @rdatatype_any" , "url in @domain_dnssec"]
    


    conditions = signature.conditions    
    num_elements = signature.num_elements
    field_check = signature.feature_fields
    field_label = signature.feature_label    

    field_complete = sorted([int(i) for i in  df[field_check].unique()] )

    print(conditions )
    print("num elem ", num_elements)
    print("field check ", field_check)
    print("field label ", field_label) 

    #return 
    #num_elements =  [i for i in range(len(conditions))] # [0, 3, 1]  #  1]# 2]

    
    #Calculate the effect of the server ..  
    filter_eff_server = [[] for x in range(0, len(AF_list) )]
    #filter_eff_rdata =  [[] for x in range(0, len(AF_list) )]
    filter_title = []

    
    for n in num_elements: 
        combined_conds = generate_combinations( conditions, n )


        for itr in range(0, len(combined_conds)): 
            cond = combined_conds[itr]
            fields_cond_str = construct_cond_str( cond , conditions)
            print("fieids_cond_str ", fields_cond_str)
            filter_title.append(signature.legends[itr])
            #continue
            num_dist_server_list = [] 
            num_dist_other_list = [] 

            hh = [] 

            for a in range(0,len(AF_list)): #AF in AF_list: 
                AF = AF_list[a]
                amp_cond = ""
                if a == len(AF_list) - 1 : 
                    amp_cond = "amp_fac > @AF "
                else: 
                    AF_next = AF_list[a+1]
                    amp_cond = "amp_fac > @AF and amp_fac <= @AF_next "

                cond_str  = "{} and  {} ".format(amp_cond, fields_cond_str) 
                print("COND STR ", cond_str)

                query=df.query( cond_str )
                query.sort_values(by=['amp_fac'], ascending=False,inplace=True)
                print("done querying ")

                num_raw_queries = query.shape[0] 
                #Figure otu num 
                num_server =  compute_num_distinct_server(query) 
                num_field_check =  compute_num_distinct_fields(query,  field_check) #compute_num_distinct_datatypes(query) 

                num_dist_server_list.append(num_server )
                num_dist_other_list.append(num_field_check )

                #num_distinct.append([num_distinct_datatype, num_distinct_server ])
                #hh_raw_queries =  compute_heavy_hitter_raw_queries(rdatatype_complete, query, field_check) 
                hh_distinct_servers =  compute_heavy_hitter_num_distinct_server(field_complete, query, field_check)     
                #result = pd.concat([hh_distinct_servers], axis=1, sort=False)
                hh_distinct_servers.columns = ["NumVulnerableServers"]
                hh.append(hh_distinct_servers)
                
                
                filter_eff_server[a].append( num_server )    
    

            color = server_color 
            df_distinct_s = pd.DataFrame(num_dist_server_list)
            df_distinct_s.columns = ['NumDistinctServer' ]
            df_distinct_s.index = AF_interval_list 
            figname =get_fig_name(fig_folder_path, len(cond), itr, "NumDistinctServer" )
            plot_num_distinct(df_distinct_s, AF_list, "Number of Distinct Servers" , "AF Range" , figname, color)   

            color = other_color
            df_distinct_other = pd.DataFrame(num_dist_other_list)
            df_distinct_other.columns = ['Num Distinct ' + field_label.replace(" " , "" ) ]
            df_distinct_other.index = AF_interval_list 
            figname =get_fig_name(fig_folder_path, len(cond), itr, "NumDistinct" + field_label.replace(" " , "" ))
            plot_num_distinct(df_distinct_other, AF_list, "Number of Distinct " + field_label , "AF Range" , figname, color )   


            color = other_color
            figname =get_fig_name(fig_folder_path, len(cond), itr, "HeavyHitter" )
            title = fields_cond_str
            if "True" in title: 
                title = RAW_DATA_STR
            plot_hh(hh, AF_interval_list,  fields_cond_str , field_label, figname, color )
            #compute_percent_reduction(  )

    print("Plot done ")
    
    legends = filter_title 
    
    for i in range(len(legends)):
        if "True" in legends[i]: 
            legends[i] = RAW_DATA_STR

    df_eff_server = convert_to_dataframe(filter_eff_server, AF_interval_list, legends)  
    print(df_eff_server) 
    
    fig_full_path = os.path.join(fig_folder_path,"signature_effective_server_partial.pdf" )
    num_column = 2
    plot_group_bar(df_eff_server,  "Effectiveness of Each Signature", \
                   "Number of Distinct Servers" , "AF Range", fig_full_path , num_column)
    
    fig_reduction = os.path.join(fig_folder_path,"signature_effective_server_percent_reduction_partial.pdf" )
    compute_percent_reduction(df_eff_server,  "Effectiveness of Each Signature", \
                   "% of Reduction in NumServers" , "AF Range", fig_full_path , num_column-1, RAW_DATA_STR )

    num_column = len(df_eff_server.columns )
    fig_full_path = os.path.join(fig_folder_path,"signature_effective_server_full.pdf" )
    
    plot_group_bar(df_eff_server,  "Effectiveness of Each Signature", \
                   "Number of Distinct Servers" , "AF Range", fig_full_path , num_column)
    
    fig_reduction = os.path.join(fig_folder_path,"signature_effective_server_percent_reduction_full.pdf" )
    compute_percent_reduction(df_eff_server,  "Effectiveness of Each Signature", \
                   "% of Reduction in NumServers" , "AF Range", fig_full_path , num_column-1, RAW_DATA_STR )

    
    
    return df_eff_server


'''
Plot the CDF of the series 
''' 

def process_cdf(data): 
    #print(len(df))
    maxAF = pd.Series(data.groupby(["server_id"])["amp_fac"].max() )

    return pd.Series(maxAF)

            
def plot_cdf(lst, proto , figname):  
    total_size = len(lst)
    print("total size ", total_size)
    #print("total size is ", lst/10)
    
    fig = plt.figure()
    print(" hello ")
    #lst.hist( cumulative = True, density=True,  bins=int(total_size/10) ) #, bins=1000)
    lst.hist( cumulative = True, normed=True, bins=int(total_size/10) ) #, bins=1000)

    #fig.suptitle(proto, fontsize=20)
    # print("title doen ")
    plt.xlabel('Maximum AF for each server', fontsize=22)
    plt.ylabel('Fraction of Servers', fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title( "", fontsize=1)
    #plt.show()

    plt.savefig(figname, bbox_inches='tight')
    #plt.show()
    return 
    
def get_cdf_fig_name( path, proto ):
    return os.path.join( path, 'CDF_{}.pdf'.format(proto) )
    


def fetch_signature(proto):

    if "dns" in proto.lower():    #"DNS": 
        return DNSSig() 
    elif "ntpprivate" in proto.lower() :  #== "NTPPrivate": 
        return NTPPrivateSig() 
    elif proto.lower() == "memcached": #"Chargen"
        return MemcachedSig()
    elif proto.lower() == "ssdp": #"Chargen"
        return SSDPSig()
    else: 
        raise ValueError("Proto " , proto, "is not supported")

#def get_intermediate_file_name( args)

    





def construct_aggregate_summary( args, protocol_lst, proto_folder_list ): 

    #d = {'AF': ['10-20','20-30','30-40','40-50','50-100','>=100' ], 'DNS': [195,43,35,43,3,0],'Chargen': [4,0,6,25,337,0],'Memcached': [29,76,19,0,0,0],'NTP': [1,2,1,8,3,3],'SSDP': [95,1114,148,13,10,8]}

    d = {}
    #d["Protocol"] = ["DNS", "NTPPrivate", "SSDP", "Chargen", "Memcached"]
    #protocol_lst = ["DNS", "NTPPrivate", "SSDP", "Chargen", "Memcached"]
    #protocol_lst= protocol_lst.reverse()
    
    protocol_lst.reverse()
    #proto_folder_list =  ["dns_10k_correct", "ntpprivate_10k_correct", "ssdp_10k_correct", "chargen_10k_correct", "memcached_10k_correct"]

    #proto_folder_list =  ["dns_10k_may29", "ntpprivate_10k_correct", "ssdp_10k_correct", "chargen_10k_correct", "memcached_10k_correct"]


    proto_folder_list.reverse()
    af_ranges = [(None,10), (10,30), (30,50), (50,100), (100,None)]
    af_legends = ["<10 (low risk)", "[10,30)", "[30,50)", "[50,100)", ">=100"]
    #d["AF"] = ["<10", "10-30", "30-50", "50-100", ">=100"]

    tmp_d = OrderedDict()  
    for index, proto in enumerate(protocol_lst): #proto_list: 
        folder_alias = proto_folder_list[index]
        cdf_df_fname=os.path.join( args.intermediate_data_folder , folder_alias +'-cdf-list-1.csv')
        exists = os.path.isfile(cdf_df_fname)
        if exists:
            print("Skipping reading the parsed file")
            cdf_df=pd.read_csv(cdf_df_fname)
            print("read the csv")
        else:

            # complete_queries_files = os.path.join( args.qps_data_folder, "complete_info.csv") 
            # print("\n\nReading qll queries files " ,complete_queries_files)

            # if not os.path.isfile( complete_queries_files ):
            #     raise ValueError("complete queries files ", complete_queries_files , " does not exist")

            # df = pd.read_csv(complete_queries_files )


            print("proto ", proto , cdf_df_fname)
            cdf_df=process_cdf(df)
            #cdf_df.to_csv(cdf_df_fname,index=False)
        cdf_df.columns = ["amp_fac"]
        print(cdf_df.head())
        #return 

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
        #print("Sum" , sum(norm))
        #print("Total ", total )
        #print(norm)
    print(d)
    df=pd.DataFrame(data=d).transpose()
    df.columns = af_legends
    return df




def aggregate(args):

    # sns.set(font_scale=1) 

    tips = sns.load_dataset("tips")
    # print(tips.head())
    # highamp=np.array(highamp)
    # lowamp=np.array(lowamp)
    # fulllist=(highamp+lowamp)
    # print(fulllist)
    # percenthigh=100.0*highamp/(highamp+lowamp)
    # percentlow=100.0*lowamp/(highamp+lowamp)

    '''
    COMBINED 
    '''

    # protocol_lst = ["DNS",    "NTP$^{\mathrm{AND}}$","SNMP$^{\mathrm{AND}}$", "SSDP", "Chargen", "Memcached"] 
    # proto_folder_list =  ["DNS_10k_may29", "ntp_10k", "snmp_10k" , \
    #     "SSDP_10k_may30", "chargen_10k_may30", "memcached_10k_may28"]
        

    protocol_lst = ["DNS",   "NTP$^{\mathrm{AND}}$" ,  "SNMP$^{\mathrm{AND}}$" ,\
         "SSDP", "Chargen", "Memcached" ]

    proto_folder_list =  ["dns_10k",  "ntp-and_not10k", "snmp-and_10k", \
         "ssdp_10k", "chargen_10k", "memcached_10k"]


    # protocol_lst = ["DNS",   "SSDP", "Chargen", "Memcached", "NTPPrivate$^{AND}$",  "SNMPBulk$^{AND}$", "SNMPNext$^{AND}$", "SNMPGet$^{AND}$"]
    # proto_folder_list =  ["DNS_10k_may29",  "SSDP_10k_may30", "chargen_10k_may30", "memcached_10k_may28", \
    #     "NTPPrivate_10k_may30",   "SNMPBulk_10k", "SNMPNext_10k" , "SNMPGet_10k",]

    # protocol_lst = ["DNS",   "SSDP", "Chargen", "Memcached", "NTPPrivate", "NTPPrivate$^{AND}$" ,\
    #      "SNMPBulk","SNMPBulk$^{AND}$" ,  \
    #       "SNMPNext","SNMPNext$^{AND}$", \
    #        "SNMPGet", "SNMPGet$^{AND}$" ]
    # proto_folder_list =  ["dns_10k",  "ssdp_10k", "chargen_10k", "memcached_10k",\
    #     "ntpprivate_10k", "ntpprivate-and_not10k"  , \
    #      "snmpbulk_10k","snmpbulk-and_10k", \
    #      "snmpnext_10k","snmpnext-and_10k", \
    #       "snmpget_10k","snmpget-and_10k",]




    # protocol_lst = ["DNS",   "SSDP", "Chargen", "Memcached", "NTPPrivate", \
    #         "SNMPBulk","SNMPNext", "SNMPGet",  \
    #         "NTPPrivate$^{AND}$" ,\
    #         "SNMPBulk$^{AND}$" ,  \
    #          "SNMPNext$^{AND}$", \
    #         "SNMPGet$^{AND}$" ]
    # proto_folder_list =  ["dns_10k",  "ssdp_10k", "chargen_10k", "memcached_10k",\
    #         "ntpprivate_10k", "snmpbulk_10k",  "snmpnext_10k",  "snmpget_10k", \
    #         "ntpprivate-and_not10k"  , "snmpbulk-and_10k", "snmpnext-and_10k", "snmpget-and_10k",]




    df = construct_aggregate_summary(args, protocol_lst, proto_folder_list) 
    #print(d)
    #return 

    #d = {'AF': ['10-20','20-30','30-40','40-50','50-100','>=100' ], 'DNS': [195,43,35,43,3,0],'Chargen': [4,0,6,25,337,0],'Memcached': [29,76,19,0,0,0],'NTP': [1,2,1,8,3,3],'SSDP': [95,1114,148,13,10,8]}
    


    out_file = os.path.join(os.getcwd(), args.out_dir, "aggregate_info_df_yr2020.txt")
    print(df )
    

    df.to_csv(out_file,index=True)
    ()+1

    ax=df.plot.barh(stacked=True,logx=False,fontsize=22,width=0.7, cmap="Set3", figsize=(12,6), \
        edgecolor='black', linewidth=1);

    # ax=df.plot.barh(stacked=True,logx=False,fontsize=22,width=0.7, cmap="Set3", figsize=(12,8), \
    #     edgecolor='black', linewidth=1);
    #ax.set_yticklabels(df.AF)
    ax.set_xlabel("% of Servers",fontsize=28)
    ax.set_ylabel('Protocols',fontsize=28)
    ax.tick_params(axis='y',size=18)



    n_bars = int(df.shape[0])

    # ax = plt.subplot(111,aspect = 'equal')
    hatches = ['dummy', '.', '\\', 'x', '+', '*', '+', 'O', '.']

    h_index = 0 
    # setting the legend 
    print(len(ax.patches))
    for i, patch in enumerate(ax.patches):
        if i % n_bars == 0: 
           h_index += 1 
        hatch = hatches[h_index]
        print(i, h_index)
        patch.set_hatch(hatch)


    ######## does the actual hatching 
    # for i, patch in enumerate(ax.artists):
        
    #     hatch = hatches[i % num_total] 
    #     patch.set_hatch(hatch)





    aggregate_info_f = os.path.join(os.getcwd(), args.out_dir, "aggregate_info.txt")
    with open(aggregate_info_f, 'w') as f: 
        f.write("{}".format(df) ) 



    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plot_margin = 0.3
    plt.grid(True, axis='x')
    x0, x1, y0, y1 = plt.axis()
    plt.axis((x0 - 0,
              x1 + 0,
              y0 - 0,
              y1 + plot_margin))
    # ax.legend(bbox_anc)
    ax.legend(loc='upper center', fontsize=21, columnspacing=0.8,  ncol=5, bbox_to_anchor=(0.35, 1.15))
    #ax.legend(loc='upper center', fontsize=21, columnspacing=0.8,  ncol=5, bbox_to_anchor=(0.35, 1.18))



    



    plt.tight_layout()

    fig_name = os.path.join( args.out_dir, "AF_Distribution.pdf")
    savefig(fig_name,  facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)
    #plt.show()

def sanity_check(args): 
    df_fname=os.path.join( args.intermediate_data_folder ,  args.alias+'-1.csv')
    exists = os.path.isfile(df_fname)
    if exists:
        print("\n\nSkipping reading the parsed file")
        df=pd.read_csv(df_fname)
    else:
        print("\n\nReading the protocol out folder to store ")
        df=read_protodata(args.out_folder_name)
        df.to_csv(df_fname,index=False)
    print("Sample data : " , df.head())
    print("Shape of data : " , df.shape)
    srv_ids=np.unique(df['server_id'].values)
    print("Number of servers : " , len(srv_ids))

    record_types = [2]
    dnssec_off = [0]
    a = df.query("amp_fac >10 and amp_fac <30 and rdatatype in @record_types and dnssec in @dnssec_off")
    print(a.head()) 
    #print(len(a))
    print(len(np.unique(a['server_id'].values) ) )


def ntp_test(args): 
    df_fname=os.path.join( args.intermediate_data_folder ,  args.alias+'-1.csv')
    exists = os.path.isfile(df_fname)
    if exists:
        print("\n\nSkipping reading the parsed file")
        df=pd.read_csv(df_fname)
    else:
        print("\n\nReading the protocol out folder to store ")
        df=read_protodata(args.out_folder_name)
        df.to_csv(df_fname,index=False)
    print("Sample data : " , df.head())
    print("Shape of data : " , df.shape)
    srv_ids=np.unique(df['server_id'].values)
    print("Number of servers : " , len(srv_ids))

   
    # a = df.query("amp_fac >10 and amp_fac <30 and rdatatype in @record_types and dnssec in @dnssec_off")
    # print(a.head()) 
    # #print(len(a))
    # print(len(np.unique(a['server_id'].values) ) )

#For yucheng: filter prior work
def filter_prior_work(df , signature): 

    df_max = pd.DataFrame()

    #Step 0: Get the max for each server 
    server_ids=np.unique(df.server_id.values)
    for server_id in server_ids:
        newdf= df.query("server_id in @server_id")
        newdf.reset_index(drop=True,inplace=True)

        AFs=newdf.amp_fac.values
        index=np.argmax(AFs)
        query_max=newdf.iloc[[index]]




    conditions_list = signature.conditions    
    field_check = signature.feature_fields
    field_label = signature.feature_label    


    #For each conditions ... 
    print(conditions_list)
    for itr, conditions  in enumerate(conditions_list): 
        #cond = combined_conds[itr]

        if itr != 1 : 
            continue

        fields_cond_str = construct_cond_str_v2(conditions)
        cond_str  = "{} and  {} ".format("amp_fac > 10 ", fields_cond_str) 
        print(cond_str)
        #print("COND STR ", cond_str)

        query_all_filter=df.query( cond_str )
        query_max_filter=query_max.query( cond_str )
        print(query_all_filter.shape)
        print(query_max_filter.shape)
        #query.sort_values(by=['amp_fac'], ascending=False,inplace=True)
        #print("done querying ")

    query_all_filter.to_csv("dns_query_all_filter.csv",index=False)
    query_max_filter.to_csv("dns_query_max_filter.csv",index=False)

    query_all_filter_dict = {}
    # write query_max_filter
    for item in query_all_filter.groupby("server_id"):
        serverid = item[0]
        maxAF = item[1].iloc[0]["amp_fac"]
        query_all_filter_dict[serverid] = maxAF

    with open("dnsallfilter_10k.json", 'w') as f:
        json.dump(query_all_filter_dict, f)


    query_max_filter_dict = {}
    # write query_max_filter
    for item in query_max_filter.groupby("server_id"):
        serverid = item[0]
        maxAF = item[1].iloc[0]["amp_fac"]
        query_max_filter_dict[serverid] = maxAF

    with open("dnsmaxfilter_10k.json", 'w') as f:
        json.dump(query_max_filter_dict, f)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()#help="--fields_path , --data_folder_name  --proto  ")
    parser.add_argument('--fields_path', type=str, default='../new-src/field_inputs_ntpPrivate', help="relative path for the fields ") 
    #parser.add_argument('--out_folder_name', type=str , help='the locations of the out folder that stores the raw data ')#, required=True) 
    parser.add_argument('--proto', type=str, default="NTPPrivate")#, required=True)
    parser.add_argument('--alias' , type=str, default="blah" , help="Alias to save the CSV files ")#, required=True)
    
    parser.add_argument('--qps_data_folder', type=str, help = "qps/out_DNS_10k_may29/ is an example ")

    parser.add_argument('--out_dir', type=str, default="figs",  help = "figs is an example ")
    #parser.add_argument('--cdf', default=False, action='store_true')
    parser.add_argument('--map_filter', default=False, action='store_true')

    parser.add_argument('--signature', default=False, action='store_true')
    parser.add_argument('--intermediate_data_folder', type=str, default="./intermediate_data")
    parser.add_argument('--aggregate_summary',  default=False, action='store_true')
    args = parser.parse_args()
 

    fig_root_dir = args.out_dir
    make_dir(fig_root_dir)

    
    if args.signature == True: 
        '''
        Plot the query analysis 
        '''
        all_queries_file  = os.path.join( args.qps_data_folder, "complete_info.csv") 

        #print("\n\nReading qll queries files " ,complete_queries_files)

        df = pd.read_csv(all_queries_file )

        #df = df.query("amp_fac >= 10")

        print("done reading ")
        AF_list, AF_interval_list = get_AF_interval(df)
        sig = fetch_signature(args.proto)
        #print("AF list is ", AF_list )
        print("AF interval is ", AF_interval_list )  
        #fig_root_dir = os.path.join(os.getcwd(), "figs")
        
        fig_folder_path = os.path.join(fig_root_dir, args.alias)
        make_dir(fig_folder_path)      
        df_eff_server = plot_query_analysis_v2(df,AF_list, AF_interval_list, sig, fig_folder_path)
        #df_eff_server = plot_query_analysis_yucheng(df,AF_list, AF_interval_list, sig, fig_folder_path)

    # elif args.map_filter == True: 
    #     sig = fetch_signature(proto)
    #     filter_prior_work(df, sig  )



