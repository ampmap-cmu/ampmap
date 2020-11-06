
import os, json, ast

import pandas as pd
import numpy as np
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from natsort import natsorted, ns
def is_QP_match(QP1, QP2):
    count = 0
    for i, v1 in QP1.items():
        v2 = QP2[i]
        
        #if v1 == '[-1.0]' or v2 == '[-1.0]':
        #    continue
        if v1 == v2:
            continue
            
        count += 1
    
    if count == 1:
        return True
    else:
        return False
node_counter=0
node_names={}
mapper={}

G = nx.DiGraph()

def matches(l1,l2):
    global node_counter,node_names,G,mapper

    edge = 1
    fnames=0
    for a1,a2 in l1.items():

        o2=l2[a1]


        if o2 !=a2 :
            if a2 == -1:
                fnames=a1+':'+str(o2)
                continue
            else:
                edge=0
                break
            
    if(edge==1):
        node_counter+=1
        par = json.dumps(l1)
        if par in node_names:
            node_num = node_names[par]
        else:
            node_counter+=1
            node_names[par]= str(node_counter)
            node_num=node_names[par]
            
        G.add_node(node_num)
        child =  json.dumps(l2)
        if child in node_names:
            node_num2 = node_names[par]
        else:
            node_counter+=1
            node_names[child]= str(node_counter)
            node_num2=node_names[child]
#         node_num2=fnames   
        G.add_node(node_num2)
        mapper[node_num2]=str(node_num2)+':'+fnames 

#         OG.add_node(json.dumps(l1))
#         OG.add_node(json.dumps(l2))
#         OG.add_edge(json.dumps(l1),json.dumps(l2),label=fnames)

        G.add_edge(node_num,node_num2,label=fnames)



def fig_gen(secure,flag,proto_name):
    global node_counter,node_names,G,mapper

    direc = proto_name+"/"


    sig_filename = direc+'sigs.npz'
    PERCENTILE = 98

    if(flag==0):
        fft = direc+secure+"DAG_QPS_depth_median.npy"
    else:
        fft = direc+secure+"DAG_QPS_depth_max.npy"
        # depth_QPs_dict = np.load(direc+secure+"DAG_QPS_depth_max.npy",allow_pickle=True)
    if not os.path.exists(fft):
        # print("File not found")
        print("File not found")
        return -1

    depth_QPs_dict = np.load(fft,allow_pickle=True)

    QPs_set = depth_QPs_dict.item()
    depths = QPs_set.keys()
    # QPs = QPs_set[DEPTH]
    # QPs = QPs.applymap(str)
    #############################################################

    # load signature files
    sig = np.load(sig_filename,allow_pickle=True)

    print(list(sig.keys()))
    hc = sig['hc'].item()
    signatures=sig['sigs']
    print(hc)

    print(signatures)
    # store QP_match_info
    depth_QP_match = {}
    for d in depths:
        depth_QP_match[d] = {"num_QPs": -1, "matches": []}
    print(QPs_set[0].shape)
    # check whether two QPs are 1-field different

    # ()+1


    # In[ ]:





    # In[86]:


    counter=0
    col_info={}
    for col,tp in hc.items():
        counter+=1
        if tp!=2:
            continue
        curcnt= counter-1
        vals= signatures[curcnt]
        scores=[]
        for k in range(1,len(vals)):
            rr=vals[k]
            curscore=0
            for eentry in rr:
                curscore += len(eentry)
                
    #         print(rr)
            scores.append(curscore)
    #     ()+1
        rank=ss.rankdata(scores)
        values=vals[1:]
        print(rank,values,vals[0],"SC: ",scores)
        sortedsigvals = [x for _,x in sorted(zip(rank,values))]
        col_info[vals[0]]=sortedsigvals
    print("\n",col_info)

    int_list = list(col_info.keys())

    print(int_list)



    depth_QP={}

    if 1==0:
        with open("depth_QP_match.json", 'r') as f:
            depth_QP_match = json.load(f)
    else:
        for DEPTH in depths:
            QPs = QPs_set[DEPTH]
            QPs = QPs.applymap(str)        
            depth_QP_match[DEPTH]["num_QPs"] = len(QPs)

            # store QP index to QP
            # print(col_info,hc)

            simplifiedQP=[]
            print("DEPTH: ", DEPTH,QPs,QPs.shape)

            for index,currow in QPs.iterrows():
                newrow=currow.copy()
    #             print(newrow)
                for k,v in col_info.items():
                    if '-1' in newrow[k]:
                        continue
                    curvalues=ast.literal_eval(currow[k])
                    newval=[]
                    for eachval in curvalues:
    #                     print("H1:",k,v,curvalues,eachval,type(eachval))
                        eachval = float(eachval)
    #                     print(eachval,type(eachval),"\n")
                        newval.append(str(v[int(eachval)][0]))

                    newrow[k] =newval
                simplifiedQP.append(newrow.values)
            # print(simplifiedQP[0])
            simpDF=pd.DataFrame(simplifiedQP,columns=QPs.columns)
    #         simpDF.to_csv("QPs_depth_"+str(DEPTH)+".csv")
            simpDF
            print("DEPTH: ", DEPTH,simpDF.shape)

            depth_QP[DEPTH]=simpDF

    conditions={}
    print(conditions)
    mod_depth_QP={}
    for s1,s2 in depth_QP.items():
        print("Depth: " ,s1)
        newlist=[]
        for index,row in s2.iterrows():
    #         print(index,row)
            nrg=[]
            for i1,val in row.items():
    #             print(i1,val)
                if i1  not in int_list:
    #                 print(i1,val)
                    nr=ast.literal_eval(val)
                    nrg.append(nr[0])

                else:
    #                 nr=np.array(val)
                    nr = val[0]
    #                 print(nr)
                    if nr =='[':
                        vv=ast.literal_eval(val)
                        nrg.append(vv[0])
                    else:
    #                 print(nr,nr.dtype.itemsize)
    #                 print(nr.shape[0])
    #                 if nr.shap
                        nrg.append(nr)
    #         print(nrg)
    #         ()+1
            skip = 0
            for c1,c2 in conditions.items():
    #             print(c1,c2)
                curval=ast.literal_eval(row[c1])

                curval = np.array(curval).astype(float)
    #             curval = row[c1][0]
    #             print(curval)
    #             ()+1
                if curval[0] in c2:
                    continue
                else:
                    skip =1
                    break
            if(skip ==0):
                newlist.append(nrg)
    #             print(curval)
    #             if 
    #             if 
    #     print(newlist)
        simpDF=pd.DataFrame(newlist,columns=depth_QP[0].columns.values)
    #     simpDF.to_csv("QPs_depth_"+str(DEPTH)+".csv")
    #     simpDF
        print(simpDF)
        mod_depth_QP[s1]=simpDF

    from networkx.drawing.nx_agraph import write_dot, graphviz_layout

    # from IPython.display import Image, display


    from networkx.drawing.nx_agraph import graphviz_layout, to_agraph

    kk =mod_depth_QP.keys()
    sorted_depth = sorted(kk,reverse=True)
    print(sorted_depth)
    OG=nx.DiGraph()

    G = nx.DiGraph()
    node_counter=0
    edgeinfo={}
    mapper={}


    for eachdepth in sorted_depth:
        print(eachdepth)
        if(eachdepth<2):
            continue
        curdepth_QPS=mod_depth_QP[eachdepth].to_dict(orient='records')

        newdepth_QPS = mod_depth_QP[eachdepth-1].to_dict(orient='records')
    #     print(cu)
        if len(newdepth_QPS) == 0 or len(curdepth_QPS)==0:
            continue
    #     if 
    #     print("CUR : \n",curdepth_QPS)
    #     print("NEW: \n",newdepth_QPS)
        for l1 in curdepth_QPS:
            for l2 in newdepth_QPS:
                mtch=matches(l1,l2)
            


    # A = to_agraph(G)
    print(os.getcwd())
    # A.graph_attr.update(size="10,10")

    # A.layout(prog='dot')
    # A.draw('abcd2.png')

    to_pdot = nx.drawing.nx_pydot.to_pydot

    pdot = to_pdot(G)
    print('done')
    # view_pydot(pdot)



    from networkx.readwrite import json_graph

    le= [e for e in G.edges()]
    for u,v in le:
    #     print(u,v)
        if u == v:
            G.remove_edge(u,v)
    le= [e for e in G.nodes()]
    # print(le)
    print(G.number_of_nodes(), G.number_of_edges())
    for ev in le:
    #     paths= = list(G.ancestors(ev))
        paths=list(nx.all_simple_paths(G, source='2', target=ev))
        if (ev =='2'):
            continue
    #     print(ev,paths,len(paths))
        if(len(paths)==0):
            G.remove_node(ev)
            mapper.pop(ev, None)


    depth=nx.shortest_path_length(G,'2')
    # print(depth)
    leaf_nodes = [node for node in G.nodes() if G.in_degree(node)!=0 and G.out_degree(node)==0]

    leaf_depth = max(depth.values())
    print(leaf_depth,leaf_nodes)
    for ev in le:
        edgs=G.edges(ev)
    #     print(ev,edgs)
        nn=[x[1] for x in edgs]
        if_interesting=1
        for eachnn in nn:
            if eachnn not in leaf_nodes:
                if_interesting=0
                break

        if(if_interesting):
    #         all_leafs = G.nodes[nn]
            res=[mapper[k] for k in nn]
            if(len(res)<=1):
                continue
            data=[i.split(':') for i in res]
    #         data=res.split(":")
            data=np.array(data)
            unq_fields = np.unique(data[:,1])
            print(ev,edgs,nn,res)
            print(data,unq_fields,data.shape)   
            nodename=''
            nnd = data[int(data.shape[0]/2),0]
            print(nnd)
            for each in unq_fields:
    #             for 
                valid_ids = np.where(data[:,1] == each)
                allvals=natsorted(np.unique(data[valid_ids,2]))
    #             allvals = [ int(x) for x in allvals ]

                movies = ','.join(allvals)
                node_id =str(nnd) + ':'+each + ':' + movies
                print("HERE",node_id)
                child = node_id
                node_counter+=1
                node_names[child]= str(node_counter)
                node_num2=node_names[child]
                G.add_node(node_num2)
                mapper[node_num2]=node_id
                G.add_edge(ev,node_num2,label=node_id)
                
            for each_child in nn:
                G.remove_node(each_child)
                mapper.pop(each_child, None)


    mapper['2']='root'

    print("HERE" ,G,len(mapper),mapper)
    print(G.number_of_nodes(), G.number_of_edges())

    pdot = to_pdot(G)
    print('done1',direc,secure,)
    # view_pydot(pdot)


    if flag==0:
        
        pdot.write_png(direc+secure+'QPConn_median.png')
        g_fname = direc+secure+'median.json'
    else:
        pdot.write_png(direc+secure+'QPConn_max.png')
        g_fname = direc+secure+'max.json'

    G2=G
    G2=nx.relabel_nodes(G,mapper,copy=True)
    # g_tree = nx.DiGraph(nx.bfs_edges(G, '2'))
    # g_tree=nx.bfs_tree(G,'2',depth_limit=50)
    # nx.draw(g_tree,labels=True)
    pdot = to_pdot(G2)
    print('done2')
    # view_pydot(pdot)

        
    plt.show()
    print('done')
    print(G2.number_of_nodes(), G2.number_of_edges())

    data = json_graph.tree_data(G2,root='root')

    print(data)
    ss2=json.dumps(data)
    print(ss2)
    ss2 = ss2.replace("\"id\"", "\"name\"");
    ss2 = ss2.replace(".0","");
    ss2 = ss2.replace("range","");


    print(ss2)
    pp = json.loads(ss2)
    # print(pp)


    with open(g_fname, 'w') as f:
        json.dump(pp, f, ensure_ascii=False)
    # G = nx.DiGraph([(1,2)])
    # data = json_graph.tree_data(G,root=1)
    print(pp)


 

