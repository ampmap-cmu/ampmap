
import pandas as pd 
import numpy as np
# import os
# from collections import OrderedDict
# import libs 

# import libs.definition as lib_df 

def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def getKL(proto_fields,lib_df,field_to_server_freq,proto_name,KL_thresh):


    klinfo=[]
    IGNORE_LIST=[]
    for field, f_metadata  in proto_fields.items() :  #For all fields .. 
        if f_metadata.size >= lib_df.SMALL_FIELD_THRESHOLD:
            continue        
        server_to_freq_vectors = field_to_server_freq[field]
        print("For field ", field )
        # field_to_freq_vector[field] = OrderedDict()     
        start=-1
        pre=-1
        for s in server_to_freq_vectors:     
            f_v = server_to_freq_vectors[s]
            if(start==-1):
                pre=f_v
                start=1
            else:
                pre=np.add(pre,f_v) 
        pre=np.array(pre)

        print(pre,pre.shape)
        FRQ= pre / np.sum(pre)
        print(FRQ)
         
        if pre.size ==1 and pre == -1:
            IGNORE_LIST.append(field)
        else:
            unfrm = np.ones((pre.shape)) / pre.shape[0]
            curKL=KL(FRQ, unfrm)


            k2={}
            k2['name'] = field
            k2['KL'] = curKL
            k2['size'] = pre.shape[0]
            klinfo.append(k2)              
            if (curKL <KL_thresh and pre.shape[0] > 1):
                IGNORE_LIST.append(field)
            print("KL",field,curKL)
    # print("IGNORED",IGNORE_LIST)
    kl_fname=proto_name+'/klinfo.csv'
    d2d = pd.DataFrame(klinfo)
    # print(d2d)
    d2d.to_csv(kl_fname,index=False)

    return IGNORE_LIST

