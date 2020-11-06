import json, os, sys, time, collections


def write_to_json(queryBuffer, out_file):
    # first read the existing queries
    try: 
        f = open(out_file, 'r')
        queryBuffer_org = json.load(f)
        queryBuffer = queryBuffer_org + queryBuffer
        f.close()
    except FileNotFoundError:
        print("File not exists!")

    # write quries to file
    f = open(out_file, 'w')
    print("dumping starts...")
    try:
        json.dump(queryBuffer, f)
    except TypeError:
        print("Unable to serialize the object")
    print("dumping ends...")
    f.close()

def read_json(f):
    with open(f) as json_file: 
        data = json.load(json_file)
    return data


def print_ampmap(ampmap, base_out_dir, opt_name=None):
    for server_id, pq in ampmap.items():
        if opt_name: 
            out_file = os.path.join( base_out_dir , str(server_id) + ".out" + opt_name )

        else: 
            out_file = os.path.join( base_out_dir , str(server_id) + ".out" )

        count = 0  

        start = time.time()  
        with open(out_file, "w+") as f:
            is_header_written = False

            while not pq.empty():
                item = pq.get()
                
                field_dict = dict(item[2]) 
                field_dict = collections.OrderedDict(sorted(field_dict.items()))

                # write header
                if is_header_written == False:
                    f.write("amp_factor,server_ip")
                    for key, value in field_dict.items():
                        f.write(","+str(key))
                    f.write("\n")
                    is_header_written = True
            
                f.write("{0:.4f}".format(-1*item[0]) + ",{}".format(item[1]))

                for key, value in field_dict.items():
                    f.write(","+str(value))

                f.write("\n")

                count = count + 1 
                if count % 5000 == 0:
                    end = time.time() 
                    start = time.time()
            
