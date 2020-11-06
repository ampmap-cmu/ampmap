import configparser
import os, re, random
import libs.definition as df 

from collections import OrderedDict
import math
import libs.fields as realfields
import numpy as np

delimiter = "--"


def assign_server_to_modetype(size, num_modetype): 

    servers = OrderedDict() 
    #for each server size: 
    for i in range(size):
        mtype = random.randint(1, num_modetype)
        mtypeid = "ModeType" + str(mtype) 
        servers[i] = mtypeid
    return servers 


def pick_num_random_values( lst_of_vals, num_pick ): 
    return random.sample(lst_of_vals, num_pick )


def load_fields_sim(inputs_path):
    fields_data = __read_fields_ini(inputs_path)
    num_field = int(fields_data["fields"]["numFields"])
    print("num fields ", num_field)

    
    list_of_fields = OrderedDict() 
    for i in range(num_field):
        field = fields_data["Field_" + str(i+1)]

        name = field["name"]
        size = int(field["size"])
        is_critical = field["is_critical"]
        

        field_type = field["type"]
        is_int = False 
        
        if field_type == "INT" or field_type == "int":
            is_int = True
        else: 
            is_int = False

        val = None 
        if is_critical == 'True':
            if "num_good_val" in field: 
                num_good_val = int(field["num_good_val"])
                #Now select good values 
                vals = pick_num_random_values( range(0,size) ,num_good_val)                

            elif "good_val_range" in field: 

                high_ranges = field.get("good_val_range").split(",")
                high_range_list = [] 
                for ranges in high_ranges:
                    high_range_lower = int(ranges.split(delimiter)[0] )
                    high_range_upper = int(ranges.split(delimiter)[1] )
                    high_range_list.append( (high_range_lower, high_range_upper) ) 
                vals = high_range_list 
            else:
                raise ValueError("Num good val or good val range has to be given for CF")
 
            f = simfields.critical_field(name, size, range(0,size), vals, is_int )  
        elif is_critical =="False": 
            f = simfields.non_critical_field(name,size, range(0,size), is_int) 

        list_of_fields[name] = f 
 
    return list_of_fields 


def load_servers_v2(inputs_path): 
    server_ini = __read_server_ini(inputs_path)
    size = int(server_ini.get("size"))

    #Read structure 
    structure_flag = int(server_ini.get("structure") )
    #If yes , store structure 

    structure = None 
    if structure_flag ==1: 
        #high mid   
        high_range_lower = int(server_ini.get("high_range").split(delimiter)[0] )
        high_range_upper = int(server_ini.get("high_range").split(delimiter)[1] )
        mid_range_lower = int(server_ini.get("mid_range").split(delimiter)[0] )
        mid_range_upper = int(server_ini.get("mid_range").split(delimiter)[1] )
        structure = df.structure( [high_range_lower, high_range_upper], [mid_range_lower, mid_range_upper] ) 
    else: 
        num_high_val = int(server_ini.get("high_val"))
        num_mid_val = int(server_ini.get("mid_val") )
        structure = df.structureless(num_high_val, num_high_val) 


    return server.server(size, structure_flag, structure)


def __read_sim_server_ini( inputs_path ):
    path = os.path.join(inputs_path, "server.ini")
    print(path)
    server_data = configparser.ConfigParser()
    server_data.optionxform = str
    server_data.read(path)
    return server_data

def __read_server_ini( inputs_path ):
    path = os.path.join(inputs_path, "server.ini")
    print(path)
    server_data = configparser.ConfigParser()
    server_data.optionxform = str
    server_data.read(path)

    return server_data["servers"]

def __read_fields_ini(inputs_path):
    path = os.path.join(inputs_path, "fields.ini")
    # print(path)
    fields_data = configparser.ConfigParser()
    fields_data.optionxform = str
    fields_data.read(path)
    return fields_data

def __read_servers_ini(inputs_path):
    path = os.path.join(inputs_path, "servers.ini")
    # print(path)
    fields_data = configparser.ConfigParser()
    fields_data.optionxform = str
    fields_data.read(path)
    return fields_data

def __read_measurers_ini(inputs_path):
    path = os.path.join(inputs_path, "measurers.ini")
    fields_data = configparser.ConfigParser()
    fields_data.optionxform = str
    fields_data.read(path)
    return fields_data


def load_proto_servers(inputs_path):
    servers_data = __read_servers_ini(inputs_path)
    num_servers = int(servers_data["servers"]["numServers"])
    dnsServers = []
    for i in range(num_servers):
        server = servers_data["Server_" + str(i+1)]
        ip = server["ip"]
        server_rand_sample_count = 0
        dnsServers.append((ip, server_rand_sample_count))
    return dnsServers


def load_fields_real(inputs_path, url=None): 
    fields_data = __read_fields_ini(inputs_path)
    print("Inside load_fields_real : ", inputs_path)
    print("Inside load_fields_real : ", fields_data)
    num_fields = int(fields_data["fields"]["numFields"])
    Fields = OrderedDict()
    for i in range(num_fields):
        field = fields_data["Field_" + str(i+1)]
        name = field["name"]
        size = None
        field_type = field["type"]
        is_accepted_val = field["acceptedtrue"]
        is_int = False 
        accepted_range = None
        numbits = int(field["numbits"])
        print("name ", name)
        if is_accepted_val.lower() == 'true':
            if field_type == "INT":  
                accepted_range = [int(x) for x in field["acceptedset"][1:-1].split(',')]
            else: 
                accepted_range = [x.strip() for x in field["acceptedset"][1:-1].split(',')]

        else: #numbits ..
            print("name ", name, is_accepted_val)
            assert(numbits > 0) 
            size = int(math.pow(2.0, numbits)) 
        
        if name == "url" and url is not None: 
            is_int = False 
            accepted_range = [url]


        if field_type == "INT":
            is_int = True
        else: 
            is_int = False


        real_field = realfields.field(name, size, accepted_range, is_int)
        print(" field ", name , " --> ", size ," : " , accepted_range)
        real_field.printField()
        Fields[name] = real_field

    return Fields

def generate_proto_fields( inputs_path, url=None):
    #OrderedDict{field_name->field_object} consisting of all nmap query parameters for dns.
    proto_fields = load_fields_real(inputs_path, url)
    return proto_fields


def load_nodes(inputs_path): 
    measurers_data = __read_measurers_ini(inputs_path)
    num_measurers = int(measurers_data["measurers"]["numMeasurers"])
    script_directory = measurers_data["measurers"]["directory"]
    script = measurers_data["measurers"]["file"]
    measurerServers = []
    for i in range(num_measurers):
        server = measurers_data["Measurer_" + str(i+1)]
        ip = server["ip"]
        print("ip is ", ip)
        measurerServers.append((ip))

    return measurerServers


def load_critical_fields(inputs_path): 
    fields_data = __read_fields_ini(inputs_path)

    num_fields = int(fields_data["fields"]["numFields"])
    print("num criticla fields ", num_fields)

    CF = OrderedDict()
    for i in range(num_fields):

        critical_fields = fields_data["Field_" + str(i+1)]
        name = critical_fields["name"]
        size = int(critical_fields["size"])
        is_random = int(critical_fields["is_random"])
        is_toggle = int(critical_fields["is_toggle"])
        is_step =   int(critical_fields["is_step"])

        high = int(critical_fields.get("high_mid").split(';')[0])
        mid = int(critical_fields.get("high_mid").split(';')[1])
        print("high ", high , " mid ", mid)
        sparsity_matrix = [high, mid]

        cf = fields.criticalfield(name, size, is_random, is_step, is_toggle,  sparsity_matrix )
        CF[name] = cf 
        print("critical fields ", cf)     
    return CF


def load_non_critical_fields(inputs_path): 
    fields_data = __read_fields_ini(inputs_path)
    #numberCriticalFields
    num_non_critical_fields = int(fields_data["fields"]["numberNonCriticalFields"])
    print("num non criticla fields ", num_non_critical_fields)

    NCF = OrderedDict() 

    for i in range(num_non_critical_fields):
        non_critical_fields = fields_data["NonCriticalFields_" + str(i+1)]
        name = non_critical_fields["name"]
        numbit  = int(non_critical_fields["numbits"])
        size = int(math.pow(2.0, numbit) -1)  #log_list_gen(16)

        ncf = fields.noncriticalfield(name, size )
        NCF[name] = ncf 
        print("non critical fields ", non_critical_fields)     
    return NCF


def load_servers(inputs_path): 
    server_ini = __read_server_ini(inputs_path)
    size = int(server_ini.get("number"))

    server_range = range(0,size)
    num_high = int(server_ini.get("high_mid_low").split(';')[0])
    num_mid = int(server_ini.get("high_mid_low").split(';')[1])
    num_low = int(server_ini.get("high_mid_low").split(';')[2])
    print("num high", num_high)
    print("num mid", num_mid)

    if num_low + num_high + num_mid != size: 
        raise ValueError('Server.ini : num low and num high and mid  shoud add up to size ')

    random_sampled = random.sample(server_range, size)
    high = OrderedDict.fromkeys(random_sampled[:num_high], df.SERVER_HIGH ) 
    mid  = OrderedDict.fromkeys(random_sampled[num_high: num_high + num_mid], df.SERVER_MID )
    low  = OrderedDict.fromkeys(random_sampled[num_high + num_mid:], df.SERVER_LOW )

    high.update(mid)
    high.update(low)

    high_mid_low = OrderedDict(sorted(high.items()))


    print("servers ", high_mid_low)
    return server.server(size, server_range, high_mid_low)


#updates the list of table column names
def load_db_table(inputs_path, filename, columns):
    path = os.path.join(inputs_path, filename)
    data = configparser.ConfigParser()
    data.optionxform = str
    data.read(path)
    num_fields = int(data["fields"]["numFields"])
    for i in range(num_fields):
        field = data["Field_" + str(i+1)]
        columns.append(field["name"])
    if "additionalFile" in data.keys():
        load_db_table(inputs_path, data["additionalFile"]["filename"], columns)
        

#builds a dict of table name to list of column names
def load_db_data(inputs_path):
    path = os.path.join(inputs_path, "db_tables.ini")
    data = configparser.ConfigParser()
    data.optionxform = str
    data.read(path)
    tables = OrderedDict()
    num_fields = int(data["fields"]["numFields"])
    for i in range(num_fields):
        field = data["Field_" + str(i+1)]
        tables[field["name"]] = []
        table_filename = "db_" + field["name"] + ".ini"
        load_db_table(inputs_path, table_filename, tables[field["name"]])
    return tables


