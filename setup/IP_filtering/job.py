import os, sys, configparser
from scapy import *
from scapy.layers.inet import *
from scapy.layers.ntp import *
from scapy.fields import *
from scapy.all import *
import numpy as np
import time
import itertools
import argparse
from copy import deepcopy

# Test whether given IP is mil/gov:
# 0: NO
# 1: YES
def is_gov_or_mil(server_ip):
	command = "dig +noall +answer -x %s" %server_ip
	try: 
	    output = subprocess.check_output(command, stderr=sys.stdout.fileno(), shell=True).strip()

	    status = ""
	    if len(output.decode("utf-8").split()) >= 5:
	    	status = output.decode("utf-8").split()[4]

	    print("Domain name: ", status)

	    if (".gov" not in status) and (".mil" not in status):
	    	print("\tNOT gov or military")
	    	return 0
	    else:
	    	print("\tIt is gov or military")
	    	return 1

	except subprocess.CalledProcessError:
		print("subprocess call error!")
		return -1



# Test for DNS:
# 0: failed
# 1: succeeded
def run_dig_test(server_ip):
	command = "dig @%s berkeley.edu" %server_ip
	try: 
	    output = subprocess.check_output(command, stderr=sys.stdout.fileno(), shell=True).strip()
	    status = output.decode("utf-8").split("status: ")[1].split(',')[0]

	    if status == "NOERROR":
	    	print("Dig test: NOERROR")
	    	return 1
	    else:
	    	print("Dig test: ", status)
	    	return 0

	except subprocess.CalledProcessError:
		print("subprocess call error!")
		return 0



# Test for SSDP:
# 0: failed
# 1: succeeded
def run_ssdp_test(server_ip):
	payload = "M-SEARCH * HTTP/1.1\r\n" + "HOST:"+server_ip+":1900\r\n"  + "ST:upnp:rootdevice\r\n" + "MAN:\"ssdp:discover\"\r\n"  + "MX:1\r\n\r\n"

	print(payload)
	sport = random.randint(5000,65535)
	req = IP(dst=server_ip) / UDP(sport=sport, dport= 1900) / payload
	res, unanswer = sr(req, multi=True, filter = 'port {}'.format(sport), timeout=1,verbose=False)

	if res is not None:
		print(res )
		resplen = sum([len(x[1]) for x in res])
		return float(resplen)/float(len(req))
	else:
		return 0


# Test for Chargen:
# 0: failed
# 1: succeeded
def CHARGEN_query(names, inst, server_ip):
	if names[0] is not "char" or names[1] is not "len":
		print("Error")
		sys.exit(1) 
	payload = inst[0] * inst[1] 
	request = IP(dst=server_ip)/UDP(sport=random.randint(5000,65535), dport=19)/Raw(load=payload)
	res, unans = sr(request, multi=True, timeout=1, filter='port 19', verbose=0)

	if res is not None:
		resplen = sum([len(x[1]) for x in res])
		print("AF: ", float(resplen)/float(len(request)))
		return float(resplen)/float(len(request))
	else:
		print("AF: ", 0)
		print() 
		return 0 

# Test for Memcached:
# 0: failed
# 1: succeeded
def run_memcached_test(serverip): 
	data = "\x00\x00\x00\x00\x00\x01\x00\x00get\r\n"

	request = IP(dst=serverip)/UDP(sport=random.randint(5000,65535),  dport=11211)/Raw(load=data)
	res = sr1(request, timeout=1, verbose=False)
	if res is not None:   
		print("\t\tValid memcache dserver ")
		return 1   
	print("\t\t Not valid memcached ") 
	return 0 


def ntp_query(serverip, names, inst ):  
    packet = NTPHeader()
    if names[0] is not "mode":
        return 0, None
    if inst[0] == 7:
        packet = NTPPrivate()

    for i in range(len(names)):
        setattr(packet, names[i], inst[i])


    p = IP(dst=serverip)/UDP(sport=5823, dport=123)/packet
    response, unans = sr(p, multi=True, timeout=0.7, filter = 'port 123',  verbose=0)
    qlen = len(p)

    resplen = sum([len(x[1]) for x in response])

    ampfac = resplen * 1.0 / qlen
    print("ampfactor ", ampfac)
    return ampfac, packet



def ntp_query_normal(serverip): 
    standard_fields = ['mode', 'leap', 'version', 'stratum', 'poll', 'precision','delay', 'dispersion',   'ref', 'orig', 'recv', 'sent'] 
    vals= [3, 0, 3, 254, 255, 1, 0, 0, 0, 0, 0, 0] 
    amp_factor, packet = ntp_query(serverip, standard_fields, vals  )
    if amp_factor > 0: 
    	return 1 
    return 0 

def ntp_query_private(serverip): 
    private_fields = ['mode', 'response', 'more', 'version',  'auth', 'seq', 'implementation', 'request_code']

    vals = [ 7, 0, 0, 2, 1, 0, 2, 1] 
    amp_factor, pkt = ntp_query(serverip, private_fields, vals)
    if amp_factor > 0: 
    	return 1 
    return 0 


def snmp_bulk(serverip):
    OID_string = '1.3.6.1'
    snmppacket = SNMP(version=1, community='public', PDU=SNMPbulk(id= 200000000, non_repeaters = 1,  \
         max_repetitions=10000, varbindlist=[SNMPvarbind(oid=ASN1_OID(OID_string)), SNMPvarbind(oid=ASN1_OID(OID_string))]))
    p = IP(dst=serverip)/UDP(sport=10000, dport=161)/snmppacket
    response, unans = sr(p, multi=True, timeout=0.7,   verbose=0)
    qlen = len(p)

    resplen = sum([len(x[1]) for x in response])

    ampfac = resplen * 1.0 / qlen
    return ampfac 

def snmp_next(serverip):
    OID_string = '1.3.6.1'
    snmppacket = SNMP(version=1, community='public', PDU=SNMPnext(id= 200,   \
         varbindlist=[SNMPvarbind(oid=ASN1_OID(OID_string))] ) ) 
    p = IP(dst=serverip)/UDP(sport=10000, dport=161)/snmppacket
    response, unans = sr(p, multi=True, timeout=0.7,   verbose=0)
    qlen = len(p)

    resplen = sum([len(x[1]) for x in response])

    ampfac = resplen * 1.0 / qlen
    return ampfac 


def snmp_get(serverip):
    OID_string = '1.3.6.1'
    snmppacket = SNMP(version=1, community='public', PDU=SNMPget(varbindlist=[SNMPvarbind(oid=ASN1_OID(OID_string))] ))
    p = IP(dst=serverip)/UDP(sport=10000, dport=161)/snmppacket
    response, unans = sr(p, multi=True, timeout=0.7,   verbose=0)
    qlen = len(p)

    p.show() 

    print("Response")
    for x in response: 
        x[1].show()

    resplen = sum([len(x[1]) for x in response])
    ampfac = resplen * 1.0 / qlen
    return ampfac 


def read_raw_ips(filename):
	raw_ips_total = []
	f = open(filename, 'r')
	for line in f:
		raw_ips_total.append(line.strip())

	return raw_ips_total

def check_status(cmd_file, status):
	f = open(cmd_file, 'r')
	lines = f.readlines()
	if lines[0].strip() == status:
		return 1
	else:
		return 0


def main():
	proto = sys.argv[1]
	node_ip = sys.argv[2]
	input_filename = sys.argv[3]
	config = configparser.ConfigParser()
	config.read("parameters.ini")

	proto_params_key = proto+"_params"
	filter_split_dir = config[proto_params_key]["filter_split_dir"]

	raw_ips = read_raw_ips(input_filename)

	cmd_dir = config[proto_params_key]["command_dir"]
	cmd_file = os.path.join(cmd_dir, node_ip)

	cnt_dir = config[proto_params_key]["count_dir"]

	count = 0 
	# sanity check
	if check_status(cmd_file, "start") == 1:
		filter_filename = os.path.join(filter_split_dir, node_ip)
		fout = open(filter_filename, 'w')

		cnt_filename = os.path.join(cnt_dir, node_ip)
		fout_cnt = open(cnt_filename, 'w')

		# filter IP based on its protocol
		for raw_ip in raw_ips:
			print("raw ip", raw_ip)
			state = 0
			if proto == "ntp_and":
				if ntp_query_private(raw_ip) == 1 and ntp_query_normal(raw_ip) == 1:
					state += 1
					if (is_gov_or_mil(raw_ip) == 0):
						state += 1
						fout.write("%s\n" %raw_ip)
						count = count + 1 
						fout.flush()

			elif proto == "ntp_or":
				if ntp_query_private(raw_ip) == 1 or ntp_query_normal(raw_ip) == 1:
					state += 1
					if (is_gov_or_mil(raw_ip) == 0):
						state += 1
						fout.write("%s\n" %raw_ip)
						count = count + 1 
						fout.flush() 


			elif proto == "ssdp": 
				if run_ssdp_test(raw_ip) > 0:
					state += 1
					if (is_gov_or_mil(raw_ip) == 0):
						state += 1
						fout.write("%s\n" %raw_ip)
						count = count + 1 
						fout.flush()

			elif proto == "dns":
				if run_dig_test(raw_ip) == 1:
					state += 1
					if (is_gov_or_mil(raw_ip) == 0):
						state += 1
						fout.write("%s\n" %raw_ip)
						count = count + 1 
						fout.flush()

			elif proto == "memcached":
				if run_memcached_test(raw_ip) == 1:
					state += 1
					if (is_gov_or_mil(raw_ip) == 0):
						state += 1
						fout.write("%s\n" %raw_ip)
						count = count + 1 
						fout.flush()

			elif proto == "chargen":
				if CHARGEN_query(["char", "len"], ['a',1], raw_ip) > 0:
					state += 1
					if (is_gov_or_mil(raw_ip) == 0):
						state += 1
						fout.write("%s\n" %raw_ip)
						count = count + 1 
						fout.flush()


			elif proto.lower() == "snmp_and": 
				if snmp_next(raw_ip) > 0 and snmp_bulk(raw_ip) > 0 and snmp_get(raw_ip) > 0:
					state += 1
					if (is_gov_or_mil(raw_ip) == 0):
						state += 1
						fout.write("%s\n" %raw_ip)
						count = count + 1 
						fout.flush() 

			elif proto.lower() == "snmp_or": 
				if snmp_next(raw_ip) > 0 or snmp_bulk(raw_ip) > 0 or snmp_get(raw_ip) > 0:
					state += 1
					if (is_gov_or_mil(raw_ip) == 0):
						state += 1
						fout.write("%s\n" %raw_ip)
						count = count + 1 
						fout.flush() 


			else:
				print("Protocol not supported!")

			print("Cur count is ", count)

			# fout.write("%s\t%d\t%d\n" %(raw_ip, state, count))
			# fout.flush()
		
		fout_cnt.close()
		fout.close()

	else:
		print("command error!")

	# write end to cmd file
	f = open(cmd_file, 'w')
	f.write("end\n")
	f.close()



if __name__ == '__main__':
	main()