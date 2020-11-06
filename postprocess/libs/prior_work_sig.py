'''
Stores signatures  
'''


'''
DNS-related 
'''
edns_on = ['0']
rdatatype_any=['255']
rdatatype_any_txt =['255','16']
#48 DNSSEY  60 (CDNSKEY ), DS + CDS (43 59)    NSEC NSEC3 NSEC3PARAM (47, 50, 51) RRSIG (46)
rdatatype_dnssec=['255', '48', '60', '43', '59', '47', '50','51','46' ]
domain_dnssec = ['berkeley.edu', 'energy.gov', 'aetna.com', 'Nairaland.com']
domain_no_dnssec = ['chase.com', 'google.com', 'Alibaba.com', 'Cambridge.org', 'Alarabiya.net', 'Bnamericas.com']
dnssec_on = ['1']



class DNSSig(): 
	#num_elements = [0,3,1]
	feature_fields = "rdatatype"
	feature_label = "Record Types"

	conditions = []
	conditions.append([])
	conditions.append([ "edns in @edns_on", "rdatatype in @rdatatype_any_txt"  ]) 
	conditions.append([ "edns in @edns_on" ]) 
	conditions.append([ "rdatatype in @rdatatype_any_txt"]) 
	#conditions.append([ "rdatatype in @rdatatype_any", "dnssec in @dnssec_on"]) 

	legends = []
	legends.append("No Filter")
	legends.append("<EDNS, ANY|TXT>" ) 
	legends.append("<EDNS, * >" ) 
	legends.append("<*, ANY|TXT >" ) 
	#legends.append("< * , ANY, DNSSEC-OK>" ) 

class DNSSig_old(): 
	num_elements = [0,3,1]
	feature_fields = "rdatatype"
	feature_label = "Record Types"

	conditions = []
	conditions.append([])
	conditions.append([ "edns in @edns_on", "rdatatype in @rdatatype_any" , "dnssec in @dnssec_on" ]) 
	conditions.append([ "edns in @edns_on", "dnssec in @dnssec_on" ]) 
	conditions.append([ "edns in @edns_on", "rdatatype in @rdatatype_any"]) 
	conditions.append([ "rdatatype in @rdatatype_any", "dnssec in @dnssec_on"]) 

	legends = []
	legends.append("No Filter")
	legends.append("<EDNS, ANY, DNSSEC-OK>" ) 
	legends.append("<EDNS, * , DNSSEC-OK>" ) 
	legends.append("<EDNS, ANY, * >" ) 
	legends.append("< * , ANY, DNSSEC-OK>" ) 


'''
NTP-related 
''' 
code_monlist = [20, 42]
class NTPPrivateSig(): 
	num_elements = [0,1]
	feature_fields = "request_code"
	feature_label = "Request Code"

	conditions = []
	conditions.append([])
	conditions.append(["request_code in @code_monlist"])

	legends = []
	legends.append("No Filter")
	legends.append("<MONLIST>")


'''
SSDP-related 
''' 
discovery = ["ssdp:discover"]
class SSDPSig(): 
	feature_fields = "man"#, ""]
	feature_label = "man"
	conditions = []
	conditions.append([])
	conditions.append(["man in @discovery"] ) 

	legends = []
	legends.append("No Filter")
	legends.append("<Discovery>")

'''
Memcached-related 
''' 
stats = ["stats"]
#empty = [NaN]

class MemcachedSig(): 
	# feature_fields = "key"
	# feature_label = "Key"
	# conditions = []
	# conditions.append([])
	# conditions.append(["key.isnull()"] ) 

	# legends = []
	# legends.append("No Filter")
	# legends.append("<Stats NTH>")



	feature_fields = "command"
	feature_label = "Command"
	conditions = []
	conditions.append([])
	conditions.append(["command in @stats"] ) 

	legends = []
	legends.append("No Filter")
	legends.append("<Stats>")




# class ChargenSig(): 
# 	feature_fields = ""
# 	feature_label = ""

# 	conditions = []
# 	conditions.append([])
# 	conditions.append([  ])

# 	legends = [] 
# 	legends.append("No Filter")
# 	legends.append("Any request")
