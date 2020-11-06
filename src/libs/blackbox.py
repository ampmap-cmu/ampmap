import dns.message
import dns.rdataclass
import dns.rdatatype
import dns.query
import dns.flags
from collections import OrderedDict
from scapy.all import *
from scapy import *
from scapy.layers.inet import *
from scapy.layers.ntp import *
from scapy.fields import *


'''
PCAP_TO_LOCAL_DISK = True: store pcaps to local (non-NFS) dirs, i.e., /ampmap/pcap
PCAP_TO_LOCAL_DISK = False: store pcaps to NFS dirs, i.e., out/pcap

We would suggest storing pcaps to local dirs to ease the burden of NFS read/writes.
'''

PCAP_TO_LOCAL_DISK = True

# send specific query to the server and get responses 
class BlackBox:
    def __init__(self, timeout ):
        self.proto = None
        self.phase = None
        self.query_cnt_dict = {}

        self.timeout = timeout

    # SSDP
    def __ssdp_dict(self, serverip, field_dict):
        print("Hey, you reach ssdp dic ...")
        packets = []
        payload = field_dict["start_line"] + "\r\n" + \
                "HOST:" + field_dict["host"] + "\r\n" + \
                "MAN:\"" + field_dict["man"] + "\"\r\n" + \
                "MX:" + str(field_dict["mx"]) + "\r\n" + \
                "ST:" + field_dict["st"] + "\r\n\r\n" 
        print(payload)

        ssdpRequest = IP(dst=serverip) / UDP(sport=random.randint(5000,65535), dport= 1900) / payload
        res, unans = sr(ssdpRequest, multi=True, timeout=self.timeout)
        packets.append(ssdpRequest)

        # If there is response
        if res is not None:
            resplen = 0 
            for r in res: 
               resplen = resplen + len(r[1]) 
            print("server_ip: %s, AF: %f\n" %(serverip, resplen/len(ssdpRequest)))

            for x in res:
                packets.append(x[1])

            # store PCAPs of request/response
            if PCAP_TO_LOCAL_DISK == True:
                if not os.path.exists("/ampmap/pcap/"+serverip):
                    os.makedirs("/ampmap/pcap/"+serverip)

                pcap_filename = "/ampmap/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                self.query_cnt_dict[self.phase] += 1
                wrpcap(pcap_filename, packets)

            else:
                if not os.path.exists("out/pcap/"+serverip):
                    os.makedirs("out/pcap/"+serverip)

                pcap_filename = "out/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                self.query_cnt_dict[self.phase] += 1
                wrpcap(pcap_filename, packets)

            return resplen/len(ssdpRequest)

        # If there is no response
        else:
            print("server_ip: %s, AF: 0\n" %serverip)

            # store PCAPs of request/response
            if PCAP_TO_LOCAL_DISK == True:
                if not os.path.exists("/ampmap/pcap/"+serverip):
                    os.makedirs("/ampmap/pcap/"+serverip)

                pcap_filename = "/ampmap/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                self.query_cnt_dict[self.phase] += 1
                wrpcap(pcap_filename, packets)

            else:
                if not os.path.exists("out/pcap/"+serverip):
                    os.makedirs("out/pcap/"+serverip)

                pcap_filename = "out/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                self.query_cnt_dict[self.phase] += 1
                wrpcap(pcap_filename, packets)


            return 0


    # NTP: including private, normal, control modes
    def __ntp_dict(self, serverip, field_dict):
        packets = []
        
        print("Hey, you reach NTP dict...")

        if field_dict["mode"] == 7: 
            print("IN PRIVATE MODE 7 ")
            payload = NTPPrivate() 
        elif field_dict["mode"] == 6:
            print("IN CONTROL MODE 6")
            payload = NTPControl()
        else:
            payload = NTPHeader()

        for fid, val in field_dict.items(): 
            setattr(payload, fid, val)
        
        request = IP(dst=serverip)/UDP(sport=random.randint(5000,65535), dport=123)/payload
        packets.append(request)
        res, unans = sr(request, multi=True, timeout=self.timeout, verbose=0)

        if res is not None:
            for x in res:
                packets.append(x[1])

            resplen = sum([len(x[1]) for x in res])
            print("AF: ", float(resplen)/float(len(request)))


            if PCAP_TO_LOCAL_DISK == True:
                if not os.path.exists("/ampmap/pcap/"+serverip):
                    os.makedirs("/ampmap/pcap/"+serverip)

                pcap_filename = "/ampmap/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                self.query_cnt_dict[self.phase] += 1
                wrpcap(pcap_filename, packets)

            else:
                if not os.path.exists("out/pcap/"+serverip):
                    os.makedirs("out/pcap/"+serverip)

                pcap_filename = "out/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                self.query_cnt_dict[self.phase] += 1
                wrpcap(pcap_filename, packets)


            #time.sleep(5)
            return float(resplen)/float(len(request))
        else:
            print("AF: ", 0)
            print() 


            if PCAP_TO_LOCAL_DISK == True:
                if not os.path.exists("/ampmap/pcap/"+serverip):
                    os.makedirs("/ampmap/pcap/"+serverip)

                pcap_filename = "/ampmap/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                self.query_cnt_dict[self.phase] += 1
                wrpcap(pcap_filename, packets)

            else:
                if not os.path.exists("out/pcap/"+serverip):
                    os.makedirs("out/pcap/"+serverip)

                pcap_filename = "out/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                self.query_cnt_dict[self.phase] += 1
                wrpcap(pcap_filename, packets)



            return 0 


    # Quake
    def __quake_dict(self, serverip, field_dict): 
        #convert hex to bytes 
        packets = []
        data = bytearray()
        data += bytes.fromhex(field_dict["pre"])


        data += bytearray( field_dict["char"], "utf-8")

        post = field_dict["post"]

        post = struct.pack("B", post) 

        post = post * field_dict["len_post"]
        data += post 


        request = IP(dst=serverip)/UDP(sport=random.randint(5000,65535), dport=27960) \
            /Raw(load=data)

        packets.append(request)
        res, unans = sr(request, multi=True, timeout=self.timeout,  verbose=0)

        if res is not None:
            resplen = sum([len(x[1]) for x in res])
            print("AF: ", float(resplen)/float(len(request)))

            for x in res:
                packets.append(x[1])


            if PCAP_TO_LOCAL_DISK == True:
                if not os.path.exists("/ampmap/pcap/"+serverip):
                    os.makedirs("/ampmap/pcap/"+serverip)

                pcap_filename = "/ampmap/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                self.query_cnt_dict[self.phase] += 1
                wrpcap(pcap_filename, packets)

            else:
                if not os.path.exists("out/pcap/"+serverip):
                    os.makedirs("out/pcap/"+serverip)

                pcap_filename = "out/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                self.query_cnt_dict[self.phase] += 1
                wrpcap(pcap_filename, packets)

            return float(resplen)/float(len(request))
        else:
            print("AF: ", 0)
            print() 


            if PCAP_TO_LOCAL_DISK == True:
                if not os.path.exists("/ampmap/pcap/"+serverip):
                    os.makedirs("/ampmap/pcap/"+serverip)

                pcap_filename = "/ampmap/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                self.query_cnt_dict[self.phase] += 1
                wrpcap(pcap_filename, packets)

            else:
                if not os.path.exists("out/pcap/"+serverip):
                    os.makedirs("out/pcap/"+serverip)

                pcap_filename = "out/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                self.query_cnt_dict[self.phase] += 1
                wrpcap(pcap_filename, packets)

            
            return 0 


        return 

    # CharGen
    def __chargen_dict(self, serverip, field_dict):
        print("Hey, you reach chargen dic...")
        print(field_dict["character"], field_dict["length"])

        packets = []

        character = field_dict["character"]
        length = int(field_dict["length"])

        payload = bytearray()

        # if character is '0' - '9'
        if character >= '0' and character <= '9':
            payload = bytearray([int(character)])*length
        else:
            payload = bytearray(character, 'utf-8')*length

        request = IP(dst=serverip)/UDP(sport=random.randint(5000,65535), dport=19)/Raw(load=payload)
        print("request " , request)

        packets.append(request)
        res, unans = sr(request, multi=True, timeout=self.timeout,  verbose=0)

        print("response ", res)
            
        if res is not None:
            resplen = 0
            for x in res:
                if len(x) >= 2:
                    resplen += len(x[1])

            print("AF: ", float(resplen)/float(len(request)))
            print()

            for x in res:
                packets.append(x[1])



            if PCAP_TO_LOCAL_DISK == True:
                if not os.path.exists("/ampmap/pcap/"+serverip):
                    os.makedirs("/ampmap/pcap/"+serverip)

                pcap_filename = "/ampmap/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                self.query_cnt_dict[self.phase] += 1
                wrpcap(pcap_filename, packets)

            else:
                if not os.path.exists("out/pcap/"+serverip):
                    os.makedirs("out/pcap/"+serverip)

                pcap_filename = "out/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                self.query_cnt_dict[self.phase] += 1
                wrpcap(pcap_filename, packets)



            return float(resplen)/float(len(request))
        else:
            print("AF: ", 0)
            print() 



            if PCAP_TO_LOCAL_DISK == True:
                if not os.path.exists("/ampmap/pcap/"+serverip):
                    os.makedirs("/ampmap/pcap/"+serverip)

                pcap_filename = "/ampmap/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                self.query_cnt_dict[self.phase] += 1
                wrpcap(pcap_filename, packets)

            else:
                if not os.path.exists("out/pcap/"+serverip):
                    os.makedirs("out/pcap/"+serverip)

                pcap_filename = "out/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                self.query_cnt_dict[self.phase] += 1
                wrpcap(pcap_filename, packets)

            
            return 0 

    # memcached
    def __memcached_dict(self, serverip, field_dict):
        packets = []
        print("Hey, you reach memcached dic...")
        data = bytearray()

        data += b'\x00\x00\x00\x00\x00\x01\x00\x00'
        data = data + bytearray(field_dict["command"], 'utf-8')
        if field_dict["key"] != "": 
            data += bytearray(" ", 'utf-8')
        data += bytearray(field_dict["key"], 'utf-8')
        data += b'\r\n'

        print("data ", data)
        request = IP(dst=serverip)/UDP(sport=random.randint(5000,65535),  dport=11211)/Raw(load=data)
        print("request " , request)

        packets.append(request)

        res, unans = sr(request, multi=True, timeout=self.timeout, verbose=0)

        
        if res is not None:
            resplen = sum([len(x[1]) for x in res])
            print("AF: ", float(resplen)/float(len(request)))
            
            for x in res:
                packets.append(x[1])


            if PCAP_TO_LOCAL_DISK == True:
                if not os.path.exists("/ampmap/pcap/"+serverip):
                    os.makedirs("/ampmap/pcap/"+serverip)

                pcap_filename = "/ampmap/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                self.query_cnt_dict[self.phase] += 1
                wrpcap(pcap_filename, packets)

            else:
                if not os.path.exists("out/pcap/"+serverip):
                    os.makedirs("out/pcap/"+serverip)

                pcap_filename = "out/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                self.query_cnt_dict[self.phase] += 1
                wrpcap(pcap_filename, packets)



            return float(resplen)/float(len(request))
        else:
            print("AF: ", 0)
            print() 


            if PCAP_TO_LOCAL_DISK == True:
                if not os.path.exists("/ampmap/pcap/"+serverip):
                    os.makedirs("/ampmap/pcap/"+serverip)

                pcap_filename = "/ampmap/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                self.query_cnt_dict[self.phase] += 1
                wrpcap(pcap_filename, packets)

            else:
                if not os.path.exists("out/pcap/"+serverip):
                    os.makedirs("out/pcap/"+serverip)

                pcap_filename = "out/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                self.query_cnt_dict[self.phase] += 1
                wrpcap(pcap_filename, packets)



            return 0 

    # rpc
    def __rpc_dict(self, server_ip, field_dict):
        msg = rpc.RPCCall( )

    # SNMP_Bulk
    def __snmpbulk_dict(self, serverip, field_dict): 
        version = field_dict["version"]
        community = field_dict["community"]
        id = field_dict["id"]
        non_repeaters = field_dict["non_repeaters"]
        max_repetitions = field_dict["max_repetitions"]
        varbind_oid = field_dict["varbind_oid"]
        varbind_multiple = field_dict["varbind_multiple"]

        print(varbind_oid)

        print("field dic t" , field_dict) 

        oid_lst  = [SNMPvarbind( oid=ASN1_OID( varbind_oid  )    )] * varbind_multiple

        print ( " oid lst ", oid_lst)

        pdutype = SNMPbulk(id= id, non_repeaters = non_repeaters,  max_repetitions=max_repetitions, \
            varbindlist= oid_lst )

        snmppacket = SNMP(version=version, community=community, PDU=pdutype)
        request = IP(dst=serverip)/UDP(sport=random.randint(5000,65535),dport=161)/snmppacket
        packets = []
        packets.append(request)

        res, unans = sr(request, multi=True, timeout=self.timeout, verbose=0)


        if res is not None: 

            resplen = sum([len(x[1]) for x in res])
            print("AF: ", float(resplen)/float(len(request)))


            for x in res:
                packets.append(x[1])

            if PCAP_TO_LOCAL_DISK == True:
                if not os.path.exists("/ampmap/pcap/"+serverip):
                    os.makedirs("/ampmap/pcap/"+serverip)

                pcap_filename = "/ampmap/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                self.query_cnt_dict[self.phase] += 1
                wrpcap(pcap_filename, packets)

            else:
                if not os.path.exists("out/pcap/"+serverip):
                    os.makedirs("out/pcap/"+serverip)

                pcap_filename = "out/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                self.query_cnt_dict[self.phase] += 1
                wrpcap(pcap_filename, packets)


            return float(resplen)/float(len(request))

        else:
            print("AF: ", 0)

            if PCAP_TO_LOCAL_DISK == True:
                if not os.path.exists("/ampmap/pcap/"+serverip):
                    os.makedirs("/ampmap/pcap/"+serverip)

                pcap_filename = "/ampmap/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                self.query_cnt_dict[self.phase] += 1
                wrpcap(pcap_filename, packets)

            else:
                if not os.path.exists("out/pcap/"+serverip):
                    os.makedirs("out/pcap/"+serverip)

                pcap_filename = "out/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                self.query_cnt_dict[self.phase] += 1
                wrpcap(pcap_filename, packets)



            return 0


    # SNMP_Next / SNMP_Get
    def __snmpstandard_dict(self, serverip, field_dict , SNMPftn): 
        version = field_dict["version"]
        community = field_dict["community"]
        id = field_dict["id"]
        error = field_dict["error"]
        error_index = field_dict["error_index"]
        varbind_oid = field_dict["varbind_oid"]
        varbind_multiple = field_dict["varbind_multiple"]


        oid_lst  = [SNMPvarbind( oid=ASN1_OID( varbind_oid  )    )]  * varbind_multiple


        pdutype = SNMPftn(id= id, error = error, error_index = error_index,  varbindlist= oid_lst )

        snmppacket = SNMP(version=version, community=community, PDU=pdutype)

        request = IP(dst=serverip)/UDP(sport=random.randint(5000,65535),dport=161)/snmppacket
        packets = []
        packets.append(request)

        res, unans = sr(request, multi=True, timeout=self.timeout, verbose=0)


        if res is not None: 

            resplen = sum([len(x[1]) for x in res])
            print("AF: ", float(resplen)/float(len(request)))


            for x in res:
                packets.append(x[1])


            if PCAP_TO_LOCAL_DISK == True:
                if not os.path.exists("/ampmap/pcap/"+serverip):
                    os.makedirs("/ampmap/pcap/"+serverip)

                pcap_filename = "/ampmap/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                self.query_cnt_dict[self.phase] += 1
                wrpcap(pcap_filename, packets)

            else:
                if not os.path.exists("out/pcap/"+serverip):
                    os.makedirs("out/pcap/"+serverip)

                pcap_filename = "out/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                self.query_cnt_dict[self.phase] += 1
                wrpcap(pcap_filename, packets)



            return float(resplen)/float(len(request))

        else:
            print("AF: ", 0)


            if PCAP_TO_LOCAL_DISK == True:
                if not os.path.exists("/ampmap/pcap/"+serverip):
                    os.makedirs("/ampmap/pcap/"+serverip)

                pcap_filename = "/ampmap/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                self.query_cnt_dict[self.phase] += 1
                wrpcap(pcap_filename, packets)

            else:
                if not os.path.exists("out/pcap/"+serverip):
                    os.makedirs("out/pcap/"+serverip)

                pcap_filename = "out/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                self.query_cnt_dict[self.phase] += 1
                wrpcap(pcap_filename, packets)

            return 0


    # DNS: without EDNS field
    def __dns_message_noedns_dict(self, serverip, field_vals): 
        try:
            packets = []

            print("IN NOEDNS section")
            id = field_vals["id"]
            qr = field_vals["qr"]
            aa = field_vals["aa"]
            tc = field_vals["tc"]
            rd = field_vals["rd"]
            ra = field_vals["ra"]
            cd = field_vals["cd"]
            ad = field_vals["ad"]
            opcode = field_vals["opcode"]
            rcode = field_vals["rcode"]
            url = field_vals["url"]
            rdataclass = field_vals["rdataclass"]
            rdatatype = field_vals["rdatatype"]
           
            m = dns.message.Message()
            m.id = id
            if qr:
                m.flags |=  int(dns.flags.QR)
            if aa:
                m.flags |=  int(dns.flags.AA)
            if tc:
                m.flags |=  int(dns.flags.TC)
            if rd:
                m.flags |=  int(dns.flags.RD)
            if ra:
                m.flags |=  int(dns.flags.RA )        
            if ad:
                m.flags |=  int(dns.flags.AD )        
            if cd:
                m.flags |=  int(dns.flags.CD )        
           
            m.set_opcode(int(opcode))
            m.set_rcode(int(rcode))

            qname = dns.name.from_text(url)
            m.find_rrset(m.question, qname , rdataclass  , rdatatype , create=True, force_unique=True) 
            
            data = m.to_wire()

            request = IP(dst=serverip)/UDP(sport=random.randint(5000,65535),dport=53)/Raw(load=data)
            print("request ", request)
            packets.append(request)

            ###################### write to pcap then read ##################
            # NOTE: to correctly parse DNS request using scapy, 
            #       we write the request first into a pcap then read
            #       to ensure the correct packet format

            if PCAP_TO_LOCAL_DISK == True:

                if not os.path.exists("/ampmap/pcap_temp/"+serverip):
                    os.makedirs("/ampmap/pcap_temp/"+serverip)

                temp_pcap_filename = "/ampmap/pcap_temp/"+serverip+"/temp.pcap"

                wrpcap(temp_pcap_filename, packets)
                request = rdpcap(temp_pcap_filename)[0]
            
            else:
                if not os.path.exists("out/pcap_temp/"+serverip):
                    os.makedirs("out/pcap_temp/"+serverip)
                temp_pcap_filename = "out/pcap_temp/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                wrpcap(temp_pcap_filename, packets)
                request = rdpcap(temp_pcap_filename)[0]

            #################################################################


            res, unans= sr(request, multi=True, timeout=self.timeout, verbose=0)

            if res is not None: 
                for x in res:
                    packets.append(x[1])

                resplen = sum([len(x[1]) for x in res])
                print("AF: ", float(resplen)/float(len(request)))
                
                # pcap dump...

                if PCAP_TO_LOCAL_DISK == True:
                    if not os.path.exists("/ampmap/pcap/"+serverip):
                        os.makedirs("/ampmap/pcap/"+serverip)

                    pcap_filename = "/ampmap/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                    self.query_cnt_dict[self.phase] += 1
                    wrpcap(pcap_filename, packets)

                else:
                    if not os.path.exists("out/pcap/"+serverip):
                        os.makedirs("out/pcap/"+serverip)

                    pcap_filename = "out/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                    self.query_cnt_dict[self.phase] += 1
                    wrpcap(pcap_filename, packets)


                return float(resplen)/float(len(request))

            else:
                print("AF: ", 0)
                print()

                if PCAP_TO_LOCAL_DISK == True:
                    if not os.path.exists("/ampmap/pcap/"+serverip):
                        os.makedirs("/ampmap/pcap/"+serverip)

                    pcap_filename = "/ampmap/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                    self.query_cnt_dict[self.phase] += 1
                    wrpcap(pcap_filename, packets)

                else:
                    if not os.path.exists("out/pcap/"+serverip):
                        os.makedirs("out/pcap/"+serverip)

                    pcap_filename = "out/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                    self.query_cnt_dict[self.phase] += 1
                    wrpcap(pcap_filename, packets)


                return 0

        except dns.exception.DNSException:
            return 0 


    # DNS
    def __dns_message_dict(self, serverip, field_vals): 
        try:
            packets = []
            m = dns.message.Message()

            id = field_vals["id"]
            qr = field_vals["qr"]
            aa = field_vals["aa"]
            tc = field_vals["tc"]
            rd = field_vals["rd"]
            ra = field_vals["ra"]
            cd = field_vals["cd"]
            ad = field_vals["ad"]
            opcode = field_vals["opcode"]
            rcode = field_vals["rcode"]
            edns = field_vals["edns"]
            payload = field_vals["payload"]
            url = field_vals["url"]
            rdataclass = field_vals["rdataclass"]
            rdatatype = field_vals["rdatatype"]
            dnssec =  field_vals["dnssec"]

            m.id = id
            if qr:
                m.flags |=  int(dns.flags.QR)
            if aa:
                m.flags |=  int(dns.flags.AA)
            if tc:
                m.flags |=  int(dns.flags.TC)
            if rd:
                m.flags |=  int(dns.flags.RD)
            if ra:
                m.flags |=  int(dns.flags.RA )        
            if ad:
                m.flags |=  int(dns.flags.AD )        
            if cd:
                m.flags |=  int(dns.flags.CD )        
             
            m.set_opcode(int(opcode))
            m.set_rcode(int(rcode))

            m.edns = int(edns)
            m.payload=int(payload)
            if dnssec:
                m.ednsflags |= int( dns.flags.DO)
                
            qname = dns.name.from_text(url)     

            m.find_rrset(m.question, qname , rdataclass  , rdatatype , create=True,
                             force_unique=True) 

            data = m.to_wire()

            request = IP( dst=serverip)/UDP(sport=random.randint(5000,65535),dport=53)/Raw(load=data)
            #print("request ", request)
            packets.append(request)

            ###################### write to pcap then read ##################
            # NOTE: to correctly parse DNS request using scapy, 
            #       we write the request first into a pcap then read
            #       to ensure the correct packet format

            if PCAP_TO_LOCAL_DISK == True:

                if not os.path.exists("/ampmap/pcap_temp/"+serverip):
                    os.makedirs("/ampmap/pcap_temp/"+serverip)

                temp_pcap_filename = "/ampmap/pcap_temp/"+serverip+"/temp.pcap"

                wrpcap(temp_pcap_filename, packets)
                request = rdpcap(temp_pcap_filename)[0]
            
            else:
                if not os.path.exists("out/pcap_temp/"+serverip):
                    os.makedirs("out/pcap_temp/"+serverip)
                temp_pcap_filename = "out/pcap_temp/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                wrpcap(temp_pcap_filename, packets)
                request = rdpcap(temp_pcap_filename)[0]
            #################################################################


            res, unans= sr(request, multi=True, timeout=self.timeout, verbose=0)

            if res is not None: 
                for x in res:
                    packets.append(x[1])

                resplen = sum([len(x[1]) for x in res])
                print("AF: ", float(resplen)/float(len(request)))
                
                # pcap dump...
                if PCAP_TO_LOCAL_DISK == True:
                    if not os.path.exists("/ampmap/pcap/"+serverip):
                        os.makedirs("/ampmap/pcap/"+serverip)

                    pcap_filename = "/ampmap/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                    self.query_cnt_dict[self.phase] += 1
                    wrpcap(pcap_filename, packets)

                else:
                    if not os.path.exists("out/pcap/"+serverip):
                        os.makedirs("out/pcap/"+serverip)

                    pcap_filename = "out/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                    self.query_cnt_dict[self.phase] += 1
                    wrpcap(pcap_filename, packets)


                return float(resplen)/float(len(request))

            else:
                print("AF: ", 0)
                print()

                if PCAP_TO_LOCAL_DISK == True:
                    if not os.path.exists("/ampmap/pcap/"+serverip):
                        os.makedirs("/ampmap/pcap/"+serverip)

                    pcap_filename = "/ampmap/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                    self.query_cnt_dict[self.phase] += 1
                    wrpcap(pcap_filename, packets)

                else:
                    if not os.path.exists("out/pcap/"+serverip):
                        os.makedirs("out/pcap/"+serverip)

                    pcap_filename = "out/pcap/"+serverip+"/"+self.phase+"_"+str(self.query_cnt_dict[self.phase])+".pcap"
                    self.query_cnt_dict[self.phase] += 1
                    wrpcap(pcap_filename, packets)

                return 0

        except dns.exception.DNSException:
            return 0

    # Given protocol, server ip and query, return the amplification factor (AF) and the response
    def get_af_dict(self, serverip, field_dict):
        if self.proto.lower() == "dns": 
            if "edns" in field_dict: 
                return self.__dns_message_dict( serverip, field_dict )
            else: 
                return self.__dns_message_noedns_dict(serverip, field_dict )

        elif self.proto.lower() == "memcached":
            return self.__memcached_dict( serverip, field_dict)

        elif self.proto.lower() =='chargen' : 
            return self.__chargen_dict( serverip, field_dict)

        elif self.proto.lower() == "ntp":
            return self.__ntp_dict(serverip, field_dict )

        elif self.proto.lower() == "ssdp":
            return self.__ssdp_dict(serverip, field_dict)

        elif self.proto.lower() == "quake":
            return self.__quake_dict(serverip, field_dict)

        elif self.proto.lower() == "snmpbulk":
            return self.__snmpbulk_dict(serverip, field_dict)

        elif self.proto.lower() == "snmpnext": 
            return self.__snmpstandard_dict(serverip, field_dict, SNMPnext) 

        elif self.proto.lower() == "snmpget": 
            return self.__snmpstandard_dict(serverip, field_dict, SNMPget) 

        else: 
            raise ValueError("Protocol is not supported ")

    def get_af(self, serverip, field_name, field_values):
        assert(len(field_name) == len(field_values))
        field_dict = OrderedDict(zip(field_name, field_values))
        return self.get_af_dict( serverip, field_dict ) 
    
    def register_protocol(self,proto):
        self.proto = proto

    def register_phase(self, phase):
        self.phase = phase
        self.query_cnt_dict[phase] = 1

def blackbox(timeout): 
    return BlackBox(timeout)

