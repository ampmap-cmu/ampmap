#/bin/bash

proto=$1

echo $proto
echo "Available Options: dns, ssdp, ntp_cnt, ntp_cnt_or, ntp_pvt, ntp_pvt_or, ntp_normal, ntp_normal_or, snmp_next, snmp_next_or, snmp_bulk, snmp_bulk_or, snmp_get, snmp_get_or, chargen, memcached"

if [[ "$proto" == "DNS" ]]
then
  echo "Running DNS"
    python rad_dns.py

else
  echo "Running" $proto
    python rad_others.py $proto
fi 
