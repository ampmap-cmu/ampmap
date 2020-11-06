[all]
kl_thresh=0.08
af_thresh=10
data_path=./data_dir/
sv_path=./signatures/

[ssdp]
ipp  =field_inputs_ssdp
proto_name=query_searchout_SSDP_20200523




[ntp_cnt]
ipp  =field_inputs_ntpControl
proto_name=query_searchout_NTPControl_AND_20200608

[ntp_pvt]
ipp  =field_inputs_ntpPrivate
proto_name=query_searchout_NTPPrivate_AND_20200526

[ntp_normal]
ipp  =field_inputs_ntpNormal
proto_name=query_searchout_NTPNormal_AND_20200608


[snmp_next]
ipp  =field_inputs_snmp_standard
proto_name=query_searchout_SNMPNext_AND_20200601

[snmp_bulk]
ipp  =field_inputs_snmp_bulk
proto_name=query_searchout_SNMPBulk_AND_20200528

[snmp_get]
ipp  =field_inputs_snmp_standard
proto_name=query_searchout_SNMPGet_AND_20200529


[snmp_next_or]
ipp  =field_inputs_snmp_standard
proto_name=query_searchout_SNMPNext_OR_20200608

[snmp_bulk_or]
ipp  =field_inputs_snmp_bulk
proto_name=query_searchout_SNMPBulk_20200521

[snmp_get_or]
ipp  =field_inputs_snmp_standard
proto_name=query_searchout_SNMPGet_OR_20200531


[chargen]
ipp  =field_inputs_chargen
proto_name=query_searchout_chargen_20200524

[memcached]
ipp  =field_inputs_memcached
proto_name=query_searchout_memcached_20200526

[ntp_cnt_or]
ipp  =field_inputs_ntpControl
proto_name=query_searchout_NTPControl_20200521

[ntp_pvt_or]
ipp  =field_inputs_ntpPrivate
proto_name=query_searchout_NTPPrivate_20200518

[ntp_normal_or]
ipp  =field_inputs_ntpNormal
proto_name=query_searchout_NTPNormal_OR_20200529

[dns]
ipp  =field_inputs_dns_db2
proto_name=query_searchout_DNS_20200516


[group1]
l1=[ntp_cnt_or,ntp_pvt_or,ntp_normal_or]

[group2]
l1=[ntp_cnt,ntp_pvt,ntp_normal]

[group3]
l1=[snmp_next_or,snmp_bulk_or,snmp_get_or]

[group4]
l1=[snmp_next,snmp_bulk,snmp_get]