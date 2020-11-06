#!/bin/bash 


# Figure 9 and 10 
echo "Figure 9 and 10 " 
python fig9_10_protocol_summary.py

echo "Table 5 and 6 -- CALCULATING risk " 
python  querypattern_risk_metric_v3.py  --qp_dir ./data_dir/query_searchout_DNS_20200516/  --sig_input_file ./known_patterns/dns_any_only.json   --proto dns --out_dir risk_out/out_dns_10k_any_only 

echo "comparing risk " 
python compare_risk.py --risk_dir risk_out


sleep 1 
echo "Figure 12 " 

python fig12_analyze_proto.py --qps_data_folder data_dir/query_searchout_DNS_20200516/ --proto dns --alias dns    --intermediate_data_folder intermediate_files --out_dir figs --signature
 
sleep 1 
echo "Figure 13" 
 python fig13_dns_stackekd_bar.py --out_dir figs/ --parsed_data figs/dns/


