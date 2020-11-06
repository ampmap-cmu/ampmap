#!/bin/bash


proto='DNS'
      #Number of servers to 
root_dir=$1 
array=('ampmap' )
totalblist=( 100 200 500 800 1200 1500  )
num_dfs_starting=(  1  ) 
choose_starting=( 'max-val' ) 


temp=0 

#random_percent=53
random_percent=50
probe_percent=0
#num_serv_sim=20

URLSETS=('urlset1'  'urlset1-noAnyTxtRRSIG')
server_ip='172.31.34.176'

for urlset in "${URLSETS[@]}"
do

	#urlset='urlset1'
	results_folder="$root_dir/results_"$urlset
	

	if [ "$urlset" = "urlset1" ]; then 

		config='field_inputs_dns'
	elif [ "$urlset" = "urlset1-noAnyTxtRRSIG" ]; then
		config='field_inputs_dns_noAnyTxtRRSIG_analysis'
	else 
		echo " SHOULD NOT REACH HERE"
		exit
	fi 
	echo "CONFIG IS " $config


	mkdir -p $results_folder

	

	for i in {1..20}
	do
		for totalb in "${totalblist[@]}"
		do
			for algo in  "${array[@]}"
			do 

					for K in  "${num_dfs_starting[@]}"
					do  
						for chooseK in  "${choose_starting[@]}"
						do 

								if [[ $algo == *"ampmap"* ]]; then 
								#if [ "$algo" = "ampmap_sim"  ];then



									iteration=$algo'--num_starting'$K'--'$chooseK'--'$urlset 

									sudo rm -rf out

									# 15 % of the total budget 
									let "rand = $random_percent * $totalb"
									let "orand = $rand / 100"

									# 5	% of the total budget 
									let "probe =  $probe_percent  * $totalb"
									let "probe = $probe/ 100"

									echo "Original:" $totalb," Random "$rand," Probe "$probe," Total "  $totalb
									
									#exit 



									#[ "$algo" = "ampmap_approach" ];then
									if [ "$chooseK" = "max-val" ]; then 
										opt_flag="--choose_K_max_val"
									elif [ "$chooseK" = "max-dist" ]; then
										opt_flag="--choose_K_max_dist"
									elif [ "$chooseK" = "random" ]; then
										opt_flag="--choose_K_random"
									fi 

									if [[ $algo == *"disable_sampling"* ]]; then 
										opt_flag=$opt_flag' --disable_sampling'
									elif [[ $algo == *"disable_check_new_mode"* ]]; then 
										opt_flag=$opt_flag' --disable_check_new_mode'

									fi 

									file=$results_folder/$iteration"--"$totalb"--"$orand"--"$i.out
									fold=$results_folder/$iteration"--"$totalb"--"$orand"--"$i
									echo $file
									if [ -e "$file" ]; then
										    echo "File exists"
										        continue
									else
										echo "File does not exist"
											        # continue
									fi

								python3 controller.py  --per_server_random_sample $orand --num_probe $probe --per_server_budget $totalb \
										 --in_dir $config  --update_db_at_once 20 --query_wait 0.1 \
										--measurement --server_timeout 0.2   --proto $proto  \
										--choose_K $K  $opt_flag      
								fi



			            

								if [ "$algo" = "random_accepted" ];then
									#iteration=$algo

									iteration=$algo'--num_starting'$temp'--'$temp'--'$urlset 


									file=$results_folder/$iteration"--"$totalb"--"$totalb"--"$i.out
									fold=$results_folder/$iteration"--"$totalb"--"$totalb"--"$i
									echo $file
									if [ -e "$file" ]; then
										    echo "File exists"
										        continue
									else
										echo "File does not exist"
											        # continue
									fi
									python3 controller.py 	--per_server_random_sample $totalb --per_server_budget $totalb --in_dir $config  --update_db_at_once 20 --query_wait 0.1 \
										--measurement --server_timeout 0.2 --proto $proto --random_accepted

								fi 




								cp out/searchout/$server_ip'.out' $file
								cp out/searchout/$server_ip'.out.total' $file'.total'

								sleep 2

								cp -r out $fold
								sleep 2




								#exit
						done 
					done 
				#done 
			done 
		done
	done
done
