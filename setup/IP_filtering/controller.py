# controller.py for IP filter
# Input is a list of raw (unfiltered) ips

import sys, os
import configparser
import time
import subprocess

def make_out_dir(out_dir):
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

def read_raw_ips(filename):
	raw_ips_total = []
	f = open(filename, 'r')
	for line in f:
		raw_ips_total.append(line.strip())

	return raw_ips_total

def chunks(a, n):
	k, m = divmod(len(a), n)
	return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def read_nodes_ip(filename):
	measurer_servers = []
	config = configparser.ConfigParser()
	config.read(filename)

	for i in range(int(config["measurers"]["numMeasurers"])):
		ip = config["Measurer_"+str(i+1)]["ip"]
		measurer_servers.append(ip)
		print(i+1, ip)

	return measurer_servers

def check_status(cmd_file, status):
	f = open(cmd_file, 'r')
	lines = f.readlines()
	if lines[0].strip() == status:
		return 1
	else:
		return 0

def check_is_finished(cmd_dir, measurer_servers):
	is_finished = True

	for measurer_ip in measurer_servers:
		if check_status(os.path.join(cmd_dir, measurer_ip), "end") == 1:
			print("node %s finished!" %measurer_ip)
		else:
			print("node %s not finished!" %measurer_ip)
			is_finished = False

	return is_finished

def output_filter_ips(filter_split_dir, outfile):
	files = os.listdir(filter_split_dir)
	fout = open(outfile, 'w')
	for file in files:
		fin = open(os.path.join(filter_split_dir, file), 'r')
		lines = fin.readlines()
		for line in lines:
			fout.write(line)
		fin.close()

	fout.close()



def main():
	# make out dir
	make_out_dir("out")

	# read parameters.ini
	proto = sys.argv[1].lower()
	config = configparser.ConfigParser()
	config.read("parameters.ini")

	measurer_config = configparser.ConfigParser()
	measurer_config.read("measurer.ini")

	proto_params_key = proto+"_params"

	raw_ips_filename = config[proto_params_key]["raw_ips_filename"]
	num_measurers = int(measurer_config["measurers"]["numMeasurers"])
	measurer_servers = read_nodes_ip(config[proto_params_key]["node_ips_filename"])


	# Clean up remaining python3 processes if any
	for i, node_ip in enumerate(measurer_servers):
		cmd = "sudo ssh -o StrictHostKeyChecking=no root@{} \"sudo killall python3\""
		cmd = cmd.format(node_ip)
		print(cmd)

		subprocess.Popen(cmd, stderr=sys.stdout.fileno(), shell=True)
		time.sleep(2)


	# write commands
	command_dir = config[proto_params_key]["command_dir"]
	make_out_dir(command_dir)
	for ip in measurer_servers:
		f = open(os.path.join(command_dir, ip), 'w')
		f.write("start\n")
		f.close()

	# create logs
	log_dir = config[proto_params_key]["log_dir"]
	make_out_dir(log_dir)

	# create filter ips dir: for job.py use
	filter_split_dir = config[proto_params_key]["filter_split_dir"]
	make_out_dir(filter_split_dir)

	# read raw ips and split
	raw_ips_total = read_raw_ips(raw_ips_filename)
	raw_ips_splited = chunks(raw_ips_total, num_measurers)

	raw_ips_split_dir = config[proto_params_key]["raw_split_dir"]
	make_out_dir(raw_ips_split_dir)

	cnt = 1
	for raw_ips_split in raw_ips_splited:
		f = open(os.path.join(raw_ips_split_dir, proto+"_raw_ips_"+str(cnt)), 'w')
		for ip in raw_ips_split:
			f.write("%s\n" %ip)
		f.close()
		cnt = cnt + 1

	time.sleep(2)

	# count dir storing cnt stats
	count_dir = config[proto_params_key]["count_dir"]
	make_out_dir(count_dir)


	# launch job.py for each measurer
	activate_conda = "source /root/anaconda3/bin/activate mypy3"
	for i, node_ip in enumerate(measurer_servers):
		raw_split_filename = os.path.join(raw_ips_split_dir, proto+"_raw_ips_"+str(i+1))
		log_filename = os.path.join(log_dir, node_ip)
		cmd = "sudo ssh -o StrictHostKeyChecking=no root@{} \"cd {} && {} && python3 job.py {} {} {} >> {} 2>&1 &\""

		cmd = cmd.format(node_ip, config["cloudlab_config"]["ip_filter_dir"], activate_conda,  \
			proto, node_ip, raw_split_filename, log_filename)
		print(cmd)

		subprocess.Popen(cmd, stderr=sys.stdout.fileno(), shell=True)
		time.sleep(3)

	# check whether all measurers finishes the filtering
	while True:
		if check_is_finished(command_dir, measurer_servers) == False:
			print("In Progress!!!")
		else:
			print("All finished!!!")
			output_filter_ips(filter_split_dir, config[proto_params_key]["filter_ip_outfile"])
			break

		time.sleep(5)


if __name__ == '__main__':
	main()