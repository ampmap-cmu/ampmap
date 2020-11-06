#!/bin/bash 


if [ ! -d "/tmp" ] 
then
	echo "makign folder"
   sudo mkdir /tmp 

fi

cd /tmp 

sudo curl -O https://repo.anaconda.com/archive/Anaconda3-4.3.1-Linux-x86_64.sh

sudo rm -rf /root/anaconda3

export HOME="/root"
sudo bash Anaconda3-4.3.1-Linux-x86_64.sh  -b 

sudo bash -c  'echo "export PATH=\$PATH:/root/anaconda3/bin" >> /root/.bashrc'

/root/anaconda3/bin/conda create -n mypy3 python=3.6 anaconda  -y

source /root/anaconda3/bin/activate mypy3
/root/anaconda3/bin/conda install -c conda-forge hyperopt -y
/root/anaconda3/bin/conda install -c anaconda dnspython -y

# installing scapy through conda just does not work on this VM
# /root/anaconda3/bin/conda install -c conda-forge scapy  -y

# install scapy using pip
/root/anaconda3/bin/conda install -c anaconda pip -y
pip install scapy


sudo bash -c  'echo "source /root/anaconda3/bin/activate mypy3" >> /root/.bashrc'
