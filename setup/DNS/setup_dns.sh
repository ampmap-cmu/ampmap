LOCAL_IP=`ifconfig | grep Bcast | tr -s ' ' |  cut -d' ' -f4 | cut -d':' -f2`
NET_MASK=`ifconfig | grep Bcast | tr -s ' '     | cut -d' ' -f5 | cut -d':' -f2`

sudo apt-get update
sudo apt-get --assume-yes install vim
sudo apt-get --assume-yes install tmux
sudo apt-get --assume-yes install bind9 bind9utils bind9-doc

echo "127.0.0.1 $HOSTNAME" | sudo tee -a /etc/hosts
sudo cp named.conf.options /etc/bind/
sudo cp resolv.conf /etc/
sudo named-checkconf
sudo service bind9 restart
