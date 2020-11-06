#!/bin/bash 


# Suppose that you already installed anaconda: 
# https://docs.anaconda.com/anaconda/install/


sudo apt-get install graphviz
conda create -n ampmap python=3.7
pip install -r requirements.txt
