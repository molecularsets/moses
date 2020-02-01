#!/bin/bash
# A script to install everything needed to run MOSES in a new environment
set -e
conda create -y -n moses_env python=3.7
eval "$(conda shell.bash hook)"
conda activate moses_env
git clone https://github.com/pcko1/Deep-Drug-Coder.git --branch moses
cd Deep-Drug-Coder
python setup.py install
cd ..
git clone https://github.com/EBjerrum/molvecgen.git
cd molvecgen
python setup.py install
cd ..
conda install -y tensorflow-gpu==1.12
conda install -y -c conda-forge rdkit
python setup.py install
