#!/bin/bash
# A script to install everything needed to run MOSES in a new environment
set -e
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
