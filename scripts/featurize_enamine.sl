#!/usr/bin/bash

#SBATCH --time 1-00:00:00
#SBATCH --partition pbatch
#SBATCH --array=1-11

source activate lbann
cd /g/g13/jones289/workspace/lbann/applications/ATOM/moses/scripts

python preprocess_data.py --vocab-path /g/g13/jones289/workspace/lbann/applications/ATOM/data/enamine/full_vocab.pt --smiles-path ~/data/enamine/2018q1-2_Enamine_REAL_680M_SMILES_part${SLURM_ARRAY_TASK_ID}.smiles --smiles-col smiles --smiles-sep '\t' --add-bos --add-eos --n-jobs 72 --test-size 0.2 --val-size 0.1 --split-dataset --output-dir /g/g13/jones289/workspace/lbann/applications/ATOM/data/enamine/part${SLURM_ARRAY_TASK_ID} 
