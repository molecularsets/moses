#!/usr/bin/bash

#SBATCH --time 1-00:00:00
#SBATCH --partition pbatch
#SBATCH --array=1-11

source activate lbann
cd /g/g13/jones289/workspace/lbann/applications/ATOM/moses/scripts

python compute_charrnn_vocab.py --smiles-path ~/data/enamine/2018q1-2_Enamine_REAL_680M_SMILES_part${SLURM_ARRAY_TASK_ID}.smiles --smiles-col smiles --smiles-sep='\t' --n-jobs 72 --output-dir /g/g13/jones289/workspace/lbann/applications/ATOM/data/enamine/part${SLURM_ARRAY_TASK_ID}
