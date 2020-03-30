#!/usr/bin/bash

#SBATCH --time 1-00:00:00
#SBATCH --partition pbatch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --array=1-11

#define the variables for location of moses repo (MOSES_HOME), directory containing smiles chunks (SMILES_DIR), and the directory where output is to be stored (OUTPUT_DIR)
#it is assumedthe code will be run from $MOSES_HOME/scripts

MOSES_HOME=/g/g13/jones289/workspace/lbann/applications/ATOM/moses
SMILES_DIR=/p/lustre2/jones289/data/enamine
OUTPUT_DIR=/g/g13/jones289/workspace/lbann/applications/ATOM/data/enamine

# activate the python environment of you choice below...if using spack then just comment out
source activate lbann

cd $MOSES_HOME/scripts

python compute_vocab_main.py --smiles-path ${SMILES_DIR}/2018q1-2_Enamine_REAL_680M_SMILES_part${SLURM_ARRAY_TASK_ID}.smiles --smiles-col smiles --smiles-sep='\t' --n-jobs ${SLURM_CPUS_PER_TASK} --output-dir ${OUTPUT_DIR}/part${SLURM_ARRAY_TASK_ID}

