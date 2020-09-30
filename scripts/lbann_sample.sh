#!/usr/bin/bash

# this is a driver script for the python lbann_sample.py code


MODEL="char_rnn"
#LBANN_WEIGHTS_DIR="/p/lustre1/jones289/lbann/weights/5M/weights/trainer0/model0"
LBANN_WEIGHTS_DIR="/p/lustre1/jones289/lbann/weights/620M/weights/trainer0/model0"
VOCAB_PATH="/p/lustre1/jones289/lbann/vocab_all2018q1-2REAL680SMILES.pt"
DROPOUT=0
NUM_LAYERS=1
HIDDEN=768
WEIGHT_PREFIX="sgd.training"
N_SAMPLES=10000
MAX_LEN=100
N_BATCH=48
#GEN_SAVE="train_5m"
GEN_SAVE="train_620m"
TEST_PATH="lbann_data/data/splits/test_all2018q1-2REAL10kSMILES.csv"
#PTEST_PATH=" "
#PTEST_SCAFFOLDS_PATH=" "
KS="100 1000"
N_JOBS=12
GPU=-1  #having some issues with the eval code not being able to see the gpu...sounds like a TF problem
BATCH_SIZE=16  # not sure if this is needed for eval?
# there's going to be one embedding matrix so...
weight_files=$(find $LBANN_WEIGHTS_DIR/$WEIGHT_PREFIX*-emb_matrix-Weights.txt)

for weight in ${weight_files[@]}
do
    weight_prefix=${weight%%-*}

    lbann_load_epoch=${weight##*epoch.}
    lbann_load_epoch=${lbann_load_epoch%%.*}

    lbann_load_step=${weight##*step.}
    lbann_load_step=${lbann_load_step%%-*}

    gen_save=${GEN_SAVE}_${WEIGHT_PREFIX}_${lbann_load_epoch}_${lbann_load_step}_gen.csv
    metrics_save=${GEN_SAVE}_${WEIGHT_PREFIX}_${lbann_load_epoch}_${lbann_load_step}_metrics.csv


    python lbann_sample.py --model $MODEL --lbann-weights-dir $LBANN_WEIGHTS_DIR --lbann-load-epoch $lbann_load_epoch --lbann-load-step $lbann_load_step --vocab-path $VOCAB_PATH  --dropout $DROPOUT --weight-prefix $WEIGHT_PREFIX --num-layers $NUM_LAYERS --n-samples $N_SAMPLES --max-len $MAX_LEN --n-batch $N_BATCH --gen-save $gen_save --test-path $TEST_PATH --ks $KS --n-jobs $N_JOBS --gpu $GPU --batch-size $BATCH_SIZE --metrics $metrics_save --hidden $HIDDEN 


    
done




