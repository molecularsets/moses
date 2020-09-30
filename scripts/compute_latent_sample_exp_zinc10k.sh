#!/usr/bin/bash

python -m ipdb compute_latent_sample_exp.py --model vae --lbann-weights-dir /usr/workspace/atom/lbann/vae/zinc10k/weights/ --lbann-load-epoch 100 --lbann-load-step 1800 --gen-save foo_test --test-path /p/lustre1/jones289/lbann/data/newEnamineFrom2020q1-2/newEnamineFrom2020q1-2_test100kSMILES.csv --vocab-path zinc10Kckpt/vae_vocab.pt --model-config zinc10Kckpt/vae_config.pt --weight-prefix sgd.training --seed-molecules ../data/zinc/test_smiles_only_no_header.csv  --k-neighbor-samples 1000 --scale-factor 0.5 --output zink10k_scale_factor_0.5_results.csv


