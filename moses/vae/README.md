# vae

## Links

* https://arxiv.org/pdf/1610.02415.pdf
* https://onlinelibrary.wiley.com/doi/pdf/10.1002/minf.201700123

## Description

1-layer bidirectional GRU as encoder with linears at the end, predicting
latent space of size 128 distribution parameters. 3-layers GRU decoder with dropout of 0.2 and 512 hidden dimensionality. 
Training was with batch size of 128, gradients clipping of 50, KL term weight of 1 and Adam optimizer with learning rate of 3 * 1e-4 for 50 epochs.

## Workflow

Training on `train.csv`, testing on `test.csv`:

```
python moses/scripts/vae/train.py --train_load train.csv --device cuda:0
python moses/scripts/vae/sample.py --n_samples 10000 --device cuda:0
python moses/scripts/metrics/eval.py --ref_path test.csv --gen_path gen.csv
```