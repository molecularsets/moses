# AAE (Adversarial autoencoder)

## Links

* [Adversarial Autoencoders](https://arxiv.org/abs/1511.05644)
* [The cornucopia of meaningful leads: Applying deep adversarial autoencoders for new molecule development in oncology (for molecular fingerprints)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5355231)

## Workflow

Training on `train.csv`, testing on `test.csv`:

```
python moses/scripts/aae/train.py --train_load train.csv --device cuda:0
python moses/scripts/aae/sample.py --n_samples 10000 --device cuda:0
python moses/scripts/metrics/eval.py --ref_path test.csv --gen_path gen.csv
```
