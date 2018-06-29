# vae

## workflow

Training on `train.csv`, testing on `test.csv`:

```
python mnist4molecules/vae/scrips/train.py --train_load train.csv
python mnist4molecules/vae/scrips/sample.py --n_samples 10000
python mnist4molecules/metrics/scrips/eval.py --ref_path test.csv --gen_path gen.csv
```