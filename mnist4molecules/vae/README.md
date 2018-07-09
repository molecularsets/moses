# vae

## workflow

Training on `train.csv`, testing on `test.csv`:

```
python mnist4molecules/scripts/vae/train.py --train_load /data/Insilico/mmnist_dataset/data/mcf_dataset_train_200k.csv --device cuda:1
python mnist4molecules/scripts/vae/sample.py --n_samples 10000 --device cuda:1
python mnist4molecules/scripts/metrics/eval.py --ref_path /data/Insilico/mmnist_dataset/data/mcf_dataset_test_10k.csv --gen_path gen.csv
```

test:
{'valid': 0.48029999999999995, 'unique@1000': 1.0, 'unique@10000': 0.9997917967936706, 'FCD': 1.3358089709867045, 'morgan': 0.3792429497229316, 'fragments': 0.9881470332820432, 'scaffolds': 0.5094579020327228, 'internal_diversity': 0.860313643738011, 'filters': 0.9485738080366438}
scaffolds:
{'valid': 0.48029999999999995, 'unique@1000': 1.0, 'unique@10000': 0.9997917967936706, 'FCD': 1.7168888051074518, 'morgan': 0.37057692845637613, 'fragments': 0.9878008349790495, 'scaffolds': 0.09137685494807068, 'internal_diversity': 0.860313643738011, 'filters': 0.9485738080366438}
