

# ORGAN (Objective-Reinforced GenerativeAdversarial Network)

## Links

* [Objective-Reinforced Generative Adversarial Networks (ORGAN) for Sequence Generation Models](https://arxiv.org/abs/1705.10843)
* [Optimizing distributions over molecular space.An Objective-Reinforced GenerativeAdversarial Network for Inverse-designChemistry (ORGANIC)](https://chemrxiv.org/articles/ORGANIC_1_pdf/5309668)

## Workflow

Training on `train.csv`, testing on `test.csv`:

```
python mnist4molecules/scripts/organ/train.py --train_load train.csv --device cuda:0
python mnist4molecules/scripts/organ/sample.py --n_samples 10000 --device cuda:0
python mnist4molecules/scripts/metrics/eval.py --ref_path test.csv --gen_path gen.csv
```
