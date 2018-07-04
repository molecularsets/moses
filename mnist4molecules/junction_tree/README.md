# Junction Tree Variational Autoencoder for Molecular Graph Generation

Implementation of Junction Tree Variational Autoencoder [https://arxiv.org/abs/1802.04364](https://arxiv.org/abs/1802.04364) 

This implementation contains the model files from [https://github.com/wengong-jin/icml18-jtnn](https://github.com/wengong-jin/icml18-jtnn) which were rewritten to python 3 & pytorch 0.4.0)

* `scripts/junction_tree/generate_vocab.py` for generating vocabulary using dataset (details in *Training* block)
* `scripts/junction_tree/train.py` for training model (details in *Training* block)
* `scripts/junction_tree/sample.py` for sampling new molecules with pretrained models (details in *Testing* block) 
## Training

Firstly, you should generate vocabulary for model (reason for it is that generating vocabulary takes ~20 minutes on 200k molecules):
```
mkdir mnist4molecules/junction_tree/trained_model
python mnist4molecules/scripts/junction_tree/generate_vocab.py \
--train_load <PATH_TO_DATASET> \
--vocab_save mnist4molecules/junction_tree/trained_model/vocab.pt
 ```

Then you can train model (see config if you want to change parameters):

```
export CUDA_VISIBLE_DEVICES=0
python mnist4molecules/scripts/junction_tree/train.py \
--train_load <PATH_TO_DATASET> \ 
--model_save mnist4molecules/junction_tree/trained_model/ \
--config_save mnist4molecules/junction_tree/trained_model/config.pt \ 
--vocab_load mnist4molecules/junction_tree/trained_model/vocab.pt \
--vocab_save mnist4molecules/junction_tree/trained_model/vocab.pt \
--device cuda:0  
```

## Testing
To sample new molecules with pretrained model, run:
```
mkdir mnist4molecules/junction_tree/new_data
python mnist4molecules/scripts/junction_tree/sample.py \ 
--model_load mnist4molecules/junction_tree/trained_model/model-iter4.pt \
--config_load mnist4molecules/junction_tree/trained_model/config.pt \
--vocab_load mnist4molecules/junction_tree/trained_model/vocab.pt \
--n_samples 100 --gen_save mnist4molecules/junction_tree/new_data/generated_smiles.csv \
--device cuda:0
```
This script saves generated SMILES at provided file. 

## Result

test:

{'valid': 1.0, 'unique@1000': 1.0, 'unique@10000': 0.9975, 'FCD': 3.0479025356561777, 'morgan': 0.38939896399378776, 'fragments': 0.9521836003585373, 'scaffolds': 0.3051769573003378, 'internal_diversity': 0.8484836989051874, 'filters': 0.9521}

test_scaffolds:

{'valid': 1.0, 'unique@1000': 1.0, 'unique@10000': 0.9975, 'FCD': 2.8890306938310033, 'morgan': 0.38564662508144976, 'fragments': 0.9514141227789296, 'scaffolds': 0.09766374692511892, 'internal_diversity': 0.8484836989051874, 'filters': 0.9521}