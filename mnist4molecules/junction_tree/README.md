# Junction Tree Variational Autoencoder for Molecular Graph Generation

Implementation of Junction Tree Variational Autoencoder [https://arxiv.org/abs/1802.04364](https://arxiv.org/abs/1802.04364) 

This implementation contains the following:
* `jtnn` folder with model files (the most files are from [https://github.com/wengong-jin/icml18-jtnn](https://github.com/wengong-jin/icml18-jtnn) which were rewritten to python 3 & pytorch 0.4.0)
* `scripts/generate_vocab.py` for generating vocabulary using dataset (details in *Training* block)
* `scripts/train.py` for training model (details in *Training* block)
* `scripts/sample.py` for sampling new molecules with pretrained models (details in *Testing* block) 
## Training

Firstly, you should generate vocabulary for model (reason for it is that generating vocabulary takes ~20 minutes on 200k molecules):
```
mkdir mnist4molecules/junction_tree/trained_model
python mnist4molecules/junction_tree/scripts/generate_vocab.py \
--train_load <PATH_TO_DATASET> \
--vocab_save mnist4molecules/junction_tree/trained_model/vocab.pt
 ```

Then you can train model (see config if you want to change parameters):

```
export CUDA_VISIBLE_DEVICES=0
python mnist4molecules/junction_tree/scripts/train.py \
--train_load <PATH_TO_DATASET> \ 
--model_save mnist4molecules/junction_tree/trained_model/ \
--config_save mnist4molecules/junction_tree/trained_model/config.pt \ 
--vocab_load mnist4molecules/junction_tree/trained_model/vocab.pt \
--vocab_save mnist4molecules/junction_tree/trained_model/vocab.pt   
```

## Testing
To sample new molecules with pretrained model, run:
```
mkdir mnist4molecules/junction_tree/new_data
python mnist4molecules/junction_tree/scripts/sample.py \ 
--model_load mnist4molecules/junction_tree/trained_model/model-iter4.pt \
--config_load mnist4molecules/junction_tree/trained_model/config.pt \
--vocab_load mnist4molecules/junction_tree/trained_model/vocab.pt \
--n_samples 100 --gen_save mnist4molecules/junction_tree/new_data/generated_smiles.csv
```
This script saves generated SMILES at provided file. 