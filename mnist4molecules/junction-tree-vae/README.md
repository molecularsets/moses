# Junction Tree Variational Autoencoder for Molecular Graph Generation

Implementation of Junction Tree Variational Autoencoder [https://arxiv.org/abs/1802.04364](https://arxiv.org/abs/1802.04364)

This implementation contains the following:
* `jtnn` folder with model files 
* `download.py` for uploading data (TODO) & pretrained model
* `generate_vocab.py` for generating vocabulary using dataset (details in *Training* block)
* `train.py` for training model (details in *Training* block)
* `sample.py` for sampling new molecules with pretrained models (details in *Testing* block) 
* `reconstruct.py` for molecule reconstruction (TODO) (details in *Testing* block)

## Training

Firstly, you should generate vocabulary for model (reason for it is that generating vocabulary takes ~20 minutes on 200k molecules):
```
mkdir junction-tree-vae/new_data
python junction-tree-vae/generate_vocab.py --train_load <PATH_TO_DATASET> \
 --vocab_save junction-tree-vae/new_data/vocab.pt
 ```

Then you can train model (see config if you want to change parameters):

```
mkdir junction-tree-vae/trained_model
export CUDA_VISIBLE_DEVICES=0
python junction-tree-vae/train.py --train_load <PATH_TO_DATASET> --model_save junction-tree-vae/trained_model/ \
--config_save junction-tree-vae/trained_model/config.pt \ 
--vocab_load junction-tree-vae/new_data/vocab.pt \
--vocab_save junction-tree-vae/new_data/vocab.pt   
```

## Testing
To sample new molecules with pretrained models, run:
```
python junction-tree-vae/sample.py \ 
--model_load junction-tree-vae/trained_model/model-iter4.pt \
--config_load junction-tree-vae/trained_model/model-iter4.pt \
--vocab_load junction-tree-vae/new_data/vocab.pt \
--n_samples 100 --gen_save junction-tree-vae/new_data/generated_smiles.csv
```
This script saves generated SMILES in provided file. 