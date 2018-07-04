# CharRNN for Generation of Molecules

## Training

Train model (see config if you want to change parameters):

```
mkdir mnist4molecules/char_rnn/trained_model
export CUDA_VISIBLE_DEVICES=0
python mnist4molecules/scripts/char_rnn/train.py \
--train_load <PATH_TO_DATASET> \ 
--model_save mnist4molecules/char_rnn/trained_model/ \
--config_save mnist4molecules/char_rnn/trained_model/config.pt \ 
--vocab_save mnist4molecules/char_rnn/trained_model/vocab.pt \
--device cuda:0 
```

## Testing
To sample new molecules with pretrained model, run:
```
mkdir mnist4molecules/char_rnn/new_data
python mnist4molecules/scripts/char_rnn/sample.py \ 
--model_load mnist4molecules/char_rnn/trained_model/model-iter4.pt \
--config_load mnist4molecules/char_rnn/trained_model/config.pt \
--vocab_load mnist4molecules/char_rnn/trained_model/vocab.pt \
--n_samples 100 --n_batch 50 \ 
--gen_save mnist4molecules/char_rnn/new_data/generated_smiles.csv \
--device cuda:0 
```
This script saves generated SMILES at provided file. 

## Result

test:

{'valid': 0.9501, 'unique@1000': 1.0, 'unique@10000': 0.9996842437638144, 'FCD': 0.2589840956739309, 'morgan': 0.4512296988423053, 'fragments': 0.9975333787668187, 'scaffolds': 0.737961623381718, 'internal_diversity': 0.8578564340600587, 'filters': 0.9936848752762867}

test_scaffolds:

{'valid': 0.9501, 'unique@1000': 1.0, 'unique@10000': 0.9996842437638144, 'FCD': 0.46310818786530916, 'morgan': 0.4392426738929352, 'fragments': 0.9973428545629365, 'scaffolds': 0.0781476502204419, 'internal_diversity': 0.8578564340600587, 'filters': 0.9936848752762867}
