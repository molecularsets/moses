# CharRNN for Generation of Molecules

## Training

Train model (see config if you want to change parameters):

```
mkdir mnist4molecules/char_rnn/trained_model
export CUDA_VISIBLE_DEVICES=0
python mnist4molecules/char_rnn/scripts/train.py \
--train_load <PATH_TO_DATASET> \ 
--model_save mnist4molecules/char_rnn/trained_model/ \
--config_save mnist4molecules/char_rnn/trained_model/config.pt \ 
--vocab_save mnist4molecules/char_rnn/trained_model/vocab.pt   
```

## Testing
To sample new molecules with pretrained model, run:
```
mkdir mnist4molecules/char_rnn/new_data
python mnist4molecules/char_rnn/scripts/sample.py \ 
--model_load mnist4molecules/char_rnn/trained_model/model-iter4.pt \
--config_load mnist4molecules/char_rnn/trained_model/config.pt \
--vocab_load mnist4molecules/char_rnn/trained_model/vocab.pt \
--n_samples 100 --gen_save mnist4molecules/char_rnn/new_data/generated_smiles.csv
```
This script saves generated SMILES at provided file. 