# MOSES: Molecular Sets

TODO: Abstract

## Dataset

TODO: Description of dataset

## Models

TODO: Check links in models

* [Adversarial Autoencoder (AAE)](./moses/aae/README.md)
* [Char-RNN (CRNN)](./moses/char_rnn/README.md)
* [Junction Tree Variational Autoencoder (JT)](./moses/junction_tree/README.md)
* [Objective-Reinforced Generative Adversarial Network (ORGAN)](./moses/organ/README.md)
* [Variational Autoencoder (VAE)](./moses/vae/README.md)

## Metrics

| Model             | Valid | Unique@1k | Unique@10k | FCD            | Morgan         | Fragments      | Scaffolds      | LogP           | SA             | QED            | NP             | Weight         | Internal Diversity | Filters |
|:---:              |:---:  |:---:      |:---:       |:---: |:---:    |:---: |:---:    |:---: |:---:    |:---: |:---:    |:---: |:---:    |:---: |:---:    |:---: |:---:    |:---: |:---:    |:---: |:---:    |:---:               |:---:    |
| AAE (scaffolds)   |       |           |            |      |         |      |         |      |         |      |         |      |         |      |         |      |         |      |         |      |         |                    |         |
| CRNN (scaffolds)  |       |           |            |      |         |      |         |      |         |      |         |      |         |      |         |      |         |      |         |      |         |                    |         |
| JT (scaffolds)    |       |           |            |      |         |      |         |      |         |      |         |      |         |      |         |      |         |      |         |      |         |                    |         |
| ORGAN (scaffolds) |       |           |            |      |         |      |         |      |         |      |         |      |         |      |         |      |         |      |         |      |         |                    |         |
| VAE (scaffolds)   |       |           |            |      |         |      |         |      |         |      |         |      |         |      |         |      |         |      |         |      |         |                    |         |


### Description of models

TODO: Check this

* **AAE**:
  1-layer Bi-LSTM (380 hidden size) as encoder and 2-layer LSTM (640 hidden size) as decoder, shared embeddings with size 32. Latent size - 640. Discriminator - MLP (2 layer - 640, 256) with ELU activation. Batch size - 128, number of epochs - 25, lr - 1e-3, optimizer - Adam.

* **CRNN**:
  3-layer LSTM with 600 hidden each, and everyone followed by a dropout layer, with a dropout ratio of 0.2, and a softmax layer on top. Training was with batch size of 64, Adam optimizer with learning rate of 1e-3 for 50 epochs.

* **JT**:
  Training was with batch size of 40, KL term weight of 0.005 and Adam optimizer with learning rate of 1e-3 for 5 epochs. KL term was taken into consideration starting from second epoch, i.e we trained model as just autoencoder one epoch. Other parameters were taken from original paper: hidden size is 450, latent dimensionality is 56 and depth of graph message passing is 3.
* **ORGAN**:
  TODO: Description of parameters
* **VAE**:
  1-layer bidirectional GRU as encoder with linears at the end, predicting latent space of size 128 distribution parameters. 3-layers GRU decoder with dropout of 0.2 and 512 hidden dimensionality. Training was with batch size of 128, gradients clipping of 50, KL term weight of 1 and Adam optimizer with learning rate of 3 * 1e-4 for 50 epochs.


### Calculation of metrics for all models

You can calculate all metrics with:
```
cd scripts
python run.py 
```
If necessary, dataset will be downloaded, splited and all models will be trained. As result in current directory will appear `metrics.csv` with values.
For more details use `python run.py --help`.

## Installation
* [Install RDKit](https://www.rdkit.org/docs/Install.html) for metric calculation.
* Install models with `python setup.py install`

## Usage

### Downloading of dataset
You can download dataset (and split it) with:
```
cd scripts
python download_dataset.py --output_dir <directory for dataset>
```
For more details use `python download_dataset.py --help`.

### Training of model
You can train model with:
```
cd scripts/<model name>
python train.py --train_load <path to train dataset> --model_save <path to model> --config_save <path to config> --vocab_save <path to vocabulary>
```
For more details use `python train.py --help`.

### Calculation of metrics for trained model
You can calculate metrics with:
```
cd scripts/<model name>
python sample.py --model_load <path to model> --config_load <path to config> --vocab_load <path to vocabulary> --n_samples <number of smiles> --gen_save <path to generated smiles>
cd ../metrics
python eval.py --ref_path <path to referenced smiles> --gen_path <path to generated smiles>
```
All metrics output to screen.
For more details use `python sample.py --help` and `python eval.py --help`.

You also can use `python run.py --model <model name>` for calculation metrics.