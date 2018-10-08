# MOSES: Molecular Sets

TODO: Abstract

## Dataset

TODO: Description of dataset

## Models

* Adversarial Autoencoder (AAE)
* Char-RNN (CRNN)
* Junction Tree Variational Autoencoder (JTVA)
* Objective-Reinforced Generative Adversarial Network (ORGAN)
* Variational Autoencoder (VAE)

## Metrics

| Model    | Valid | Unique@1k | Unique@10k | FCD           || Morgan        || Fragments     || Scaffolds     || LogP          || SA            || QED           || NP            || Weight        || Internal Diversity | Filters |
|          |       |           |            | Test | Test SF | Test | Test SF | Test | Test SF | Test | Test SF | Test | Test SF | Test | Test SF | Test | Test SF | Test | Test SF | Test | Test SF |                    |         |
|:---:     |:---:  |:---:      |:---:       |:---: |:---:    |:---: |:---:    |:---: |:---:    |:---: |:---:    |:---: |:---:    |:---: |:---:    |:---: |:---:    |:---: |:---:    |:---: |:---:    |:---:               |:---:    |
| AAE      |       |           |            |      |         |      |         |      |         |      |         |      |         |      |         |      |         |      |         |      |         |                    |         |
| CRNN     |       |           |            |      |         |      |         |      |         |      |         |      |         |      |         |      |         |      |         |      |         |                    |         |
| JTVA     |       |           |            |      |         |      |         |      |         |      |         |      |         |      |         |      |         |      |         |      |         |                    |         |
| ORGAN    |       |           |            |      |         |      |         |      |         |      |         |      |         |      |         |      |         |      |         |      |         |                    |         |
| VAE      |       |           |            |      |         |      |         |      |         |      |         |      |         |      |         |      |         |      |         |      |         |                    |         |


### Description of models
* AAE
  1-layer Bi-LSTM (380 hidden size) as encoder and 2-layer LSTM (640 hidden size) as decoder, shared embeddings with size 32. Latent size - 640. Discriminator - MLP (2 layer - 640, 256) with ELU activation. Batch size - 128, number of epochs - 25, lr - 1e-3, optimizer - Adam.

* CRNN
  3-layer LSTM with 600 hidden each, and everyone followed by a dropout layer, with a dropout ratio of 0.2, and a softmax layer on top. Training was with batch size of 64, Adam optimizer with learning rate of 1e-3 for 50 epochs.

* JTVA
  Training was with batch size of 40, KL term weight of 0.005 and Adam optimizer with learning rate of 1e-3 for 5 epochs. KL term was taken into consideration starting from second epoch, i.e we trained model as just autoencoder one epoch. Other parameters were taken from original paper: hidden size is 450, latent dimensionality is 56 and depth of graph message passing is 3.
* ORGAN
  TODO: Description of parameters
* VAE
  1-layer bidirectional GRU as encoder with linears at the end, predicting latent space of size 128 distribution parameters. 3-layers GRU decoder with dropout of 0.2 and 512 hidden dimensionality. Training was with batch size of 128, gradients clipping of 50, KL term weight of 1 and Adam optimizer with learning rate of 3 * 1e-4 for 50 epochs.


### Calculation of metrics for all trained models
```
```

## Installation
```
```

## Usage

### Training of model

```
```

### Calculation of metrics for trained model

```
```