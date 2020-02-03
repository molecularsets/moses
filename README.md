# Molecular Sets (MOSES): A benchmarking platform for molecular generation models

[![Build Status](https://travis-ci.com/molecularsets/moses.svg?branch=master)](https://travis-ci.com/molecularsets/moses) [![PyPI version](https://badge.fury.io/py/molsets.svg)](https://badge.fury.io/py/molsets)

Deep generative models such as generative adversarial networks, variational autoencoders, and autoregressive models are rapidly growing in popularity for the discovery of new molecules and materials. In this work, we introduce MOlecular SEtS (MOSES), a benchmarking platform to support research on machine learning for drug discovery. MOSES implements several popular molecular generation models and includes a set of metrics that evaluate the diversity and quality of generated molecules. MOSES is meant to standardize the research on molecular generation and facilitate the sharing and comparison of new models. Additionally, we provide a large-scale comparison of existing state of the art models and elaborate on current challenges for generative models that might prove fertile ground for new research. Our platform and source code are freely available here.

__For more details, please refer to the [paper](https://arxiv.org/abs/1811.12823).__

If you are using MOSES in your research paper, please cite us as
```
@article{polykovskiy2018molecular,
  title={{M}olecular {S}ets ({MOSES}): {A} {B}enchmarking {P}latform for {M}olecular {G}eneration {M}odels},
  author={Polykovskiy, Daniil and Zhebrak, Alexander and Sanchez-Lengeling, Benjamin and Golovanov, Sergey and Tatanov, Oktai and Belyaev, Stanislav and Kurbanov, Rauf and Artamonov, Aleksey and Aladinskiy, Vladimir and Veselov, Mark and Kadurin, Artur and Nikolenko, Sergey and Aspuru-Guzik, Alan and Zhavoronkov, Alex},
  journal={arXiv preprint arXiv:1811.12823},
  year={2018}
}
```

![pipeline](images/pipeline.png)

## Dataset

We propose [a benchmarking dataset](https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv) refined from the ZINC database.

The set is based on the ZINC Clean Leads collection. It contains 4,591,276 molecules in total, filtered by molecular weight in the range from 250 to 350 Daltons, a number of rotatable bonds not greater than 7, and XlogP less than or equal to 3.5. We removed molecules containing charged atoms or atoms besides C, N, S, O, F, Cl, Br, H or cycles longer than 8 atoms. The molecules were filtered via medicinal chemistry filters (MCFs) and PAINS filters.

The dataset contains 1,936,962 molecular structures. For experiments, we split the dataset into a training, test and scaffold test sets containing around 1.6M, 176k, and 176k molecules respectively. The scaffold test set contains unique Bemis-Murcko scaffolds that were not present in the training and test sets. We use this set to assess how well the model can generate previously unobserved scaffolds.

## Models

* [Character-level Recurrent Neural Network (CharRNN)](./moses/char_rnn/README.md)
* [Variational Autoencoder (VAE)](./moses/vae/README.md)
* [Adversarial Autoencoder (AAE)](./moses/aae/README.md)
* [Junction Tree Variational Autoencoder (JTN-VAE)](https://github.com/wengong-jin/icml18-jtnn/tree/master/fast_molvae)
* [Latent Generative Adversarial Network (LatentGAN)](./moses/latentgan/README.md)


## Metrics
Besides standard uniqueness and validity metrics, MOSES provides other metrics to access the overall quality of generated molecules. Fragment similarity (Frag) and Scaffold similarity (Scaff) are cosine distances between vectors of fragment or scaffold frequencies correspondingly of the generated and test sets. Nearest neighbor similarity (SNN) is the average similarity of generated molecules to the nearest molecule from the test set. Internal diversity (IntDiv) is an average pairwise similarity of generated molecules. Fréchet ChemNet Distance (FCD) measures the difference in distributions of last layer activations of ChemNet. Novelty is a fraction of unique valid generated molecules not present in the training set.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th rowspan="2">Model</th>
      <th rowspan="2">Valid (↑)</th>
      <th rowspan="2">Unique@1k (↑)</th>
      <th rowspan="2">Unique@10k (↑)</th>
      <th colspan="2">FCD (↓)</th>
      <th colspan="2">SNN (↑)</th>
      <th colspan="2">Frag (↑)</th>
      <th colspan="2">Scaf (↑)</th>
      <th rowspan="2">IntDiv (↑)</th>
      <th rowspan="2">IntDiv2 (↑)</th>
      <th rowspan="2">Filters (↑)</th>
      <th rowspan="2">Novelty (↑)</th>
    </tr>
    <tr>
      <th>Test</th>
      <th>TestSF</th>
      <th>Test</th>
      <th>TestSF</th>
      <th>Test</th>
      <th>TestSF</th>
      <th>Test</th>
      <th>TestSF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><i>Train</i></td>
      <td><i>1.0</i></td>
      <td><i>1.0</i></td>
      <td><i>1.0</i></td>
      <td><i>0.008</i></td>
      <td><i>0.476</i></td>
      <td><i>0.642</i></td>
      <td><i>0.586</i></td>
      <td><i>1.0</i></td>
      <td><i>0.999</i></td>
      <td><i>0.991</i></td>
      <td><i>0.0</i></td>
      <td><i>0.857</i></td>
      <td><i>0.851</i></td>
      <td><i>1.0</i></td>
      <td><i>1.0</i></td>
    </tr>
    <tr>
      <td>AAE</td>
      <td>0.937±0.034</td>
      <td><b>1.0±0.0</b></td>
      <td>0.997±0.002</td>
      <td>0.556±0.203</td>
      <td>1.057±0.237</td>
      <td>0.608±0.004</td>
      <td>0.568±0.005</td>
      <td>0.991±0.005</td>
      <td>0.99±0.004</td>
      <td>0.902±0.037</td>
      <td>0.079±0.009</td>
      <td>0.856±0.003</td>
      <td><b>0.85±0.003</b></td>
      <td>0.996±0.001</td>
      <td>0.793±0.028</td>
    </tr>
    <tr>
      <td>CharRNN</td>
      <td>0.975±0.026</td>
      <td><b>1.0±0.0</b></td>
      <td><b>0.999±0.0</b></td>
      <td><b>0.073±0.025</b></td>
      <td><b>0.52±0.038</b></td>
      <td>0.601±0.021</td>
      <td>0.565±0.014</td>
      <td><b>1.0±0.0</b></td>
      <td><b>0.998±0.0</b></td>
      <td>0.924±0.006</td>
      <td><b>0.11±0.008</b></td>
      <td>0.856±0.0</td>
      <td><b>0.85±0.0</b></td>
      <td>0.994±0.003</td>
      <td>0.842±0.051</td>
    </tr>
    <tr>
      <td>JTN-VAE</td>
      <td><b>1.0</b></td>
      <td><b>1.0</b></td>
      <td><b>0.999</b></td>
      <td>0.422</td>
      <td>0.996</td>
      <td>0.556</td>
      <td>0.527</td>
      <td>0.996</td>
      <td>0.995</td>
      <td>0.892</td>
      <td>0.1</td>
      <td>0.851</td>
      <td>0.845</td>
      <td>0.978</td>
      <td>0.915</td>
    </tr>
    <tr>
      <td>VAE</td>
      <td>0.977±0.001</td>
      <td><b>1.0±0.0</b></td>
      <td>0.998±0.001</td>
      <td>0.099±0.013</td>
      <td>0.567±0.034</td>
      <td><b>0.626±0.0</b></td>
      <td><b>0.578±0.001</b></td>
      <td>0.999±0.0</td>
      <td><b>0.998±0.0</b></td>
      <td><b>0.939±0.002</b></td>
      <td>0.059±0.01</td>
      <td>0.856±0.0</td>
      <td><b>0.85±0.0</b></td>
      <td><b>0.997±0.0</b></td>
      <td>0.695±0.007</td>
    </tr>
    <tr>
      <td>LatentGAN</td>
      <td>0.897±0.002</td>
      <td><b>1.0±0.0</b></td>
      <td>0.997±0.005</td>
      <td>0.296±0.021</td>
      <td>0.824±0.030</td>
      <td>0.538±0.001</td>
      <td>0.514±0.009</td>
      <td>0.999±0.003</td>
      <td><b>0.998±0.003</b></td>
      <td>0.886±0.006</td>
      <td>0.1±0.015</td>
      <td><b>0.857±0.0</b></td>
      <td><b>0.85±0.0</b></td>
      <td>0.973±0.001</td>
      <td><b>0.949±0.001</b></td>
    </tr>

  </tbody>
</table>

For comparison of molecular properties, we computed the Frèchet distance between distributions of molecules in the generated and test sets. Below, we provide plots for lipophilicity (logP), Synthetic Accessibility (SA), Quantitative Estimation of Drug-likeness (QED), Natural Product-likeness (NP) and molecular weight.

|logP|SA|
|----|--|
|![logP](images/logP.png)|![SA](images/SA.png)|
|NP|QED|
|![NP](images/NP.png)|![QED](images/QED.png)|
|weight|
|![weight](images/weight.png)|

# Installation

### PyPi
The simplest way to install MOSES (models and metrics) is to install [RDKit](https://www.rdkit.org/docs/Install.html): `conda install -yq -c rdkit rdkit` and then install MOSES (`molsets`) from pip (`pip install molsets`). If you want to use LatentGAN, you should also install additional dependencies using `bash install_latentgan_dependencies.sh`.

If you are using Ubuntu, you should also install `sudo apt-get install libxrender1 libxext6` for RDKit.

### Docker

1. Install [docker](https://docs.docker.com/install/) and [nvidia-docker](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)).

2. Pull an existing image (4.1Gb to download) from DockerHub:

```bash
docker pull molecularsets/moses
```

or clone the repository and build it manually:

```bash
git clone https://github.com/molecularsets/moses.git
nvidia-docker image build --tag molecularsets/moses moses/
```

3. Create a container:
```bash
nvidia-docker run -it --name moses --network="host" --shm-size 10G molecularsets/moses
```

4. The dataset and source code are available inside the docker container at /moses:
```bash
docker exec -it molecularsets/moses bash
```

### Manually
Alternatively, install dependencies and MOSES manually.

1. Clone the repository:
```bash
git lfs install
git clone https://github.com/molecularsets/moses.git
```

2. [Install RDKit](https://www.rdkit.org/docs/Install.html) for metrics calculation.

3. Install MOSES:
```bash
python setup.py install
```

4. (Optional) Install dependencies for LatentGAN:
```bash
bash install_latentgan_dependencies.sh
```


# Benchmarking your models

* Install MOSES as described in the previous section.

* Calculate metrics for the trained model:

```bash
python scripts/eval.py --ref_path <reference dataset> --gen_path <generated dataset>
```

* Add both generated samples and metrics to your repository

# Platform usage

### End-to-End launch

You can run pretty much everything with:
```bash
python scripts/run.py
```
This will **split** the dataset, **train** the models, **generate** new molecules, and **calculate** the metrics. Evaluation results will be saved in `metrics.csv`.

You can specify the GPU device index as `cuda:n` (or `cpu` for CPU) and/or model by running:
```bash
python scripts/run.py --device cuda:1 --model aae
```

For more details run `python scripts/run.py --help`.

You can reproduce evaluation of all models with several seeds by running:
```bash
sh scripts/run_all_models.sh
```

### Training

```bash
python scripts/train.py <model name> \
       --train_load <train dataset> \
       --model_save <path to model> \
       --config_save <path to config> \
       --vocab_save <path to vocabulary>
```

To get a list of supported models run `python scripts/train.py --help`.

For more details of certain model run `python scripts/train.py <model name> --help`.

### Generation

```bash
python scripts/sample.py <model name> \
       --model_load <path to model> \
       --vocab_load <path to vocabulary> \
       --config_load <path to config> \
       --n_samples <number of samples> \
       --gen_save <path to generated dataset>
```

To get a list of supported models run `python scripts/sample.py --help`.

For more details of certain model run `python scripts/sample.py <model name> --help`.

### Evaluation

```bash
python scripts/eval.py \
       --ref_path <reference dataset> \
       --gen_path <generated dataset>
```

For more details run `python scripts/eval.py --help`.
