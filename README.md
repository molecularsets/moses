# Molecular Sets (MOSES): A benchmarking platform for molecular generation models

[![Build Status](https://travis-ci.com/molecularsets/moses.svg?branch=master)](https://travis-ci.com/molecularsets/moses) [![PyPI version](https://badge.fury.io/py/molsets.svg)](https://badge.fury.io/py/molsets)

Deep generative models are rapidly becoming popular for the discovery of new molecules and materials. Such models learn on a large collection of molecular structures and produce novel compounds. In this work, we introduce Molecular Sets (MOSES), a benchmarking platform to support research on machine learning for drug discovery. MOSES implements several popular molecular generation models and provides a set of metrics to evaluate the quality and diversity of generated molecules. With MOSES, we aim to standardize the research on molecular generation and facilitate the sharing and comparison of new models.

__For more details, please refer to the [paper](https://arxiv.org/abs/1811.12823).__

If you are using MOSES in your research paper, please cite us as
```
@article{10.3389/fphar.2020.565644,
  title={{M}olecular {S}ets ({MOSES}): {A} {B}enchmarking {P}latform for {M}olecular {G}eneration {M}odels},
  author={Polykovskiy, Daniil and Zhebrak, Alexander and Sanchez-Lengeling, Benjamin and Golovanov, Sergey and Tatanov, Oktai and Belyaev, Stanislav and Kurbanov, Rauf and Artamonov, Aleksey and Aladinskiy, Vladimir and Veselov, Mark and Kadurin, Artur and Johansson, Simon and  Chen, Hongming and Nikolenko, Sergey and Aspuru-Guzik, Alan and Zhavoronkov, Alex},
  journal={Frontiers in Pharmacology},
  year={2020}
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
      <td><i>0.4755</i></td>
      <td><i>0.6419</i></td>
      <td><i>0.5859</i></td>
      <td><i>1.0</i></td>
      <td><i>0.9986</i></td>
      <td><i>0.9907</i></td>
      <td><i>0.0</i></td>
      <td><i>0.8567</i></td>
      <td><i>0.8508</i></td>
      <td><i>1.0</i></td>
      <td><i>1.0</i></td>
    </tr>
    <tr>
      <td>HMM</td>
      <td>0.076±0.0322</td>
      <td>0.623±0.1224</td>
      <td>0.5671±0.1424</td>
      <td>24.4661±2.5251</td>
      <td>25.4312±2.5599</td>
      <td>0.3876±0.0107</td>
      <td>0.3795±0.0107</td>
      <td>0.5754±0.1224</td>
      <td>0.5681±0.1218</td>
      <td>0.2065±0.0481</td>
      <td>0.049±0.018</td>
      <td>0.8466±0.0403</td>
      <td>0.8104±0.0507</td>
      <td>0.9024±0.0489</td>
      <td><b>0.9994±0.001</b></td>
    </tr>
    <tr>
      <td>NGram</td>
      <td>0.2376±0.0025</td>
      <td>0.974±0.0108</td>
      <td>0.9217±0.0019</td>
      <td>5.5069±0.1027</td>
      <td>6.2306±0.0966</td>
      <td>0.5209±0.001</td>
      <td>0.4997±0.0005</td>
      <td>0.9846±0.0012</td>
      <td>0.9815±0.0012</td>
      <td>0.5302±0.0163</td>
      <td>0.0977±0.0142</td>
      <td><b>0.8738±0.0002</b></td>
      <td>0.8644±0.0002</td>
      <td>0.9582±0.001</td>
      <td>0.9694±0.001</td>
    </tr>
    <tr>
      <td>Combinatorial</td>
      <td><b>1.0±0.0</b></td>
      <td>0.9983±0.0015</td>
      <td>0.9909±0.0009</td>
      <td>4.2375±0.037</td>
      <td>4.5113±0.0274</td>
      <td>0.4514±0.0003</td>
      <td>0.4388±0.0002</td>
      <td>0.9912±0.0004</td>
      <td>0.9904±0.0003</td>
      <td>0.4445±0.0056</td>
      <td>0.0865±0.0027</td>
      <td>0.8732±0.0002</td>
      <td><b>0.8666±0.0002</b></td>
      <td>0.9557±0.0018</td>
      <td>0.9878±0.0008</td>
    </tr>
    <tr>
      <td>CharRNN</td>
      <td>0.9748±0.0264</td>
      <td><b>1.0±0.0</b></td>
      <td>0.9994±0.0003</td>
      <td><b>0.0732±0.0247</b></td>
      <td><b>0.5204±0.0379</b></td>
      <td>0.6015±0.0206</td>
      <td>0.5649±0.0142</td>
      <td><b>0.9998±0.0002</b></td>
      <td>0.9983±0.0003</td>
      <td>0.9242±0.0058</td>
      <td><b>0.1101±0.0081</b></td>
      <td>0.8562±0.0005</td>
      <td>0.8503±0.0005</td>
      <td>0.9943±0.0034</td>
      <td>0.8419±0.0509</td>
    </tr>
    <tr>
      <td>AAE</td>
      <td>0.9368±0.0341</td>
      <td><b>1.0±0.0</b></td>
      <td>0.9973±0.002</td>
      <td>0.5555±0.2033</td>
      <td>1.0572±0.2375</td>
      <td>0.6081±0.0043</td>
      <td>0.5677±0.0045</td>
      <td>0.991±0.0051</td>
      <td>0.9905±0.0039</td>
      <td>0.9022±0.0375</td>
      <td>0.0789±0.009</td>
      <td>0.8557±0.0031</td>
      <td>0.8499±0.003</td>
      <td>0.996±0.0006</td>
      <td>0.7931±0.0285</td>
    </tr>
    <tr>
      <td>VAE</td>
      <td>0.9767±0.0012</td>
      <td><b>1.0±0.0</b></td>
      <td>0.9984±0.0005</td>
      <td>0.099±0.0125</td>
      <td>0.567±0.0338</td>
      <td><b>0.6257±0.0005</b></td>
      <td><b>0.5783±0.0008</b></td>
      <td>0.9994±0.0001</td>
      <td><b>0.9984±0.0003</b></td>
      <td><b>0.9386±0.0021</b></td>
      <td>0.0588±0.0095</td>
      <td>0.8558±0.0004</td>
      <td>0.8498±0.0004</td>
      <td><b>0.997±0.0002</b></td>
      <td>0.6949±0.0069</td>
    </tr>
    <tr>
      <td>JTN-VAE</td>
      <td><b>1.0±0.0</b></td>
      <td><b>1.0±0.0</b></td>
      <td><b>0.9996±0.0003</b></td>
      <td>0.3954±0.0234</td>
      <td>0.9382±0.0531</td>
      <td>0.5477±0.0076</td>
      <td>0.5194±0.007</td>
      <td>0.9965±0.0003</td>
      <td>0.9947±0.0002</td>
      <td>0.8964±0.0039</td>
      <td>0.1009±0.0105</td>
      <td>0.8551±0.0034</td>
      <td>0.8493±0.0035</td>
      <td>0.976±0.0016</td>
      <td>0.9143±0.0058</td>
    </tr>
    <tr>
      <td>LatentGAN</td>
      <td>0.8966±0.0029</td>
      <td><b>1.0±0.0</b></td>
      <td>0.9968±0.0002</td>
      <td>0.2968±0.0087</td>
      <td>0.8281±0.0117</td>
      <td>0.5371±0.0004</td>
      <td>0.5132±0.0002</td>
      <td>0.9986±0.0004</td>
      <td>0.9972±0.0007</td>
      <td>0.8867±0.0009</td>
      <td>0.1072±0.0098</td>
      <td>0.8565±0.0007</td>
      <td>0.8505±0.0006</td>
      <td>0.9735±0.0006</td>
      <td>0.9498±0.0006</td>
    </tr>
  </tbody>
</table>


For comparison of molecular properties, we computed the Wasserstein-1 distance between distributions of molecules in the generated and test sets. Below, we provide plots for lipophilicity (logP), Synthetic Accessibility (SA), Quantitative Estimation of Drug-likeness (QED) and molecular weight.

|logP|SA|
|----|--|
|![logP](images/logP.png)|![SA](images/SA.png)|
|weight|QED|
|![weight](images/weight.png)|![QED](images/QED.png)|

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

* Get `train`, `test` and `test_scaffolds` datasets using the following code:

```python
import moses

train = moses.get_dataset('train')
test = moses.get_dataset('test')
test_scaffolds = moses.get_dataset('test_scaffolds')
```

* You can use a standard torch [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) in your models. We provide a simple `StringDataset` class for convenience:

```python
from torch.utils.data import DataLoader
from moses import CharVocab, StringDataset

train = moses.get_dataset('train')
vocab = CharVocab.from_data(train)
train_dataset = StringDataset(vocab, train)
train_dataloader = DataLoader(
    train_dataset, batch_size=512,
    shuffle=True, collate_fn=train_dataset.default_collate
)

for with_bos, with_eos, lengths in train_dataloader:
    ...
```

* Calculate metrics from your model's samples. We recomend sampling at least `30,000` molecules:

```python
import moses
metrics = moses.get_all_metrics(list_of_generated_smiles)
```

* Add generated samples and metrics to your repository. Run the experiment multiple times to estimate the variance of the metrics.


# Reproducing the baselines

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
