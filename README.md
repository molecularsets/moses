# Molecular Sets (MOSES): A benchmarking platform for molecular generation models

Deep generative models such as generative adversarial networks, variational autoencoders, and autoregressive models are rapidly growing in popularity for the discovery of new molecules and materials. In this work, we introduce MOlecular SEtS (MOSES), a benchmarking platform to support research on machine learning for drug discovery. MOSES implements several popular molecular generation models and includes a set of metrics that evaluate the diversity and quality of generated molecules. MOSES is meant to standardize the research on molecular generation and facilitate the sharing and comparison of new models. Additionally, we provide a large-scale comparison of existing state of the art models and elaborate on current challenges for generative models that might prove fertile ground for new research. Our platform and source code are freely available here.

__For more details, please refer to the [paper](https://arxiv.org/abs/1811.12823).__

![pipeline](images/pipeline.png)

## Dataset

We propose [a benchmarking dataset](https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset.csv) refined from the ZINC database.

The set is based on the ZINC Clean Leads collection. It contains 4,591,276 molecules in total, filtered by molecular weight in the range from 250 to 350 Daltons, a number of rotatable bonds not greater than 7, and XlogP less than or equal to 3.5. We removed molecules containing charged atoms or atoms besides C, N, S, O, F, Cl, Br, H or cycles longer than 8 atoms. The molecules were filtered via medicinal chemistry filters (MCFs) and PAINS filters.

The dataset contains 1,936,962 molecular structures. For experiments, we also provide a training, test and scaffold test sets containing 250k, 10k, and 10k molecules respectively. The scaffold test set contains unique Bemis-Murcko scaffolds that were not present in the training and test sets. We use this set to assess how well the model can generate previously unobserved scaffolds.

## Models

* [Character-level Recurrent Neural Network (CharRNN)](./moses/char_rnn/README.md)
* [Variational Autoencoder (VAE)](./moses/vae/README.md)
* [Adversarial Autoencoder (AAE)](./moses/aae/README.md)
* [Objective-Reinforced Generative Adversarial Network (ORGAN)](./moses/organ/README.md)
* [Junction Tree Variational Autoencoder (JTN-VAE)](./moses/junction_tree/README.md)

## Metrics
Besides standard uniqueness and validity metrics, MOSES provides other metrics to access the overall quality of generated molecules. Fragment similarity (Frag) and Scaffold similarity (Scaff) are cosine distances between vectors of fragment or scaffold frequencies correspondingly of the generated and test sets. Nearest neighbor similarity (SNN) is the average similarity of generated molecules to the nearest molecule from the test set. Internal diversity (IntDiv) is an average pairwise similarity of generated molecules. Fréchet ChemNet Distance (FCD) measures the difference in distributions of last layer activations of ChemNet.

<table border="1" class="dataframe">
    <thead>
        <tr style="text-align: right;">
            <th rowspan="2">Model</th>
            <th rowspan="2">Valid (↑)</th>
            <th rowspan="2">Unique@1k (↑)</th>
            <th rowspan="2">Unique@10k (↑)</th>
            <th colspan="2">FCD (↓)</th>
            <th colspan="2">SNN (↓)</th>
            <th colspan="2">Frag (↑)</th>
            <th colspan="2">Scaff (↑)</th>
            <th rowspan="2">IntDiv (↑)</th>
            <th rowspan="2">Filters (↑)</th>
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
            <th>CharRNN</th>
            <td>0.9598</td>
            <td><b>1.0000</b></td>
            <td>0.9993</td>
            <td>0.3233</td>
            <td>0.8355</td>
            <td>0.4606</td>
            <td>0.4492</td>
            <td>0.9977</td>
            <td>0.9962</td>
            <td>0.7964</td>
            <td>0.1281</td>
            <td><b>0.8561</b></td>
            <td>0.9920</td>
        </tr>
        <tr>
            <th>VAE</th>
            <td>0.9528</td>
            <td><b>1.0000</b></td>
            <td>0.9992</td>
            <td><b>0.2540</b></td>
            <td><b>0.6959</b></td>
            <td>0.4684</td>
            <td>0.4547</td>
            <td><b>0.9978</b></td>
            <td><b>0.9963</b></td>
            <td><b>0.8277</b></td>
            <td>0.0925</td>
            <td>0.8548</td>
            <td>0.9925</td>
        </tr>
        <tr>
            <th>AAE</th>
            <td>0.9341</td>
            <td><b>1.0000</b></td>
            <td><b>1.0000</b></td>
            <td>1.3511</td>
            <td>1.8587</td>
            <td>0.4191</td>
            <td>0.4113</td>
            <td>0.9865</td>
            <td>0.9852</td>
            <td>0.6637</td>
            <td><b>0.1538</b></td>
            <td>0.8531</td>
            <td>0.9759</td>
        </tr>
        <tr>
            <th>ORGAN</th>
            <td>0.8731</td>
            <td>0.9910</td>
            <td>0.9260</td>
            <td>1.5748</td>
            <td>2.4306</td>
            <td>0.4745</td>
            <td>0.4593</td>
            <td>0.9897</td>
            <td>0.9883</td>
            <td>0.7843</td>
            <td>0.0632</td>
            <td>0.8526</td>
            <td><b>0.9934</b></td>
        </tr>
        <tr>
            <th>JTN-VAE</th>
            <td><b>1.0000</b></td>
            <td>0.9980</td>
            <td>0.9972</td>
            <td>4.3769</td>
            <td>4.6299</td>
            <td><b>0.3909</b></td>
            <td><b>0.3902</b></td>
            <td>0.9679</td>
            <td>0.9699</td>
            <td>0.3868</td>
            <td>0.1163</td>
            <td>0.8495</td>
            <td>0.9566</td>
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

### Docker

1. Install [docker](https://docs.docker.com/install/) and [nvidia-docker](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)).

2. Pull an existing image from DockerHub:

```
docker pull molecularsets/moses
```

or clone the repository and build it manually:


```
git lfs install
git clone https://github.com/molecularsets/moses.git
nvidia-docker image build --tag molecularsets/moses moses/
```

3. Create a container:
```
nvidia-docker run -it moses --network="host" --shm-size 1G molecularsets/moses
```

4. The dataset and source code is available inside the docker container:
```
docker exec -it molecularsets/moses bash
```

### Manually
Alternatively, install dependencies and MOSES manually.

1. Clone the repository:
```
git lfs install
git clone https://github.com/molecularsets/moses.git
```

2. [Install RDKit](https://www.rdkit.org/docs/Install.html) for metrics calculation.

3. Install MOSES:
```
python setup.py install
```


# Benchmarking your models

* Install MOSES as described in the previous section.

* Calculate metrics for the trained model:

```
python scripts/metrics/eval.py --ref_path <reference dataset> --gen_path <generated dataset>
```

# Platform usage

### Training

```
python scripts/<model name>/train.py \
--train_load <train dataset> \
--model_save <path to model> \
--config_save <path to config> \
--vocab_save <path to vocabulary>
```
For more details run `python scripts/<model name>/train.py --help`.

### Generation

```
python scripts/<model name>/sample.py \
--model_load <path to model> \
--vocab_load <path to vocabulary> \
--config_load <path to config> \
--n_samples <number of samples> \
--gen_save <path to generated dataset>
```

For more details run `python scripts/<model name>/sample.py --help`

### Evaluation

```
python scripts/metrics/eval.py \
--ref_path <reference dataset> \
--gen_path <generated dataset>
```

For more details run `python scripts/metrics/eval.py --help`.


### End-to-End launch

You can run pretty much everything with:
```
python scripts/run.py
```
This will **download** the dataset, **train** the models, **generate** new molecules, and **calculate** the metrics. Evaluation results will be saved in `metrics.csv`.

You can specify the GPU index (-1 for CPU) and/or model by running:
```
python scripts/run.py --gpu 1 --model aae
```

For more details run `python scripts/run.py --help`.
