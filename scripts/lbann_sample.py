import os
import argparse
import rdkit
import torch
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from run import load_module
from moses.models_storage import ModelsStorage
from moses.script_utils import add_sample_args, set_seed, read_smiles_csv
from moses.metrics.metrics import get_all_metrics

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--lbann-weights-dir", required=True)
parser.add_argument("--lbann-load-epoch", type=int, required=True)
parser.add_argument("--lbann-load-step", type=int, required=True)
parser.add_argument(
    "--vocab-path", type=str, default="", help="path to experiment vocabulary"
)
parser.add_argument("--num-layers", type=int)
parser.add_argument("--dropout", type=float)
parser.add_argument("--weight-prefix")
parser.add_argument("--n-samples", type=int, default=100)
parser.add_argument("--max-len", type=int, default=100)
parser.add_argument("--n-batch", type=int, default=10)
parser.add_argument("--gen-save", required=True)

parser.add_argument("--test-path", required=True)
parser.add_argument("--test-scaffolds-path")
parser.add_argument("--ptest-path")
parser.add_argument("--ptest-scaffolds-path")

parser.add_argument("--ks", type=int, nargs="+", help="list with values for unique@k. Will calculate number of unique molecules in the first k molecules.")
parser.add_argument("--n-jobs", type=int, default=mp.cpu_count()-1) 
parser.add_argument("--gpu", type=int, help=" index of GPU for FCD metric and internal diversity, -1 means use CPU")
parser.add_argument("--batch-size", type=int, help="batch size for FCD metric")
parser.add_argument("--hidden", type=int)
parser.add_argument("--metrics", help="output path to store metrics")

model_config = parser.parse_args()


def eval_metrics(eval_config, print_metrics=True):

    # need to detect if file has the header or not
    test = read_smiles_csv(model_config.test_path)
    test_scaffolds = None
    ptest = None
    ptest_scaffolds = None
    if model_config.test_scaffolds_path is not None:
        test_scaffolds = read_smiles_csv(model_config.test_scaffolds_path)
    if model_config.ptest_path is not None:
        if not os.path.exists(model_config.ptest_path):
            warnings.warn(f"{model_config.ptest_path} does not exist")
            ptest = None
        else:
            ptest = np.load(model_config.ptest_path)["stats"].item()
    if model_config.ptest_scaffolds_path is not None:
        if not os.path.exists(model_config.ptest_scaffolds_path):
            warnings.warn(f"{model_config.ptest_scaffolds_path} does not exist")
            ptest_scaffolds = None
        else:
            ptest_scaffolds = np.load(model_config.ptest_scaffolds_path)["stats"].item()
    gen = read_smiles_csv(model_config.gen_save)
    metrics = get_all_metrics(
        test,
        gen,
        k=model_config.ks,
        n_jobs=model_config.n_jobs,
        gpu=model_config.gpu,
        test_scaffolds=test_scaffolds,
        ptest=ptest,
        ptest_scaffolds=ptest_scaffolds,
    )

    if print_metrics:
        print("Metrics:")
        for name, value in metrics.items():
            print("\t" + name + " = {}".format(value))
        return metrics
    else:
        return metrics



def sample():
    MODELS = ModelsStorage()
    model_vocab = torch.load(model_config.vocab_path)
    model = MODELS.get_model_class(model_config.model)(model_vocab, model_config)
    # load the model
    assert os.path.exists(model_config.lbann_weights_dir) is not None

    weights_prefix = f"{model_config.lbann_weights_dir}/{model_config.weight_prefix}.epoch.{model_config.lbann_load_epoch}.step.{model_config.lbann_load_step}"
    model.load_lbann_weights(
        weights_prefix,
    )


    # here we should try to wrap model in a dataparallel layer or something?
    model.cuda()
    model.eval()

    samples = []
    n = model_config.n_samples
    print("Generating Samples")
    with tqdm(total=model_config.n_samples, desc="Generating samples") as T:
        while n > 0:
            current_samples = model.sample(
                min(n, model_config.n_batch), model_config.max_len
            )
            samples.extend(current_samples)

            n -= len(current_samples)
            T.update(len(current_samples))

    samples = pd.DataFrame(samples, columns=["SMILES"])
    print("Save generated samples to ", model_config.gen_save)
    samples.to_csv(model_config.gen_save, index=False)
    return samples

def compute_metrics():
    metrics = []
    model_metrics = eval_metrics(model_config)
    #model_metrics.update({"model": model})
    metrics.append(model_metrics)

    table = pd.DataFrame(metrics)
    print("Saving computed metrics to ", model_config.metrics)
    table.to_csv(model_config.metrics, index=False)


def compute_reconstruction(model, test):
    pass 


if __name__ == "__main__":
    sample()
    compute_metrics()


