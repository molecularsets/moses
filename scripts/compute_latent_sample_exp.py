import os
import torch
from tqdm import tqdm
import argparse
import multiprocessing as mp
import pandas as pd
from moses.models_storage import ModelsStorage
from moses.metrics.utils import average_agg_tanimoto, fingerprints, fingerprint
from rdkit import DataStructs, Chem
from scipy.spatial.distance import jaccard
import numpy as np

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

parser.add_argument("--model-config", help="path to model configuration dict")

######################################
# These are things specific to the VAE
######################################

#parser.add_argument("--freeze-embeddings", action="store_true")  # this turns off grad accumulation for embedding layer (see https://github.com/samadejacobs/moses/blob/master/moses/vae/model.py#L22)
#parser.add_argument("--q-cell", default="gru")


parser.add_argument("--seed-molecules", help="points to a file with molecules to use as the reference points in the experiment", required=True)
parser.add_argument("--k-neighbor-samples", help="number of neighbors to draw from the gaussian ball", type=int, required=True)
parser.add_argument("--scale-factor", help="scale factor (std) for gaussian", type=float, required=True)
parser.add_argument("--output", help="path to save output results", required=True)
model_config = parser.parse_args()

moses_config_dict = torch.load(model_config.model_config)


def load_model():
    MODELS = ModelsStorage()
    model_vocab = torch.load(model_config.vocab_path)
    model = MODELS.get_model_class(model_config.model)(model_vocab, moses_config_dict)
    # load the model
    assert os.path.exists(model_config.lbann_weights_dir) is not None

    weights_prefix = f"{model_config.lbann_weights_dir}/{model_config.weight_prefix}"
    model.load_lbann_weights(model_config.lbann_weights_dir, epoch_count=model_config.lbann_load_epoch)

    model.cuda()
    model.eval()

    return model

    
def sample_noise_add_to_vec(latent_vec, scale_factor=model_config.scale_factor):
    noise = torch.normal(mean=0, std=torch.ones(latent_vec.shape)*scale_factor).numpy()
    #print(noise)

    return latent_vec + noise


def main(k=model_config.k_neighbor_samples):
    model = load_model()

    input_smiles_list = pd.read_csv(model_config.seed_molecules, header=None)[0].to_list()

    #import ipdb
    #ipdb.set_trace()
    reference_latent_vec_list, reference_smiles_list = model.encode_smiles(input_smiles_list)


    reference_latent_vec_list = [x.cpu().unsqueeze(0).numpy() for x in reference_latent_vec_list]
    
    
    result_list = []

    for reference_latent_vec, reference_smiles in tqdm(zip(reference_latent_vec_list, reference_smiles_list), desc="sampling neighbors for reference vec and decoding", total=len(reference_latent_vec_list)):

        # TODO: this is just for debugging
        #input_fp = fingerprint(input_smiles, fp_type='morgan')
 
        #reference_mol = Chem.MolFromSmiles(reference_smiles)
          
        neighbor_smiles_list = [model.decode_smiles(sample_noise_add_to_vec(reference_latent_vec))[0]['SMILES'][0] for i in range(k)]  
        
        neighbor_fps = [fingerprint(neighbor_smiles, fp_type='morgan') for neighbor_smiles in neighbor_smiles_list]  #here is a bug in fingerprints funciton that references first_fp before assignment...

        reference_fp = fingerprint(reference_smiles, fp_type='morgan')

        neighbor_tani_list = [jaccard(reference_fp, neighbor_fp) for neighbor_fp in neighbor_fps]
        neighbor_valid_list = [x for x in [Chem.MolFromSmiles(smiles) for smiles in neighbor_smiles_list] if x is not None]
         


        result_list.append({"reference_smiles": reference_smiles, "mean_tani_sim": np.mean(neighbor_tani_list), "min_tani_sim": np.min(neighbor_tani_list), "max_tani_sim": np.max(neighbor_tani_list), "valid_rate": len(neighbor_valid_list)/k })

    pd.DataFrame(result_list).to_csv(model_config.output)
        

if __name__ == "__main__":
    main()

