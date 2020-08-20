import argparse
import sys
import torch
import rdkit
import pandas as pd
from tqdm import tqdm
import os

from moses.models_storage import ModelsStorage
from moses.script_utils import add_sample_args, set_seed, read_smiles_csv

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

MODELS = ModelsStorage()

def get_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Models sampler script', description='available models')
    for model in MODELS.get_model_names():
        add_sample_args(subparsers.add_parser(model))
    return parser


def main(model, config):
    set_seed(config.seed)
    device = torch.device(config.device)
    # For CUDNN to work properly:
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device.index or 0)

    if(config.lbann_weights_dir):
        assert os.path.exists(config.lbann_weights_dir), ("LBANN inference mode is specified but directory "
                                                       " to load weights does not exist: '{}'".format(config.lbann_weights_dir))

    
    model_config = torch.load(config.config_load)
    trainer = MODELS.get_model_trainer(model)(model_config)
    model_vocab = torch.load(config.vocab_load)
    model_state = torch.load(config.model_load)

    model = MODELS.get_model_class(model)(model_vocab, model_config)  
    if os.path.exists(config.lbann_weights_dir): 
        model.load_lbann_weights(config.lbann_weights_dir,config.lbann_epoch_counts)
    else:
        # assume that a non-LBANN model is being loaded
        model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()

    if(config.save_reconstruction) :
      test_data = read_smiles_csv(config.test_path)
      print("Reconstructing ", len(test_data), " test samples")
      test_loader = trainer.get_dataloader(model, test_data, shuffle=False)
      batch = 0;
      tqdm_data = tqdm(test_loader, desc='Reconstruction (batch #{})'.format(batch+1))
      model.reconstruct(tqdm_data,config.pred_save)
      print("Reconstructed samples of ", config.test_path, " saved to ", config.pred_save)

    samples = []
    n = config.n_samples
    print("Generating Samples")
    with tqdm(total=config.n_samples, desc='Generating samples') as T:
        while n > 0:
            current_samples = model.sample(min(n, config.n_batch), config.max_len)
            samples.extend(current_samples)

            n -= len(current_samples)
            T.update(len(current_samples))

    samples = pd.DataFrame(samples, columns=['SMILES'])
    print("Save generated samples to ", config.gen_save)
    samples.to_csv(config.gen_save, index=False)

if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_args()
    model = sys.argv[1]
    main(model, config)
