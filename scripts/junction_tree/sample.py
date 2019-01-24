import argparse
import pandas as pd
import rdkit
import torch

from tqdm import tqdm

from moses.junction_tree import JTNNVAE
from moses.script_utils import add_sample_args, set_seed

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


def get_parser():
    return add_sample_args(argparse.ArgumentParser())


def main(config):
    set_seed(config.seed)
    device = torch.device(config.device)
    
    # For CUDNN to work properly:
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device.index or 0)

    model_config = torch.load(config.config_load)
    model_vocab = torch.load(config.vocab_load)
    model_state = torch.load(config.model_load)

    model = JTNNVAE(model_vocab, model_config)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()

    samples = []
    n = config.n_samples
    with tqdm(total=config.n_samples, desc='Generating samples') as T:
        while n > 0:
            current_samples = model.sample(min(n, config.n_batch), config.max_len)
            samples.extend(current_samples)

            n -= len(current_samples)
            T.update(len(current_samples))

    samples = pd.DataFrame(samples, columns=['SMILES'])
    samples.to_csv(config.gen_save, index=False)

    
if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
