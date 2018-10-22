import argparse

import pandas as pd
import torch
import tqdm

from moses.script_utils import add_sample_args, set_seed
from moses.vae.model import VAE


def get_parser():
    return add_sample_args(argparse.ArgumentParser())


def main(config):
    set_seed(config.seed)

    model_config = torch.load(config.config_load)
    model_vocab = torch.load(config.vocab_load)
    model_state = torch.load(config.model_load)

    device = torch.device(config.device)

    # For CUDNN to work properly:
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device.index or 0)

    model = VAE(model_vocab, model_config)
    model.load_state_dict(model_state)
    model = model.to(device)

    gen, n = [], config.n_samples
    T = tqdm.tqdm(range(config.n_samples), desc='Generating mols')
    while n > 0:
        x = model.sample(min(n, config.n_batch), config.max_len)[-1]
        mols = [model_vocab.ids2string(i_x.tolist()) for i_x in x]
        n -= len(mols)
        T.update(len(mols))
        T.refresh()
        gen.extend(mols)

    df = pd.DataFrame(gen, columns=['SMILES'])
    df.to_csv(config.gen_save, index=False)


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
