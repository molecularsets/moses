import argparse

import pandas as pd
import torch
import tqdm

from moses.char_rnn.model import CharRNN
from moses.script_utils import add_sample_args, set_seed


def main(config):
    set_seed(config.seed)

    model_vocab = torch.load(config.vocab_load)
    model_config = torch.load(config.config_load)
    model_state = torch.load(config.model_load)

    device = torch.device(config.device)

    model = CharRNN(model_vocab, model_config.hidden, model_config.num_layers, model_config.dropout, device)
    model.load_state_dict(model_state)
    model = model.to(device=device)
    model.eval()

    gen_smiles = []

    # TODO: n_samples % batch = 0
    for i in tqdm.tqdm(range(config.n_samples // config.n_batch)):
        smiles_list = model.sample_smiles(config.max_len, config.n_batch)
        for t in smiles_list:
            gen_smiles.append(model_vocab.ids2string([i.item() for i in t]))

    df = pd.DataFrame(gen_smiles, columns=['SMILES'])
    df.to_csv(config.gen_save, index=False)


if __name__ == '__main__':
    parser = add_sample_args(argparse.ArgumentParser())
    config = parser.parse_known_args()[0]
    main(config)
