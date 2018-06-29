import sys
sys.path.insert(0, '..')

import argparse
import pandas as pd
import torch

from mnist4molecules.char_rnn.config import get_parser
from mnist4molecules.char_rnn.model import CharRNN
from utils import add_sample_args, set_seed


def main(config):
    set_seed(config.seed)

    model_vocab = torch.load(config.vocab_load)
    model_config = torch.load(config.config_load)
    model_state = torch.load(config.model_load)

    device = torch.device(config.device)

    model = CharRNN(model_vocab, model_config.hidden, model_config.num_layers, device)
    model.load_state_dict(model_state)
    model = model.to(device=device)

    gen_smiles = model.sample_smiles(config.n_samples, config.max_len)
    gen_smiles = [model_vocab.ids2string(t.tolist()) for t in gen_smiles]

    df = pd.DataFrame(gen_smiles, columns=['SMILES'])
    df.to_csv(config.gen_save, index=False)


if __name__ == '__main__':
    parser = add_sample_args(argparse.ArgumentParser())
    config = parser.parse_known_args()[0]
    main(config)