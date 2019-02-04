import argparse

import rdkit
import torch

from moses.script_utils import read_smiles_csv
from moses.junction_tree.trainer import JTreeTrainer

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_load', type=str, required=True, help='Input data in csv format to train')
    parser.add_argument('--vocab_save', type=str, default='vocab.pt', help='Where to save the vocab')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of jobs')

    return parser

def main(config):
    data = read_smiles_csv(config.train_load)

    trainer = JTreeTrainer(config)
    vocab = trainer.get_vocabulary()

    torch.save(vocab, config.vocab_save)

if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
