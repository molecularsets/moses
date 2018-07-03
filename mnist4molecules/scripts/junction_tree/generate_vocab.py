import argparse

import rdkit
import torch
import tqdm

from mnist4molecules.junction_tree.datautils import JTreeVocab
from mnist4molecules.junction_tree.jtnn.mol_tree import MolTree
from mnist4molecules.script_utils import read_smiles_csv

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


def main(config):
    data = read_smiles_csv(config.train_load)

    clusters = set()

    for smiles in tqdm.tqdm(data):
        mol = MolTree(smiles)
        for c in mol.nodes:
            clusters.add(c.smiles)

    vocab = JTreeVocab(list(clusters))
    torch.save(vocab, config.vocab_save)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_load", type=str, required=True, help='Input data in csv format to train')
    parser.add_argument('--vocab_save', type=str, default='vocab.pt', help='Where to save the vocab')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
