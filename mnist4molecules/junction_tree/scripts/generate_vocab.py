import argparse

import pandas as pd
import rdkit
import torch
import tqdm
from attrdict import AttrDict

from mnist4molecules.junction_tree.datautils import JTreeVocab
from mnist4molecules.junction_tree.jtnn.mol_tree import MolTree
from mnist4molecules.utils import PandasDataset

if __name__ == '__main__':

    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_load", type=str, required=True, help='Input data in csv format to train')
    parser.add_argument('--vocab_save', type=str, default='vocab.pt', help='Where to save the vocab')

    config = AttrDict(parser.parse_known_args()[0].__dict__)

    data = pd.read_csv(config.train_load, usecols=['SMILES'], nrows=200)
    dataset = PandasDataset(data)

    clusters = set()

    for smiles in tqdm.tqdm(dataset):
        mol = MolTree(smiles)
        for c in mol.nodes:
            clusters.add(c.smiles)

    vocab = JTreeVocab(list(clusters))
    torch.save(vocab, config.vocab_save)
