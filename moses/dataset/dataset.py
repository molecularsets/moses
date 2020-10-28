import os
import numpy as np
import pandas as pd


AVAILABLE_SPLITS = ['train', 'test', 'test_scaffolds']


def get_dataset(split='train'):
    """
    Loads MOSES dataset

    Arguments:
        split (str): split to load. Must be
            one of: 'train', 'test', 'test_scaffolds'

    Returns:
        list with SMILES strings
    """
    if split not in AVAILABLE_SPLITS:
        raise ValueError(
            f"Unknown split {split}. "
            f"Available splits: {AVAILABLE_SPLITS}"
        )
    base_path = os.path.dirname(__file__)
    if split not in AVAILABLE_SPLITS:
        raise ValueError(
            f"Unknown split {split}. "
            f"Available splits: {AVAILABLE_SPLITS}")
    path = os.path.join(base_path, 'data', split+'.csv.gz')
    smiles = pd.read_csv(path, compression='gzip')['SMILES'].values
    return smiles


def get_statistics(split='test'):
    base_path = os.path.dirname(__file__)
    path = os.path.join(base_path, 'data', split+'_stats.npz')
    return np.load(path, allow_pickle=True)['stats'].item()
