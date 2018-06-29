import argparse

import pandas as pd

from mnist4molecules.config import get_config
from mnist4molecules.metrics.metrics import get_all_metrics
from mnist4molecules.utils import PandasDataset

parser = argparse.ArgumentParser()
parser.add_argument('--ref_path',
                    type=str, required=True,
                    help='Path to reference molecules csv')
parser.add_argument('--gen_path',
                    type=str, required=True,
                    help='Path to generated molecules csv')
parser.add_argument('--ks',
                    nargs='+', default=[1000, 10000],
                    help='Prefixes to calc uniqueness at')
parser.add_argument('--n_jobs',
                    type=int, default=1,
                    help='Number of processes to run metrics')
parser.add_argument('--device_code',
                    type=int, default=-1,
                    help='Device code to run (-1 for cpu)')
config = get_config(parser)

if __name__ == '__main__':
    ref = PandasDataset(pd.read_csv(config.ref_path, usecols=['SMILES']))
    gen = PandasDataset(pd.read_csv(config.gen_path, usecols=['SMILES']))
    print(get_all_metrics(ref, gen,
                          k=config.ks, n_jobs=config.n_jobs,
                          gpu=config.device_code))
