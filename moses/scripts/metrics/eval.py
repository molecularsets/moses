import argparse

import rdkit

from moses.metrics.metrics import get_all_metrics
from moses.script_utils import read_smiles_csv

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


def main(config):
    ref = read_smiles_csv(config.ref_path)
    gen = read_smiles_csv(config.gen_path)
    metrics = get_all_metrics(ref, gen, k=config.ks, n_jobs=config.n_jobs,
                              gpu=config.device_code)

    print('Metrics:')
    for name, value in metrics.items():
        print('\t' + name + ' = {}'.format(value))


def get_parser():
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

    return parser


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
