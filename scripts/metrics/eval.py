import argparse

import rdkit

from moses.metrics.metrics import get_all_metrics
from moses.script_utils import read_smiles_csv
import numpy as np

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


def main(config, print_metrics=True):
    test = read_smiles_csv(config.test_path)
    test_scaffolds = None
    ptest = None
    ptest_scaffolds = None
    if config.test_scaffolds_path is not None:
        test_scaffolds = read_smiles_csv(config.test_scaffolds_path)
    if config.test_path_precalc is not None:
        ptest = np.load(config.test_path_precalc)['stats'].item()
    if config.test_scaffolds_path_precalc is not None:
        ptest_scaffolds = np.load(config.test_scaffolds_path_precalc)['stats'].item()
    gen = read_smiles_csv(config.gen_path)
    metrics = get_all_metrics(test, gen, k=config.ks, n_jobs=config.n_jobs,
                              gpu=config.device_code, test_scaffolds=test_scaffolds,
                              ptest=ptest, ptest_scaffolds=ptest_scaffolds)
    
    if print_metrics:
        print('Metrics:')
        for name, value in metrics.items():
            print('\t' + name + ' = {}'.format(value))
    else:
        return metrics


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path',
                        type=str, required=True,
                        help='Path to test molecules csv')
    parser.add_argument('--test_scaffolds_path',
                        type=str, required=False,
                        help='Path to scaffold test molecules csv')
    parser.add_argument('--test_path_precalc',
                        type=str, required=False,
                        help='Path to precalculated test molecules npz')
    parser.add_argument('--test_scaffolds_path_precalc',
                        type=str, required=False,
                        help='Path to precalculated scaffold test molecules npz')

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
