import pandas as pd
from multiprocessing import Pool
import requests
from io import BytesIO
import tqdm
from moses.metrics import mol_passes_filters, compute_scaffold
import argparse
import gzip


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_file', type=str, default='mcf_dataset.csv',
                        help='Path for constructed dataset')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random state')
    parser.add_argument('--url', type=str,
                        default='http://zinc.docking.org/db/bysubset/11/11_p0.smi.gz',
                        help='url to .smi.gz file with smiles')
    parser.add_argument('--n_jobs', type=int,
                        default=1,
                        help='number of processes to use')
    return parser


def process_molecule(mol_row):
    mol_row = mol_row.decode('utf-8')
    smiles, _id = mol_row.split()
    if not mol_passes_filters(smiles):
        return None
    return _id, smiles


def main(config):
    print('[1/5] Downloading from {}'.format(config.url))
    req = requests.get(config.url)
    print('[2/5] Extracting')
    with gzip.open(BytesIO(req.content)) as smi:
        lines = smi.readlines()

    print('[3/5] Filtering SMILES')
    pool = Pool(16)
    dataset = [x for x in tqdm.tqdm(pool.imap(process_molecule, lines),
                                    total=len(lines)) if x is not None]
    dataset = pd.DataFrame(dataset, columns=['ID', 'SMILES'])
    dataset = dataset.drop_duplicates('SMILES')
    dataset['scaffold'] = pool.map(compute_scaffold, dataset['SMILES'].values)
    pool.close()

    print('[4/5] Splitting the dataset')
    scaffolds = pd.value_counts(dataset['scaffold'])
    scaffolds = sorted(scaffolds.items(), key=lambda x: (-x[1], x[0]))
    test_scaffolds = set([x[0] for x in scaffolds[9::10]])
    dataset['split'] = 'train'
    test_scaf_idx = [x in test_scaffolds for x in dataset['scaffold']]
    dataset.loc[test_scaf_idx, 'split'] = 'test_scaffolds'
    test_idx = dataset.loc[dataset['split'] == 'train'].sample(frac=0.1,
                                                               random_state=config.seed).index
    dataset.loc[test_idx, 'split'] = 'test'
    dataset.drop('scaffold', axis=1, inplace=True)
    dataset.to_csv(config.output_file, index=None)
    print('[5/5] Done')


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
