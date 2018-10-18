import argparse
import os
import pandas as pd
from urllib import request


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Directory for downloaded dataset')
    parser.add_argument('--dataset_url', type=str, default='',
                        help='URL of dataset')
    parser.add_argument('--no_subset', action='store_true',
                        help='Do not create subsets for training and testing')
    parser.add_argument('--train_size', type=int, default=None,
                        help='Size of training dataset')
    parser.add_argument('--test_size', type=int, default=None,
                        help='Size of testing dataset')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random state')

    return parser


def main(config):
    if not os.path.exists(config.output_dir):
        os.mkdir(config.output_dir)

    dataset_path = os.path.join(config.output_dir, 'dataset.csv')
    request.urlretrieve(config.dataset_url, dataset_path)

    if config.no_subset:
        return

    data = pd.read_csv(dataset_path)

    train_data = data[data['SPLIT'] == 'train']
    test_data = data[data['SPLIT'] == 'test']
    test_scaffolds_data = data[data['SPLIT'] == 'test_scaffolds']

    if config.train_size is not None:
        train_data = train_data.sample(config.train_size, random_state=config.seed)

    if config.test_size is not None:
        test_data = test_data.sample(config.test_size, random_state=config.seed)
        test_scaffolds_data = test_scaffolds_data.sample(config.test_size, random_state=config.seed)

    train_data.to_csv(os.path.join(config.output_dir, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(config.output_dir, 'test.csv'), index=False)
    test_scaffolds_data.to_csv(os.path.join(config.output_dir, 'test_scaffolds.csv'), index=False)


if __name__ == '__main__':
    parser = get_parser()
    config, unknown = parser.parse_known_args()
    if len(unknown) != 0:
        raise ValueError("Unknown argument "+unknown[0])
    main(config)
