import argparse
import random
import re
import numpy as np
import pandas as pd
import torch

def add_common_arg(parser):
    def torch_device(arg):
        if re.match('^(cuda(:[0-9]+)?|cpu)$', arg) is None:
            raise argparse.ArgumentTypeError('Wrong device format: {}'.format(arg))

        if arg != 'cpu':
            splited_device = arg.split(':')

            if (not torch.cuda.is_available()) or \
                    (len(splited_device) > 1 and int(splited_device[1]) > torch.cuda.device_count()):
                raise argparse.ArgumentTypeError('Wrong device: {} is not available'.format(arg))

        return arg

    # Base
    parser.add_argument('--device',
                        type=torch_device, default='cuda',
                        help='Device to run: "cpu" or "cuda:<device number>"')
    parser.add_argument('--seed',
                        type=int, default=0,
                        help='Seed')

    return parser


def add_train_args(parser):
    # Common
    common_arg = parser.add_argument_group('Common')
    add_common_arg(common_arg)
    common_arg.add_argument('--train_load',
                            type=str, required=True,
                            help='Input data in csv format to train')
    common_arg.add_argument('--val_load', type=str, 
                            help="Input data in csv format to validation")
    common_arg.add_argument('--model_save',
                            type=str, required=True, default='model.pt',
                            help='Where to save the model')
    common_arg.add_argument('--save_frequency',
                            type=int, default=20,
                            help='How often to save the model')
    common_arg.add_argument('--log_file',
                            type=str, required=False,
                            help='Where to save the log')
    common_arg.add_argument('--config_save',
                            type=str, required=True,
                            help='Where to save the config')
    common_arg.add_argument('--vocab_save',
                            type=str,
                            help='Where to save the vocab')
    common_arg.add_argument('--vocab_load',
                            type=str,
                            help='Where to load the vocab; otherwise it will be evaluated')

    return parser


def add_sample_args(parser):
    # Common
    common_arg = parser.add_argument_group('Common')
    add_common_arg(common_arg)
    common_arg.add_argument('--model_load',
                            type=str, required=True,
                            help='Where to load the model')
    common_arg.add_argument('--config_load',
                            type=str, required=True,
                            help='Where to load the config')
    common_arg.add_argument('--vocab_load',
                            type=str, required=True,
                            help='Where to load the vocab')
    common_arg.add_argument('--n_samples',
                            type=int, required=True,
                            help='Number of samples to sample')
    common_arg.add_argument('--gen_save',
                            type=str, required=True,
                            help='Where to save the gen molecules')
    common_arg.add_argument("--n_batch",
                            type=int, default=32,
                            help="Size of batch")
    common_arg.add_argument("--max_len",
                            type=int, default=100,
                            help="Max of length of SMILES")
    common_arg.add_argument('--lbann_weights_dir', type=str, default='',
                        help='Directory for LBANN weights for inference')
    common_arg.add_argument('--lbann_epoch_counts', type=int, default=30,
                        help='LBANN epoch count at which to load trained model')

    return parser


def read_smiles_csv(path, smiles_col='SMILES'):

    # need to check if the specified path even has a SMILES field, if not, just make one
    df_first = pd.read_csv(path, nrows=1)
    if smiles_col in df_first.columns:

        return pd.read_csv(path,
                           usecols=[smiles_col],
                           squeeze=True).astype(str).tolist()
    # if the specified smiles_col is not in the columns of the csv file and there are multiple columns, then it is ambigously defined so error out
    elif len(df_first.columns) > 1:
        raise RuntimeError(f"the provided value for smiles_col, {smiles_col}, is not contained in the header for this csv file, further there are multiple columns to read from, smiles_col is ambiguous.")
    # we'll now assume that if the csv has a single column, then that column must be smiles...this might not be true but that's the user responsibility
    else:
        print(f"{smiles_col} not contained in the csv file, assuming the only column contains the smiles data")
        return pd.read_csv(path, header=None, squeeze=True).astype(str).tolist()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
