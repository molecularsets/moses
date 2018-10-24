import argparse
import random
import re

import numpy as np
import pandas as pd
import torch

from moses.metrics import mapper, get_mol, fraction_valid, morgan_similarity, remove_invalid, \
                          fragment_similarity, scaffold_similarity, fraction_passes_filters, \
                          fraction_unique, internal_diversity, frechet_distance, \
                          frechet_chemnet_distance, logP, QED, SA, NP, weight


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
    common_arg.add_argument('--model_save',
                            type=str, default='model.pt',
                            help='Where to save the model')
    common_arg.add_argument('--config_save',
                            type=str, default='config.pt',
                            help='Where to save the config')
    common_arg.add_argument('--vocab_save',
                            type=str, default='vocab.pt',
                            help='Where to save the vocab')

    return parser


def add_sample_args(parser):
    # Common
    common_arg = parser.add_argument_group('Common')
    add_common_arg(common_arg)
    common_arg.add_argument('--model_load',
                            type=str, default='model.pt',
                            help='Where to load the model')
    common_arg.add_argument('--config_load',
                            type=str, default='config.pt',
                            help='Where to load the config')
    common_arg.add_argument('--vocab_load',
                            type=str, default='vocab.pt',
                            help='Where to load the vocab')
    common_arg.add_argument('--n_samples',
                            type=int, required=True,
                            help='Number of samples to sample')
    common_arg.add_argument('--gen_save',
                            type=str, default='gen.csv',
                            help='Where to save the gen molecules')
    common_arg.add_argument("--n_batch",
                            type=int, default=32,
                            help="Size of batch")
    common_arg.add_argument("--max_len",
                            type=int, default=100,
                            help="Max of length of SMILES")

    return parser


def read_smiles_csv(path):
    return pd.read_csv(path,
                       usecols=['SMILES'],
                       squeeze=True).astype(str).tolist()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)


class MetricsReward:
    supported_metrics = ['fcd', 'morgan', 'fragments', 'scaffolds', 'internal_diversity',
                         'filters', 'logp', 'sa', 'qed', 'np', 'weight']

    @staticmethod
    def _nan2zero(value):
        if value == np.nan:
            return 0

        return value

    def __init__(self, ref, n_ref_subsample, n_rollouts, n_jobs, metrics=[]):
        assert all([m in MetricsReward.supported_metrics for m in metrics])

        self.ref = remove_invalid(ref, canonize=True, n_jobs=n_jobs)
        self.ref_mols = mapper(n_jobs)(get_mol, self.ref)

        self.n_ref_subsample = n_ref_subsample
        self.n_rollouts = n_rollouts
        self.n_jobs = n_jobs
        self.metrics = metrics

    def _get_metrics(self, ref, ref_mols, rollout):
        result = [fraction_valid(rollout, n_jobs=self.n_jobs)]
        if result[0] == 0:
            return result

        rollout = remove_invalid(rollout, canonize=True, n_jobs=self.n_jobs)
        if len(rollout) < 2:
            return result

        result.append(fraction_unique(rollout, n_jobs=self.n_jobs))

        if len(self.metrics):
            rollout_mols = mapper(self.n_jobs)(get_mol, rollout)
            for metric_name in self.metrics:
                if metric_name == 'fcd':
                    result.append(frechet_chemnet_distance(ref, rollout, n_jobs=self.n_jobs))
                elif metric_name == 'morgan':
                    result.append(morgan_similarity(ref_mols, rollout_mols, n_jobs=self.n_jobs))
                elif metric_name == 'fragments':
                    result.append(fragment_similarity(ref_mols, rollout_mols, n_jobs=self.n_jobs))
                elif metric_name == 'scaffolds':
                    result.append(scaffold_similarity(ref_mols, rollout_mols, n_jobs=self.n_jobs))
                elif metric_name == 'internal_diversity':
                    result.append(internal_diversity(rollout_mols, n_jobs=self.n_jobs))
                elif metric_name == 'filters':
                    result.append(fraction_passes_filters(rollout_mols, n_jobs=self.n_jobs))
                elif metric_name == 'logp':
                    result.append(frechet_distance(ref_mols, rollout_mols, logP, n_jobs=self.n_jobs))
                elif metric_name == 'sa':
                    result.append(frechet_distance(ref_mols, rollout_mols, SA, n_jobs=self.n_jobs))
                elif metric_name == 'qed':
                    result.append(frechet_distance(ref_mols, rollout_mols, QED, n_jobs=self.n_jobs))
                elif metric_name == 'np':
                    result.append(frechet_distance(ref_mols, rollout_mols, NP, n_jobs=self.n_jobs))
                elif metric_name == 'weight':
                    result.append(frechet_distance(ref_mols, rollout_mols, weight, n_jobs=self.n_jobs))

                result[-1] = MetricsReward._nan2zero(result[-1])

        return result

    def __call__(self, gen):
        idxs = random.sample(range(len(self.ref)), self.n_ref_subsample)
        ref_subsample = [self.ref[idx] for idx in idxs]
        ref_mols_subsample = [self.ref_mols[idx] for idx in idxs]

        n = len(gen) // self.n_rollouts
        rollouts = [gen[i::n] for i in range(0, n)]

        metrics_values = [self._get_metrics(ref_subsample, ref_mols_subsample, rollout) for rollout in rollouts]
        metrics_values = map(lambda m: [sum(m, 0) / len(m)] * self.n_rollouts, metrics_values)
        reward_values = sum(metrics_values, [])

        return reward_values
