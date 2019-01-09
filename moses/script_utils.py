import argparse
import random
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch

from moses.metrics import get_mol, remove_invalid, \
                          fraction_passes_filters, internal_diversity, \
                          FCDMetric, SNNMetric, FragMetric, ScafMetric, \
                          logP, QED, SA, NP, weight
from moses.utils import mapper


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
    common_arg.add_argument('--save_frequency',
                            type=str, default=20,
                            help='How often to save the model')
    common_arg.add_argument('--log_file',
                            type=str, default='log.txt',
                            help='Where to save the log')
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
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MetricsReward:
    supported_metrics = ['fcd', 'snn', 'fragments', 'scaffolds', 'internal_diversity',
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
        rollout_mols = mapper(self.n_jobs)(get_mol, rollout)
        result = [[0 if m is None else 1] for m in rollout_mols]

        if sum([r[0] for r in result], 0) == 0:
            return result

        rollout = remove_invalid(rollout, canonize=True, n_jobs=self.n_jobs)
        if len(rollout) < 2:
            return result

        if len(self.metrics):
            for metric_name in self.metrics:
                if metric_name == 'fcd':
                    m = FCDMetric(n_jobs=self.n_jobs)(ref, rollout)
                elif metric_name == 'morgan':
                    m = SNN(n_jobs=self.n_jobs)(ref_mols, rollout_mols)
                elif metric_name == 'fragments':
                    m = FragMetric(n_jobs=self.n_jobs)(ref_mols, rollout_mols)
                elif metric_name == 'scaffolds':
                    m = ScafMetric(n_jobs=self.n_jobs)(ref_mols, rollout_mols)
                elif metric_name == 'internal_diversity':
                    m = internal_diversity(rollout_mols, n_jobs=self.n_jobs)
                elif metric_name == 'filters':
                    m = fraction_passes_filters(rollout_mols, n_jobs=self.n_jobs)
                elif metric_name == 'logp':
                    m = FrechetMetric(func=logP,  n_jobs=self.n_jobs)(ref_mols, rollout_mols)
                elif metric_name == 'sa':
                    m = FrechetMetric(func=SA, n_jobs=self.n_jobs)(ref_mols, rollout_mols)
                elif metric_name == 'qed':
                    m = FrechetMetric(func=QED, n_jobs=self.n_jobs)(ref_mols, rollout_mols)
                elif metric_name == 'np':
                    m = FrechetMetric(func=NP, n_jobs=self.n_jobs)(ref_mols, rollout_mols)
                elif metric_name == 'weight':
                    m = FrechetMetric(func=weight, n_jobs=self.n_jobs)(ref_mols, rollout_mols)

                m = MetricsReward._nan2zero(m)
                for i in range(len(rollout)):
                    result[i].append(m)

        return result

    def __call__(self, gen):
        idxs = random.sample(range(len(self.ref)), self.n_ref_subsample)
        ref_subsample = [self.ref[idx] for idx in idxs]
        ref_mols_subsample = [self.ref_mols[idx] for idx in idxs]

        gen_counter = Counter(gen)
        gen_counts = [gen_counter[g] for g in gen]

        n = len(gen) // self.n_rollouts
        rollouts = [gen[i::n] for i in range(n)]

        metrics_values = [self._get_metrics(ref_subsample, ref_mols_subsample, rollout) for rollout in rollouts]
        metrics_values = map(lambda rollout_metrics: [sum(r, 0) / len(r) for r in rollout_metrics], metrics_values)
        reward_values = sum(zip(*metrics_values), ())
        reward_values = [v / c for v, c in zip(reward_values, gen_counts)]

        return reward_values
