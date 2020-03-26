import random
from collections import Counter
import numpy as np

from moses.metrics import remove_invalid, \
                          fraction_passes_filters, internal_diversity, \
                          FCDMetric, SNNMetric, FragMetric, ScafMetric, \
                          WassersteinMetric, logP, QED, SA, weight
from moses.utils import mapper, get_mol


class MetricsReward:
    supported_metrics = ['fcd', 'snn', 'fragments', 'scaffolds',
                         'internal_diversity', 'filters',
                         'logp', 'sa', 'qed', 'weight']

    @staticmethod
    def _nan2zero(value):
        if value == np.nan:
            return 0

        return value

    def __init__(self, n_ref_subsample, n_rollouts, n_jobs, metrics=[]):
        assert all([m in MetricsReward.supported_metrics for m in metrics])

        self.n_ref_subsample = n_ref_subsample
        self.n_rollouts = n_rollouts
        # TODO: profile this. Pool works too slow.
        n_jobs = n_jobs if False else 1
        self.n_jobs = n_jobs
        self.metrics = metrics

    def get_reference_data(self, data):
        ref_smiles = remove_invalid(data, canonize=True, n_jobs=self.n_jobs)
        ref_mols = mapper(self.n_jobs)(get_mol, ref_smiles)
        return ref_smiles, ref_mols

    def _get_metrics(self, ref, ref_mols, rollout):
        rollout_mols = mapper(self.n_jobs)(get_mol, rollout)
        result = [[0 if m is None else 1] for m in rollout_mols]

        if sum([r[0] for r in result], 0) == 0:
            return result

        rollout = remove_invalid(rollout, canonize=True, n_jobs=self.n_jobs)
        rollout_mols = mapper(self.n_jobs)(get_mol, rollout)
        if len(rollout) < 2:
            return result

        if len(self.metrics):
            for metric_name in self.metrics:
                if metric_name == 'fcd':
                    m = FCDMetric(n_jobs=self.n_jobs)(ref, rollout)
                elif metric_name == 'morgan':
                    m = SNNMetric(n_jobs=self.n_jobs)(ref_mols, rollout_mols)
                elif metric_name == 'fragments':
                    m = FragMetric(n_jobs=self.n_jobs)(ref_mols, rollout_mols)
                elif metric_name == 'scaffolds':
                    m = ScafMetric(n_jobs=self.n_jobs)(ref_mols, rollout_mols)
                elif metric_name == 'internal_diversity':
                    m = internal_diversity(rollout_mols, n_jobs=self.n_jobs)
                elif metric_name == 'filters':
                    m = fraction_passes_filters(
                        rollout_mols, n_jobs=self.n_jobs
                    )
                elif metric_name == 'logp':
                    m = -WassersteinMetric(func=logP, n_jobs=self.n_jobs)(
                        ref_mols, rollout_mols
                    )
                elif metric_name == 'sa':
                    m = -WassersteinMetric(func=SA, n_jobs=self.n_jobs)(
                        ref_mols, rollout_mols
                    )
                elif metric_name == 'qed':
                    m = -WassersteinMetric(func=QED, n_jobs=self.n_jobs)(
                        ref_mols, rollout_mols
                    )
                elif metric_name == 'weight':
                    m = -WassersteinMetric(func=weight, n_jobs=self.n_jobs)(
                        ref_mols, rollout_mols
                    )

                m = MetricsReward._nan2zero(m)
                for i in range(len(rollout)):
                    result[i].append(m)

        return result

    def __call__(self, gen, ref, ref_mols):

        idxs = random.sample(range(len(ref)), self.n_ref_subsample)
        ref_subsample = [ref[idx] for idx in idxs]
        ref_mols_subsample = [ref_mols[idx] for idx in idxs]

        gen_counter = Counter(gen)
        gen_counts = [gen_counter[g] for g in gen]

        n = len(gen) // self.n_rollouts
        rollouts = [gen[i::n] for i in range(n)]

        metrics_values = [self._get_metrics(
            ref_subsample, ref_mols_subsample, rollout
        ) for rollout in rollouts]
        metrics_values = map(
            lambda rm: [
                sum(r, 0) / len(r)
                for r in rm
            ], metrics_values)
        reward_values = sum(zip(*metrics_values), ())
        reward_values = [v / c for v, c in zip(reward_values, gen_counts)]

        return reward_values
