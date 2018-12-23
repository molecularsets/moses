import warnings

import numpy as np
from scipy.spatial.distance import cosine

from .utils import compute_fragments, average_agg_tanimoto, \
    compute_scaffolds, fingerprints, \
    get_mol, canonic_smiles, mol_passes_filters, \
    logP, QED, SA, NP, weight
from moses.utils import mapper
from .utils_fcd import get_predictions, calculate_frechet_distance
from multiprocessing import Pool
from moses.utils import disable_rdkit_log, enable_rdkit_log


def get_all_metrics(test, gen, k=[1000, 10000], n_jobs=1, gpu=-1,
                    batch_size=512, test_scaffolds=None,
                    ptest=None, ptest_scaffolds=None):
    '''
    Computes all available metrics between test (scaffold test) and generated sets of SMILES.
    Parameters:
        test: list of test SMILES
        gen: list of generated SMILES
        k: list with values for unique@k.
            Will calculate number of unique molecules in the first k molecules.
        n_jobs: number of workers for parallel processing
        gpu: index of GPU for FCD metric
        batch_size: batch size for FCD metric
        test_scaffolds: list of scaffold test SMILES
            Will compute only on the general test set if not specified
        ptest: dict with precalculated statistics of the test set
        ptest_scaffolds: dict with precalculated statistics of the scaffold test set
        
    
    Available metrics:
        * %valid
        * %unique@k
        * Frechet ChemNet Distance (FCD)
        * Fragment similarity (Frag)
        * Scaffold similarity (Scaf)
        * Similarity to nearest neighbour (SNN)
        * Internal diversity (IntDiv)
        * %passes filters (Filters)
        * Distribution difference for logP, SA, QED, NP, weight
    '''
    disable_rdkit_log()
    metrics = {}
    if n_jobs != 1:
        pool = Pool(n_jobs)
    else:
        pool = 1
    metrics['valid'] = fraction_valid(gen, n_jobs=n_jobs)
    gen = remove_invalid(gen, canonize=True)
    if not isinstance(k, (list, tuple)):
        k = [k]
    for _k in k:
        metrics['unique@{}'.format(_k)] = fraction_unique(gen, _k, pool)

    if ptest is None:
        ptest = compute_intermediate_statistics(test, n_jobs=n_jobs, gpu=gpu, batch_size=batch_size)
    if test_scaffolds is not None and ptest_scaffolds is None:
        ptest_scaffolds = compute_intermediate_statistics(test_scaffolds, n_jobs=n_jobs,
                                                                 gpu=gpu, batch_size=batch_size)
    mols = mapper(pool)(get_mol, gen)
    kwargs = {'n_jobs': pool, 'gpu': gpu, 'batch_size': batch_size}
    metrics['FCD/Test'] = FCDMetric(**kwargs)(gen=gen, ptest=ptest['FCD'])
    metrics['SNN/Test'] = SNNMetric(**kwargs)(gen=mols, ptest=ptest['SNN'])
    metrics['Frag/Test'] = FragMetric(**kwargs)(gen=mols, ptest=ptest['Frag'])
    metrics['Scaf/Test'] = ScafMetric(**kwargs)(gen=mols, ptest=ptest['Scaf'])
    if ptest_scaffolds is not None:
        metrics['FCD/TestSF'] = FCDMetric(**kwargs)(gen=gen, ptest=ptest_scaffolds['FCD'])
        metrics['SNN/TestSF'] = SNNMetric(**kwargs)(gen=mols, ptest=ptest_scaffolds['SNN'])
        metrics['Frag/TestSF'] = FragMetric(**kwargs)(gen=mols, ptest=ptest_scaffolds['Frag'])
        metrics['Scaf/TestSF'] = ScafMetric(**kwargs)(gen=mols, ptest=ptest_scaffolds['Scaf'])

    metrics['IntDiv'] = internal_diversity(mols, pool)
    metrics['IntDiv2'] = internal_diversity(mols, pool, p=2)
    metrics['Filters'] = fraction_passes_filters(mols, pool)

    # Properties
    for name, func in [('logP', logP), ('SA', SA),
                       ('QED', QED), ('NP', NP),
                       ('weight', weight)]:
        metrics[name] = FrechetMetric(func, **kwargs)(gen=mols, ptest=ptest[name])
    enable_rdkit_log()
    if n_jobs != 1:
        pool.terminate()
    return metrics



def compute_intermediate_statistics(smiles, n_jobs=1, gpu=-1, batch_size=512):
    '''
    The function precomputes statistics such as mean and variance for FCD, etc.
    It is useful to compute the statistics for test and scaffold test sets to
        speedup metrics calculation.
    '''

    if n_jobs != 1:
        pool = Pool(n_jobs)
    else:
        pool = 1
    statistics = {}
    mols = mapper(pool)(get_mol, smiles)
    kwargs = {'n_jobs': n_jobs, 'gpu': gpu, 'batch_size': batch_size}
    statistics['FCD'] = FCDMetric(**kwargs).precalc(smiles)
    statistics['SNN'] = SNNMetric(**kwargs).precalc(mols)
    statistics['Frag'] = FragMetric(**kwargs).precalc(mols)
    statistics['Scaf'] = ScafMetric(**kwargs).precalc(mols)
    for name, func in [('logP', logP), ('SA', SA),
                       ('QED', QED), ('NP', NP),
                       ('weight', weight)]:
        statistics[name] = FrechetMetric(func, **kwargs).precalc(mols)
    if n_jobs != 1:
        pool.terminate()
    return statistics


def fraction_passes_filters(gen, n_jobs=1):
    '''
    Computes the fraction of molecules that pass filters:
    * MCF
    * PAINS
    * Only allowed atoms ('C','N','S','O','F','Cl','Br','H')
    * No charges
    '''
    passes = mapper(n_jobs)(mol_passes_filters, gen)
    return np.mean(passes)


def internal_diversity(gen, n_jobs=1, gpu=-1, fp_type='morgan', gen_fps=None, p=1):
    '''
    Computes internal diversity as:
    1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    '''
    if gen_fps is None:
        gen_fps = fingerprints(gen, fp_type=fp_type, n_jobs=n_jobs)
    return 1 - (average_agg_tanimoto(gen_fps, gen_fps,
                                     agg='mean', gpu=gpu, p=p)).mean()


def fraction_unique(gen, k=None, n_jobs=1, check_validity=True):
    '''
    Computes a number of unique molecules
    :param gen: list of SMILES
    :param k: compute unique@k
    :param check_validity: raises ValueError if invalid molecules are present
    '''
    if k is not None:
        if len(gen) < k:
            warnings.warn(
                "Can't compute unique@{}. gen contains only {} molecules".format(
                    k, len(gen)))
        gen = gen[:k]
    canonic = set(mapper(n_jobs)(canonic_smiles, gen))
    if None in canonic and check_validity:
        raise ValueError("Invalid molecule passed to unique@k")
    return len(canonic) / len(gen)


def fraction_valid(gen, n_jobs=1):
    '''
    Computes a number of valid molecules
    :param gen: list of SMILES
    '''
    gen = mapper(n_jobs)(get_mol, gen)
    return 1 - gen.count(None) / len(gen)


def remove_invalid(gen, canonize=True, n_jobs=1):
    '''
    Removes invalid molecules from the dataset
    '''
    if canonize:
        mols = mapper(n_jobs)(get_mol, gen)
        return [gen_ for gen_, mol in zip(gen, mols) if mol is not None]
    else:
        return [x for x in mapper(n_jobs)(canonic_smiles, gen) if
                x is not None]


class Metric:
    def __init__(self, n_jobs=1, gpu=-1, batch_size=512, **kwargs):
        self.n_jobs = n_jobs
        self.gpu = gpu
        self.batch_size = batch_size
        for k, v in kwargs.values():
            setattr(self, k, v)

    def __call__(self, test=None, gen=None, ptest=None, pgen=None):
        assert (test is None) != (ptest is None), "specify test xor ptest"
        assert (gen is None) != (pgen is None), "specify gen xor pgen"
        if ptest is None:
            ptest = self.precalc(test)
        if pgen is None:
            pgen = self.precalc(gen)
        return self.metric(ptest, pgen)

    def precalc(self, moleclues):
        raise NotImplementedError

    def metric(self, ptest, pgen):
        raise NotImplementedError


class FCDMetric(Metric):
    '''
    Computes Frechet ChemNet Distance
    '''        
    def precalc(self, smiles):
        if len(smiles) < 2:
            warnings.warn("Can't compute FCD for less than 2 molecules")
            return np.nan

        chemnet_activations = get_predictions(smiles, gpu=self.gpu,
                                              batch_size=self.batch_size)
        mu = chemnet_activations.mean(0)
        sigma = np.cov(chemnet_activations.T)
        return {'mu': mu, 'sigma': sigma}
    
    def metric(self, ptest, pgen):
        return calculate_frechet_distance(ptest['mu'], ptest['sigma'],
                                          pgen['mu'], pgen['sigma'])



class SNNMetric(Metric):
    '''
    Computes average max similarities of gen SMILES to test SMILES
    '''
    def __init__(self, fp_type='morgan', **kwargs):
        self.fp_type = fp_type
        super().__init__(**kwargs)

    def precalc(self, mols):
        return {'fps': fingerprints(mols, n_jobs=self.n_jobs, fp_type=self.fp_type)}
    
    def metric(self, ptest, pgen):
        return average_agg_tanimoto(ptest['fps'], pgen['fps'], gpu=self.gpu)


def cos_distance(test_counts, gen_counts):
    '''
    Computes 1 - cosine similarity between
     dictionaries of form {name: count}. Non-present
     elements are considered zero
    '''
    if len(test_counts) == 0 or len(gen_counts) == 0:
        return np.nan
    keys = np.unique(list(test_counts.keys()) + list(gen_counts.keys()))
    test_vec = np.array([test_counts.get(k, 0) for k in keys])
    gen_vec = np.array([gen_counts.get(k, 0) for k in keys])
    return 1 - cosine(test_vec, gen_vec)


class FragMetric(Metric):
    def precalc(self, mols):
        return {'frag': compute_fragments(mols, n_jobs=self.n_jobs)}

    def metric(self, ptest, pgen):
        return cos_distance(ptest['frag'], pgen['frag'])


class ScafMetric(Metric):
    def precalc(self, mols):
        return {'scaf': compute_scaffolds(mols, n_jobs=self.n_jobs)}

    def metric(self, ptest, pgen):
        return cos_distance(ptest['scaf'], pgen['scaf'])


class FrechetMetric(Metric):
    def __init__(self, func=None, **kwargs):
        self.func = func
        super().__init__(**kwargs)
        
    def precalc(self, mols):
        if self.func is not None:
            values = mapper(self.n_jobs)(self.func, mols)
        else:
            values = mols
        return {'mu': np.mean(values), 'var': np.var(values)}
    
    def metric(self, ptest, pgen):
        return calculate_frechet_distance(ptest['mu'], ptest['var'],
                                          pgen['mu'], pgen['var'])
