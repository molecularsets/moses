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


def get_all_metrics(ref, gen, k=[1000, 10000], n_jobs=1, gpu=-1, batch_size=512):
    '''
    Computes all available metrics between two lists of SMILES:
    * %valid
    ----- Next metrics are only computed for valid molecules -----
    * %unique@k
    * Frechet ChemNet Distance (FCD)
    * fragment similarity
    * scaffold similarity
    * morgan similarity
    '''
    metrics = {}
    if n_jobs != 1:
        pool = Pool(n_jobs)
    else:
        pool = 1
    metrics['valid'] = fraction_valid(gen, n_jobs=n_jobs)
    gen = remove_invalid(gen, canonize=True)
    ref = remove_invalid(ref, canonize=True)
    gen_mols = mapper(pool)(get_mol, gen)
    ref_mols = mapper(pool)(get_mol, ref)

    if not isinstance(k, (list, tuple)):
        k = [k]
    for k_ in k:
        metrics['unique@{}'.format(k_)] = fraction_unique(gen, k_, pool)

    metrics['FCD'] = frechet_chemnet_distance(ref, gen, gpu=gpu, batch_size=batch_size)
    metrics['morgan'] = morgan_similarity(ref_mols, gen_mols, pool, gpu=gpu)
    metrics['fragments'] = fragment_similarity(ref_mols, gen_mols, pool)
    metrics['scaffolds'] = scaffold_similarity(ref_mols, gen_mols, pool)
    metrics['internal_diversity'] = internal_diversity(gen_mols, pool)
    metrics['filters'] = fraction_passes_filters(gen_mols, pool)
    metrics['logP'] = frechet_distance(ref_mols, gen_mols, logP, pool)
    metrics['SA'] = frechet_distance(ref_mols, gen_mols, SA, pool)
    metrics['QED'] = frechet_distance(ref_mols, gen_mols, QED, pool)
    metrics['NP'] = frechet_distance(ref_mols, gen_mols, NP, pool)
    metrics['weight'] = frechet_distance(ref_mols, gen_mols, weight, pool)
    if n_jobs != 1:
        pool.close()
    return metrics


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


def internal_diversity(gen, n_jobs=1, gpu=-1, fp_type='morgan'):
    '''
    Computes internal diversity as:
    1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    '''
    gen_fps = fingerprints(gen, fp_type=fp_type, n_jobs=n_jobs)
    return 1 - (average_agg_tanimoto(gen_fps, gen_fps,
                                     agg='mean', gpu=gpu)).mean()


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


def morgan_similarity(ref, gen, n_jobs=1, gpu=-1):
    return fingerprint_similarity(ref, gen, 'morgan', n_jobs=n_jobs, gpu=gpu)


def frechet_chemnet_distance(ref, gen, gpu=-1, batch_size=512):
    '''
    Computes Frechet ChemNet Distance between two lists of SMILES
    '''
    if len(ref) < 2 or len(gen) < 2:
        warnings.warn("Can't compute FCD for less than 2 molecules")
        return np.nan
    gen_activations, ref_activations = get_predictions(gen, ref, gpu=gpu,
                                                       batch_size=batch_size)
    mu1 = gen_activations.mean(0)
    mu2 = ref_activations.mean(0)
    sigma1 = np.cov(gen_activations.T)
    sigma2 = np.cov(ref_activations.T)
    fcd = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fcd


def fingerprint_similarity(ref, gen, fp_type='morgan', n_jobs=1, gpu=-1):
    '''
    Computes average max similarities of gen SMILES to ref SMILES
    '''
    ref_fp = fingerprints(ref, n_jobs=n_jobs, fp_type=fp_type, morgan__r=2,
                          morgan__n=1024)
    gen_fp = fingerprints(gen, n_jobs=n_jobs, fp_type=fp_type, morgan__r=2,
                          morgan__n=1024)
    similarity = average_agg_tanimoto(ref_fp, gen_fp, gpu=gpu)
    return similarity


def count_distance(ref_counts, gen_counts):
    '''
    Computes 1 - cosine similarity between
     dictionaries of form {type: count}. Non-present
     elements are considered zero
    '''
    if len(ref_counts) == 0 or len(gen_counts) == 0:
        return np.nan
    keys = np.unique(list(ref_counts.keys()) + list(gen_counts.keys()))
    ref_vec = np.array([ref_counts.get(k, 0) for k in keys])
    gen_vec = np.array([gen_counts.get(k, 0) for k in keys])
    return 1 - cosine(ref_vec, gen_vec)


def fragment_similarity(ref, gen, n_jobs=1):
    ref_fragments = compute_fragments(ref, n_jobs=n_jobs)
    gen_fragments = compute_fragments(gen, n_jobs=n_jobs)
    return count_distance(ref_fragments, gen_fragments)


def scaffold_similarity(ref, gen, n_jobs=1):
    ref_scaffolds = compute_scaffolds(ref, n_jobs=n_jobs)
    gen_scaffolds = compute_scaffolds(gen, n_jobs=n_jobs)
    return count_distance(ref_scaffolds, gen_scaffolds)


def frechet_distance(ref, gen, func, n_jobs=1):
    ref_values = mapper(n_jobs)(func, ref)
    gen_values = mapper(n_jobs)(func, gen)
    ref_mean = np.mean(ref_values)
    ref_var = np.var(ref_values)
    gen_mean = np.mean(gen_values)
    gen_var = np.var(gen_values)
    return calculate_frechet_distance(ref_mean, ref_var,
                                      gen_mean, gen_var)
