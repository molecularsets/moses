import numpy as np
from .utils_fcd import get_predictions, calculate_frechet_distance
from .utils import compute_fragments, average_max_tanimoto, compute_scaffolds, tanimoto, fingerprints, get_mol, canonic_smiles
from scipy.spatial.distance import cosine
import warnings


def get_all_metrics(ref, gen, k=[1000, 10000], n_jobs=1, gpu=-1):
    '''
    Computes all available metrics between two lists of SMILES:
    1. %valid
    ----- Next metrics are only computed for valid molecules -----
    2. %unique@k
    3. FCD
    4. fragment similarity
    5. scaffold similarity
    6. topological similarity
    7. morgan similarity
    '''
    metrics = {}

    metrics['valid'] = fraction_valid(gen)
    gen = remove_invalid(gen)
    gen = [canonic_smiles(smiles) for smiles in gen]

    if not isinstance(k, (list, tuple)):
        k = [k]
    for k_ in k:
        metrics['unique@{}'.format(k_)] = fraction_unique(gen)

    metrics['FCD'] = frechet_chembl_distance(ref, gen, gpu=gpu)
    metrics['morgan'] = morgan_similarity(ref, gen, n_jobs=n_jobs, gpu=gpu)
    metrics['fragments'] = fragment_similarity(ref, gen, n_jobs=n_jobs)
    metrics['scaffolds'] = scaffold_similarity(ref, gen, n_jobs=n_jobs)
    metrics['internal_diversity'] = internal_diversity(gen, n_jobs=n_jobs)
    return metrics


def internal_diversity(gen, type='morgan', n_jobs=1):
    '''
    Computes internal diversity as:
    1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    '''
    gen_fps = fingerprints(gen, type=type, n_jobs=n_jobs)
    return (1-tanimoto(gen_fps, gen_fps)).mean()


def fraction_unique(gen, k=None, check_validity=True):
    '''
    Computes a number of unique molecules
    :param gen: list of SMILES
    :param k: compute unique@k
    :param check_validity: raises ValueError if invalid molecules are present
    '''
    if k is not None:
        if len(gen) < k:
            warnings.warn("Can't compute unique@{}. gen contains only {} molecules".format(len(gen), k))
        gen = gen[:k]
    canonic = set(map(canonic_smiles, gen))
    if None in canonic and check_validity:
        raise ValueError("Invalid molecule passed to unique@k")
    return len(canonic) / len(gen)


def fraction_valid(gen):
    '''
    Computes a number of valid molecules
    :param gen: list of SMILES
    '''
    gen = list(map(get_mol, gen))
    return 1 - gen.count(None) / len(gen)


def remove_invalid(gen, canonize=True):
    '''
    Removes invalid molecules from the dataset
    '''
    if canonize:
        mols = list(map(get_mol, gen))
    else:
        mols = list(map(canonic_smiles, gen))
    return [gen_ for gen_, mol in zip(gen, mols) if mol is not None]


def morgan_similarity(ref, gen, n_jobs=1, gpu=-1):
    return fingerprint_similarity(ref, gen, 'morgan', n_jobs=n_jobs, gpu=gpu)


def frechet_chembl_distance(ref, gen, gpu=-1):
    '''
    Computes Frechet Chembl Distance between two lists of SMILES
    '''
    ref_activations = get_predictions(ref, gpu=gpu)
    gen_activations = get_predictions(gen, gpu=gpu)
    mu1 = gen_activations.mean(0)
    mu2 = ref_activations.mean(0)
    sigma1 = np.cov(gen_activations.T)
    sigma2 = np.cov(ref_activations.T)
    fcd = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fcd


def fingerprint_similarity(ref, gen, type='topological', n_jobs=1, gpu=-1):
    '''
    Computes average max similarities of gen SMILES to ref SMILES
    '''
    ref_fp = fingerprints(ref, n_jobs=n_jobs, type=type, morgan__r=2, morgan__n=1024)
    gen_fp = fingerprints(gen, n_jobs=n_jobs, type=type, morgan__r=2, morgan__n=1024)
    similarity = average_max_tanimoto(ref_fp, gen_fp, gpu=gpu)
    return similarity


def count_distance(ref_counts, gen_counts):
    '''
    Computes 1 - cosine similarity between
     dictionaries of form {type: count}. Non-present
     elements are considered zero
    '''
    keys = np.unique(list(ref_counts.keys())+list(gen_counts.keys()))
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
