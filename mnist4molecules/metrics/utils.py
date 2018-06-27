from .SA_Score import sascorer
import numpy as np
from collections import Counter
from functools import partial
from multiprocessing import Pool
import pandas as pd
import scipy.sparse
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.QED import qed
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan


def mapper(n_jobs, interactive=False):
    if n_jobs == 1:
        return map
    elif interactive:
        return Pool(n_jobs).imap
    else:
        return Pool(n_jobs).map


def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        return Chem.MolFromSmiles(smiles_or_mol)
    else:
        return smiles_or_mol

    
def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    return Chem.MolToSmiles(mol)


def logP(mol):
    '''
    Computes RDKit's logP
    '''
    return Chem.Crippen.MolLogP(mol)


def SA(mol):
    '''
    Computes RDKit's SA score
    '''
    return sascorer.calculateScore(mol)


def QED(mol):
    '''
    Computes RDKit's QED score
    '''
    return qed(mol)


def get_n_rings(mol):
    '''
    Computes the number of rings in a molecule
    '''
    r = len([len(x) for x in mol.GetRingInfo().AtomRings()])
    return r

def fragmenter(mol):
    '''
    fragment mol using BRICS and return smiles list
    '''
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    fgs = AllChem.FragmentOnBRICSBonds(mol)
    fgs_smi = Chem.MolToSmiles(fgs).split(".")
    return fgs_smi


def compute_fragments(mol_list, n_jobs=1):
    '''
    fragment list of mols using BRICS and return smiles list
    '''
    map_ = mapper(n_jobs)
    fragments = Counter()
    for mol_frag in map_(fragmenter, mol_list):
        fragments.update(mol_frag)
    return fragments


def compute_scaffolds(mol_list, n_jobs=1, min_rings=2):
    '''
    Extracts a scafold from a molecule in a form of a canonic SMILES
    '''
    scaffolds = Counter()
    map_ = mapper(n_jobs)
    scaffolds = Counter(map_(partial(compute_scaffold, min_rings=min_rings), mol_list))
    if None in scaffolds:
        scaffolds.pop(None)
    return scaffolds


def compute_scaffold(mol, min_rings=2):
    if isinstance(mol, str):
        mol = get_mol(mol)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    n_rings = get_n_rings(scaffold)
    scaffold_smiles = Chem.MolToSmiles(scaffold) 
    if scaffold_smiles == '' or n_rings < min_rings:
        return None
    else:
        return scaffold_smiles


def average_max_tanimoto(stock_vecs, gen_vecs, batch_size=10000, gpu=-1):
    '''
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules
    :param stock_vecs: numpy array <n_vectors x dim>
    :param gen_vecs: numpy array <n_vectors' x dim>
    '''
    if gpu != -1:
        device = "cuda:{}".format(gpu)
    else:
        device = "cpu"
    best_tanimoto = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j:j+batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen =  torch.tensor(gen_vecs[i:i+batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) + y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            best_tanimoto[i:i+y_gen.shape[1]] = np.maximum(best_tanimoto[i:i+y_gen.shape[1]], jac.max(0))
    return np.mean(best_tanimoto)


def fingerprint(smiles_or_mol, type='MACCS', dtype=None, morgan__r=4, morgan__n=2048, *args, **kwargs):
    '''
    Generates fingerprint for SMILES
    If smiles is invalid, returns None
    Returns numpy array of fingerprint bits
    :param smiles: SMILES string
    :param type: type of fingerprint: [MACCS|morgan]
    :param dtype: if not None, specifies the dtype of returned array
    '''
    ftype = type.lower()
    molecule = get_mol(smiles_or_mol, *args, **kwargs)
    if molecule is None:
        return None
    if ftype == 'maccs':
        keys  = MACCSkeys.GenMACCSKeys(molecule)
        keys = np.array(keys.GetOnBits())
        fingerprint = np.zeros(166, dtype='uint8')
        if len(keys) != 0:
            fingerprint[keys-1] = 1 # We drop 0-th key that is always zero
    elif ftype == 'morgan':
        fingerprint = np.asarray(Morgan(molecule, morgan__r, nBits=morgan__n), dtype='uint8')
    else:
        raise ValueError
    if dtype is not None:
        fingerprint = fingerprint.astype(dtype)
    return fingerprint


def fingerprints(smiles_mols_array, n_jobs=1, already_unique=False, *args, **kwargs):
    '''
    Computes fingerprints of smiles np.array/list/pd.Series with n_jobs workers. e.g. fingerprints(smiles_mols_array, type='morgan', n_jobs=10)
    Inserts np.NaN to rows corresponding to incorrect smiles. IMPORTANT: if there is at least one np.NaN, the dtype would be float
    :param smiles_mols_array: list/array/pd.Series of smiles or already computed RDKit molecules
    :param n_jobs: number of parralel workers to execute
    :param already_unique: flag for performance reasons, if smiles array is big and already unique. Its value is set to True if smiles_mols_array contain RDKit molecules already.
    '''
    assert n_jobs > 0 and isinstance(n_jobs, int), 'n_jobs must be positive and integer'
    if isinstance(smiles_mols_array, pd.Series):
        smiles_mols_array = smiles_mols_array.values
    else:
        smiles_mols_array = np.asarray(smiles_mols_array)
    if not isinstance(smiles_mols_array[0], str):
        already_unique = True

    if not already_unique:
        smiles_mols_array, inv_index = np.unique(smiles_mols_array, return_inverse=True)
    
    if n_jobs > 1:
        if smiles_mols_array.shape[0] < n_jobs:
            n_jobs = smiles_mols_array.shape[0]
        from multiprocessing import Pool
        from functools import partial
        p = Pool(n_jobs)
        try:
            fps = p.map(partial(fingerprint, *args, **kwargs), smiles_mols_array)
        finally:
            p.terminate()
    else:
        fps = [fingerprint(smiles, *args, **kwargs) for smiles in smiles_mols_array]
    
    length = 1 # Need to know the length to convert None into np.array with nan values
    for fp in fps:
        if fp is not None:
            length = fp.shape[-1]
            first_fp = fp
            break
    fps = [fp if fp is not None else np.array([np.NaN]).repeat(length)[None, :] for fp in fps]
    if scipy.sparse.issparse(first_fp):
        fps = scipy.sparse.vstack(fps).tocsr()
    else:
        fps = np.vstack(fps)
    if not already_unique: 
        return fps[inv_index]
    else:
        return fps


def tanimoto(fingerprints, fingerprints_right=None, mode='pairwise'):
    '''
    Computes pairwize Tanimoto similarity between all pairs of fingerprints.
    If fingerprints_right is given, will compute distances between all molecules
    from fingerprints and fingerprints_right. If mode == 'paired', than will compute
    Tanimoto between corresponding rows. Lengths of fingerprints andfingerprints_right
    will have to be the same
    :param fingerprints: numpy array or torch tensor 
    :param fingerprints_right: numpy array or torch tensor 
    :param mode: [pairwise|paired]
    Output:
    :result similarity: Tanimoto score for each row
    '''
    if fingerprints_right is None:
        fingerprints_right = fingerprints

    if mode == 'pairwise':
        if isinstance(fingerprints_right, torch.Tensor):
            fingerprints_right = fingerprints_right.transpose(0, 1)
            total = fingerprints.sum(1, keepdim=True) + fingerprints_right.sum(0, keepdim=True)
        else:
            fingerprints_right = fingerprints_right.T
            total = fingerprints.sum(1, keepdims=True) + fingerprints_right.sum(0, keepdims=True)
        intersection = fingerprints @ fingerprints_right
        union = total - intersection
    elif mode == 'paired':
        assert fingerprints.shape == fingerprints_right.shape, "For paired mode should have the same number of rows"
        intersection = (fingerprints * fingerprints_right).sum(1)
        union = fingerprints.sum(1) + fingerprints_right.sum(1) - intersection
    else:
        raise ValueError

    if isinstance(fingerprints_right, torch.Tensor):
        intersection = intersection.float()
    else:
        intersection = intersection.astype(float)
    intersection[union == 0] = 1
    union[union == 0] = 1
    scores = intersection / union
    return scores
