import numpy as np

from rdkit.Chem import AllChem as Chem
from rdkit import DataStructs


def diversity(smiles, other_smiles=None):
    def remap(x, x_min, x_max):
        if x_max == 0 and x_min == 0:
            return 0

        if x_max - x_min == 0:
            return x

        return (x - x_min) / (x_max - x_min)

    def calc_diversity(smile, fps):
        mol = Chem.MolFromSmiles(smile)
        if smile != '' and mol is not None and mol.GetNumAtoms() > 1:
            ref_fps = Chem.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
            dist = DataStructs.BulkTanimotoSimilarity(ref_fps, fps, returnDistance=True)
            mean_dist = np.mean(dist)

            low_rand_dst, mean_div_dst = 0.9, 0.945
            val = remap(mean_dist, low_rand_dst, mean_div_dst)
            val = np.clip(val, 0.0, 1.0)

            return val

        return 0

    if other_smiles is None:
        other_smiles = smiles

    mols = [Chem.MolFromSmiles(s) for s in other_smiles]
    fps = [Chem.GetMorganFingerprintAsBitVect(m, 4, nBits=2048) for m in mols if m is not None]
    divs = [calc_diversity(s, fps=fps) for s in smiles]

    return np.mean(divs)


def validity(smiles):
    def verify_sequence(smile):
        mol = Chem.MolFromSmiles(smile)

        return smile != '' and mol is not None and mol.GetNumAtoms() > 1

    n_smiles = len(smiles)
    n_valid_smiles = sum(map(verify_sequence, smiles))

    return n_valid_smiles / n_smiles


def uniqueness(smiles):
    n_smiles = len(smiles)
    n_uniq_smiles = len(set(smiles))

    return n_uniq_smiles / n_smiles