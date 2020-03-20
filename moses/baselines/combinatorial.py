from collections import Counter
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem

import moses
from moses.metrics.utils import fragmenter
from moses.utils import mapper


class CombinatorialGenerator:
    def __init__(self, n_jobs=1):
        """
        Combinatorial Generator randomly connects BRICS fragments

        Arguments:
            n_jobs: number of processes for training
        """
        self.n_jobs = n_jobs
        self.fitted = False

    def fit(self, data):
        """
        Collects fragment frequencies in a training set

        Arguments:
            data: list of SMILES, training dataset

        """
        # Split molecules from dataset into BRICS fragments
        fragments = mapper(self.n_jobs)(fragmenter, data)

        # Compute fragment frequencies
        counts = Counter()
        for mol_frag in fragments:
            counts.update(mol_frag)
        counts = pd.DataFrame(
            pd.Series(counts).items(),
            columns=['fragment', 'count']
        )
        counts['attachment_points'] = [
            fragment.count('*')
            for fragment in counts['fragment'].values
        ]
        counts['frequency'] = counts['count'] / counts['count'].sum()
        self.fragment_counts = counts

        # Compute number of fragments distribution
        fragments_count_distribution = Counter([len(f) for f in fragments])
        total = sum(fragments_count_distribution.values())
        for k in fragments_count_distribution:
            fragments_count_distribution[k] /= total
        self.fragments_count_distribution = fragments_count_distribution
        self.fitted = True
        return self

    def save(self, path):
        """
        Saves a model using pickle

        Arguments:
            path: path to .pkl file for saving
        """
        if not self.fitted:
            raise RuntimeError("Can't save empty model."
                               " Fit the model first")
        data = {
            'fragment_counts': self.fragment_counts,
            'fragments_count_distribution': self.fragments_count_distribution
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path):
        """
        Loads saved model

        Arguments:
            path: path to saved .pkl file

        Returns:
            Loaded CombinatorialGenerator
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls()
        model.fragment_counts = data['fragment_counts']
        model.fragments_count_distribution = \
            data['fragments_count_distribution']
        model.fitted = True
        return model

    def generate_one(self, seed=None):
        """
        Generates a SMILES string using fragment frequencies

        Arguments:
            seed: if specified, will set numpy seed before sampling

        Retruns:
            SMILES string
        """
        if seed is not None:
            np.random.seed(seed)
        if not self.fitted:
            raise RuntimeError("Fit the model before generating")
        mol = None

        # Sample the number of fragments
        count_values, count_probs = zip(
            *self.fragments_count_distribution.items())
        total_fragments = np.random.choice(count_values, p=count_probs)

        counts = self.fragment_counts
        for i in range(total_fragments):
            # Enforce lower and upper limit on the number of connection points
            if mol is None:
                current_attachments = 0
                max_attachments = total_fragments - 1
            else:
                connections_mol = self.get_connection_points(mol)
                current_attachments = len(connections_mol)
                max_attachments = (
                    total_fragments - i - current_attachments + 1
                )
            counts_masked = counts[
                counts['attachment_points'] <= max_attachments
            ]
            if total_fragments == 1:
                min_attachments = 0
            elif i != 0 and current_attachments == 1 and \
                    total_fragments > i + 1:
                min_attachments = 2
            else:
                min_attachments = 1
            counts_masked = counts_masked[
                counts_masked['attachment_points'] >= min_attachments
            ]

            # Sample a new fragment
            new_fragment = counts_masked.sample(
                weights=counts_masked['frequency']
            )
            new_fragment = dict(new_fragment.iloc[0])
            fragment = Chem.MolFromSmiles(new_fragment['fragment'])
            if mol is None:
                mol = fragment
            else:
                # Connect a new fragment to the molecule
                connection_mol = np.random.choice(connections_mol)
                connections_fragment = self.get_connection_points(fragment)
                connection_fragment = np.random.choice(connections_fragment)
                mol = self.connect_mols(mol, fragment,
                                        connection_mol, connection_fragment)
        smiles = Chem.MolToSmiles(mol)
        return smiles

    @staticmethod
    def get_connection_points(mol):
        atoms = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == '*':
                atoms.append(atom)
        return atoms

    @staticmethod
    def connect_mols(mol1, mol2, atom1, atom2):
        combined = Chem.CombineMols(mol1, mol2)
        emol = Chem.EditableMol(combined)
        neighbor1_idx = atom1.GetNeighbors()[0].GetIdx()
        neighbor2_idx = atom2.GetNeighbors()[0].GetIdx()
        atom1_idx = atom1.GetIdx()
        atom2_idx = atom2.GetIdx()
        bond_order = atom2.GetBonds()[0].GetBondType()
        emol.AddBond(neighbor1_idx,
                     neighbor2_idx + mol1.GetNumAtoms(),
                     order=bond_order)
        emol.RemoveAtom(atom2_idx + mol1.GetNumAtoms())
        emol.RemoveAtom(atom1_idx)
        mol = emol.GetMol()
        return mol


def reproduce(seed, samples_path=None, metrics_path=None,
              n_jobs=1, device='cpu', verbose=False,
              samples=30000):
    train = moses.get_dataset('train')
    model = CombinatorialGenerator(n_jobs=n_jobs)

    if verbose:
        print("Training...")
    model.fit(train)

    if verbose:
        print(f"Sampling for seed {seed}")
    seeds = list(range(
        (seed - 1) * samples, seed * samples
    ))
    samples = mapper(n_jobs)(model.generate_one, seeds)
    if samples_path is not None:
        with open(samples_path, 'w') as f:
            f.write('SMILES\n')
            for sample in samples:
                f.write(sample+'\n')
    if verbose:
        print(f"Computing metrics for seed {seed}")
    metrics = moses.get_all_metrics(
        samples, n_jobs=n_jobs, device=device)
    if metrics_path is not None:
        with open(metrics_path, 'w') as f:
            for key, value in metrics.items():
                f.write("%s,%f\n" % (key, value))
    return samples, metrics


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        "Reproduce combinatorial experiment")
    parser.add_argument(
        '--n_jobs', type=int, required=False,
        default=1, help='Number of threads for computing metrics')
    parser.add_argument(
        '--device', type=str, required=False,
        default='cpu', help='Device for computing metrics')
    parser.add_argument(
        '--metrics_path', type=str, required=False,
        default='.', help='Path to save metrics')
    args = parser.parse_known_args()[0]

    for seed in [1, 2, 3]:
        filename = f'combinatorial_metrics_{seed}.csv'
        metrics_path = os.path.join(args.metrics_path, filename)

        reproduce(seed=seed, n_jobs=args.n_jobs,
                  device=args.device, verbose=True,
                  metrics_path=metrics_path)
