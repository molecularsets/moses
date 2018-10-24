import copy

import rdkit.Chem as Chem
import tqdm
from torch.utils.data import DataLoader

from moses.junction_tree.jtnn.mol_tree import MolTree


class JTreeVocab:

    def __init__(self, smiles_list):
        self.vocab = smiles_list
        self.vmap = {x: i for i, x in enumerate(self.vocab)}
        self.slots = [get_slots(smiles) for smiles in self.vocab]

    def get_index(self, smiles):
        return self.vmap[smiles]

    def get_smiles(self, idx):
        return self.vocab[idx]

    def get_slots(self, idx):
        return copy.deepcopy(self.slots[idx])

    def size(self):
        return len(self.vocab)


class JTreeCorpus:
    def __init__(self, n_batch, device):
        self.n_batch = n_batch
        self.device = device
        self.vocab = None

    def fit(self, dataset=None, vocabulary=None):
        if vocabulary is None:
            if dataset is None:
                raise ValueError("You should specify either dataset or vocabulary")

            clusters = set()

            for smiles in tqdm.tqdm(dataset):
                mol = MolTree(smiles)
                for c in mol.nodes:
                    clusters.add(c.smiles)

            self.vocab = JTreeVocab(sorted(list(clusters)))
        else:
            self.vocab = vocabulary

        return self

    def transform(self, dataset):
        return DataLoader(dataset, batch_size=self.n_batch, shuffle=True, collate_fn=self._collate_fn, num_workers=4,
                          drop_last=True)

    def _collate_fn(self, l_smiles):
        mol_trees = []

        for i in l_smiles:
            mol_tree = MolTree(i)
            mol_tree.recover()
            mol_tree.assemble()
            mol_trees.append(mol_tree)

        return mol_trees


def get_slots(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return [(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs()) for atom in mol.GetAtoms()]
