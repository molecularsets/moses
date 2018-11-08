import copy

import rdkit.Chem as Chem
import tqdm
from torch.utils.data import DataLoader

from moses.junction_tree.jtnn.mol_tree import MolTree
from multiprocessing import Pool
from moses.utils import SmilesDataset


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

    def fit(self, dataset=None, vocabulary=None, n_jobs=4):
        if vocabulary is None:
            if dataset is None:
                raise ValueError("You should specify either dataset or vocabulary")
            clusters = set()
            pool = Pool(n_jobs)
            for mol in tqdm.tqdm(pool.imap(MolTree, dataset),
                                 total=len(dataset),
                                 postfix=['Creating vocab']):
                for c in mol.nodes:
                    clusters.add(c.smiles)
            pool.close()
            self.vocab = JTreeVocab(sorted(list(clusters)))
        else:
            self.vocab = vocabulary

        return self

    def transform(self, dataset, num_workers=4):
        return DataLoader(SmilesDataset(dataset, transform=self.parse_molecule),
                          batch_size=self.n_batch, shuffle=True,
                          num_workers=num_workers,
                          collate_fn=self.dummy_collate,
                          drop_last=True)

    @staticmethod
    def dummy_collate(smiles_list):
        return list(smiles_list)

    @staticmethod
    def parse_molecule(smiles):
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        mol_tree.assemble()

        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)
                node.cand_mols.append(node.label_mol)

        return mol_tree


def get_slots(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return [(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs()) for atom in mol.GetAtoms()]
