import torch

from torch.utils.data import DataLoader
from mnist4molecules.utils import CharVocab


class OneHotCorpus:
    def __init__(self, n_batch, device):
        self.n_batch = n_batch
        self.device = device

        self.vocab = None

    def fit(self, dataset):
        self.vocab = CharVocab.from_data(dataset)

        return self

    def transform(self, dataset):
        return DataLoader(dataset, batch_size=self.n_batch, num_workers=4, shuffle=True, collate_fn=self._collate_fn)

    def _collate_fn(self, l_smiles):
        l_smiles.sort(key=len, reverse=True)

        data = [self.vocab.string2ids(smiles, add_bos=True, add_eos=True) for smiles in l_smiles]

        inputs = [torch.tensor(d[:-1], dtype=torch.long) for d in data]
        targets = [torch.tensor(d[1:], dtype=torch.long) for d in data]

        return inputs, targets
