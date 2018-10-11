import torch

from torch.utils.data import DataLoader
from moses.utils import CharVocab


class OneHotVocab(CharVocab):
    def __init__(self, *args, **kwargs):
        super(OneHotVocab, self).__init__(*args, **kwargs)
        self.vectors = torch.eye(len(self.c2i))


class OneHotCorpus:
    def __init__(self, n_batch, device):
        self.n_batch = n_batch
        self.device = device

        self.vocab = None

    def fit(self, dataset):
        self.vocab = OneHotVocab.from_data(dataset)

        return self

    def transform(self, dataset):
        return DataLoader(dataset, batch_size=self.n_batch,
                          shuffle=True, collate_fn=self._collate_fn)

    def _collate_fn(self, l_smiles):
        l_smiles.sort(key=len, reverse=True)
        data = [self.vocab.string2ids(smiles, add_bos=True, add_eos=True) for smiles in l_smiles]

        return [torch.tensor(d, dtype=torch.long, device=self.device) for d in data]
