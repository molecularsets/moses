import torch
import torch.nn.utils.rnn as rnn_utils

from torch.utils.data import DataLoader
from moses.utils import CharVocab


class OneHotCorpus:
    def __init__(self, n_batch, device):
        self.n_batch = n_batch
        self.device = device

        self.vocab = None

    def fit(self, dataset):
        self.vocab = CharVocab.from_data(dataset)

        return self

    def transform(self, dataset):
        return DataLoader(dataset, batch_size=self.n_batch, shuffle=True, collate_fn=self._collate)

    def _collate(self, l_smiles):
        l_smiles.sort(key=len, reverse=True)
        tensors = []

        for s in l_smiles:
            ids = self.vocab.string2ids(s, add_bos=True, add_eos=True)
            tensors.append(torch.tensor(ids, dtype=torch.long, device=self.device))

        prevs = rnn_utils.pad_sequence([t[:-1] for t in tensors], batch_first=True, padding_value=self.vocab.pad)
        nexts = rnn_utils.pad_sequence([t[1:] for t in tensors], batch_first=True, padding_value=self.vocab.pad)
        lens = torch.tensor([len(t) - 1 for t in tensors], dtype=torch.long, device=self.device)

        return prevs, nexts, lens
