from collections import defaultdict

import torch
import tqdm
from keras.preprocessing.sequence import pad_sequences
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets.lfw import Bunch
from torch.utils.data import DataLoader, Dataset

SS = Bunch(
    bos='<bos>',
    eos='<eos>',
    pad='<pad>',
    unk='<unk>'
)


class ListDataset(Dataset):
    def __init__(self, l_data):
        super().__init__()

        self.l_data = l_data

    def __len__(self):
        return len(self.l_data)

    def __getitem__(self, idx):
        return self.l_data[idx]


class Vocab:
    def __init__(self, tokens, unk):
        assert unk in tokens

        stoi = {v: i for i, v in enumerate(tokens)}
        self.stoi = defaultdict(lambda: stoi[unk])
        for k, v in stoi.items():
            self.stoi[k] = v

        itos = {v: k for k, v in self.stoi.items()}
        self.itos = defaultdict(lambda: unk)
        for k, v in itos.items():
            self.itos[k] = v

        self.vectors = torch.eye(len(self.stoi))

    def __len__(self):
        return len(self.stoi)


class OneHotCorpus(BaseEstimator, TransformerMixin):
    def __init__(self, n_batch, device, ss=SS):
        self.n_batch = n_batch
        self.device = device
        self.ss = ss

        self.vocab, self.n_len = None, None

    def fit(self, dataset):
        chars, n_len = set(), 0
        for smiles in tqdm.tqdm_notebook(dataset):
            chars.update(smiles)
            n_len = max(n_len, len(smiles))

        self.vocab = Vocab(set(self.ss.values()) | chars, self.ss.unk)
        self.n_len = n_len + 2  # + <bos> and <eos>

        return self

    def transform(self, dataset):
        return DataLoader(dataset, batch_size=self.n_batch, shuffle=True,
                          collate_fn=self._collate_fn)

    def reverse(self, x):
        t_ss = set((self.vocab.stoi[getattr(self.ss, ss)]
                    for ss in ('bos', 'eos', 'pad', 'unk')))
        return [
            ''.join(self.vocab.itos[i] for i in i_x.cpu().numpy()
                    if i not in t_ss)
            for i_x in x
        ]

    def _collate_fn(self, l_smiles):
        bos, eos, pad, unk = (self.vocab.stoi[getattr(self.ss, ss)]
                              for ss in ('bos', 'eos', 'pad', 'unk'))
        l_smiles = [
            torch.tensor([bos] + [self.vocab.stoi[c] for c in smiles] + [eos])
            for smiles in l_smiles]
        batch = pad_sequences(l_smiles, maxlen=self.n_len, dtype=int,
                              padding='post', truncating='post', value=pad)
        return torch.tensor(batch).to(self.device)
