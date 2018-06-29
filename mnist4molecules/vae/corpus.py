from collections import UserDict

import torch
import tqdm
from attrdict import AttrDict
from torch.utils.data import DataLoader

from mnist4molecules.utils import Vocab, Corpus

SS = AttrDict(
    bos='<bos>',
    eos='<eos>',
    pad='<pad>',
    unk='<unk>'
)


class DefaultValueDict(UserDict):
    def __init__(self, default_value=None):
        super().__init__()
        self.default_value = default_value

    def __getitem__(self, item):
        if item in self.data:
            return self.data[item]
        else:
            return self.default_value


class OneHotVocab(Vocab):
    def __init__(self, tokens, ss):
        tokens = tokens | set(ss.values())

        stoi = {v: i for i, v in enumerate(tokens)}

        self.stoi = DefaultValueDict(stoi[ss.unk])
        for k, v in stoi.items():
            self.stoi[k] = v

        itos = {v: k for k, v in self.stoi.items()}
        self.itos = DefaultValueDict(ss.unk)
        for k, v in itos.items():
            self.itos[k] = v

        for k, v in ss.items():
            setattr(self, k, self.stoi[v])

        self.vectors = torch.eye(len(self.stoi))

    def __len__(self):
        return len(self.stoi)

    def reverse(self, x):
        t_ss = set(getattr(self, ss)
                   for ss in ('bos', 'eos', 'pad', 'unk'))
        return [
            ''.join(self.itos[i.item()] for i in i_x
                    if i.item() not in t_ss)
            for i_x in x
        ]


class OneHotCorpus(Corpus):
    def __init__(self, n_batch, device):
        self.n_batch = n_batch
        self.device = device

        self.vocab = None

    def fit(self, dataset):
        chars = set()
        for smiles in tqdm.tqdm(dataset, desc='Fitting corpus with vocab'):
            chars.update(smiles)

        self.vocab = OneHotVocab(chars, SS)

        return self

    def transform(self, dataset):
        return DataLoader(dataset, batch_size=self.n_batch,
                          shuffle=True, collate_fn=self._collate_fn)

    def _collate_fn(self, l_smiles):
        l_smiles.sort(key=len, reverse=True)
        return [
            torch.tensor([self.vocab.bos]
                         + [self.vocab.stoi[c] for c in smiles]
                         + [self.vocab.eos],
                         dtype=torch.long, device=self.device)
            for smiles in l_smiles
        ]
