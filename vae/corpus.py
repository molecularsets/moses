import abc
from collections import defaultdict

import torch
import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from torch.utils.data import DataLoader, Dataset

SS = dict(
    bos='<bos>',
    eos='<eos>',
    pad='<pad>',
    unk='<unk>'
)


class PandasDataset(Dataset):
    def __init__(self, df):
        super().__init__()

        self.df = df.iloc[:, 0]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]


class Vocab:
    def __init__(self, tokens, ss):
        tokens = tokens | set(ss.values())

        stoi = {v: i for i, v in enumerate(tokens)}
        self.stoi = defaultdict(lambda: stoi[ss.unk])
        for k, v in stoi.items():
            self.stoi[k] = v

        itos = {v: k for k, v in self.stoi.items()}
        self.itos = defaultdict(lambda: ss.unk)
        for k, v in itos.items():
            self.itos[k] = v

        for k, v in ss.items():
            setattr(self, k, self.stoi[v])

        self.vectors = torch.eye(len(self.stoi))

    def __len__(self):
        return len(self.stoi)


class Corpus(BaseEstimator, TransformerMixin, abc.ABC):
    @abc.abstractmethod
    def fit(self, dataset):
        pass

    @abc.abstractmethod
    def transform(self, dataset):
        pass

    @abc.abstractmethod
    def reverse(self, x):
        pass


class OneHotCorpus(Corpus):
    def __init__(self, n_batch, device):
        self.n_batch = n_batch
        self.device = device

        self.vocab = None

    def fit(self, dataset):
        chars = set()
        for smiles in tqdm.tqdm_notebook(dataset):
            chars.update(smiles)

        self.vocab = Vocab(chars, SS)

        return self

    def transform(self, dataset):
        return DataLoader(dataset, batch_size=self.n_batch,
                          shuffle=True, collate_fn=self._collate_fn)

    def reverse(self, x):
        t_ss = set(getattr(self.vocab, ss)
                   for ss in ('bos', 'eos', 'pad', 'unk'))
        return [
            ''.join(self.vocab.itos[i] for i in i_x.cpu().numpy()
                    if i not in t_ss)
            for i_x in x
        ]
