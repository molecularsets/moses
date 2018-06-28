import abc

import torch
from torch.utils.data import Dataset


class Trainer(abc.ABC):
    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def fit(self, model, data):
        pass


class PandasDataset(Dataset):
    def __init__(self, df):
        super().__init__()

        self.df = df.iloc[:, 0]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]


class Vocab(abc.ABC):
    @abc.abstractmethod
    def reverse(self, x):
        pass


class Corpus(abc.ABC):
    @abc.abstractmethod
    def fit(self, dataset):
        pass

    @abc.abstractmethod
    def transform(self, dataset):
        pass


def get_device(config):
    return torch.device(
        f'cuda:{config.device_code}'
        if config.device_code >= 0 and torch.cuda.is_available()
        else 'cpu'
    )
