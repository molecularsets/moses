import abc
import logging

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class Trainer(abc.ABC):
    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def fit(self, model, data):
        pass


class PandasDataset(Dataset):
    def __init__(self, df):
        super().__init__()

        self.df = df.loc[:, ['SMILES']].iloc[:, 0]

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

    def fit_transform(self, dataset):
        return self.fit(dataset).transform(dataset)


def get_device(config):
    if config.device_code >= 0:
        if torch.cuda.is_available():
            logger.info(f"Using GPU:{config.device_code}")
            return torch.device(f'cuda:{config.device_code}')
        else:
            logger.warning("GPU is't available, CPU will be used")
            return torch.device('cpu')
    else:
        logger.info("Using CPU")
        return torch.device('cpu')


def set_logger(config):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        level=getattr(logging, config.log_level.upper()))
