import os

import torch
import pandas as pd
import random

from dataset import SmilesDataset
from model import CharLSTM
from utils import set_seed
from testing import test_model


def train_model(train_dataset, val_dataset, path_to_save, n_epochs, batch_size, lr,
                n_jobs, cuda_device, seed=0):
    set_seed(seed)
    char_rnn = CharLSTM(hidden_size=32, num_layers=1, train_dataset=train_dataset, val_dataset=val_dataset)
    char_rnn = char_rnn.to(torch.device('cuda:{device}'.format(device=cuda_device)))

    char_rnn.fit(train_dataset=train_dataset,
                 val_dataset=val_dataset,
                 n_epochs=n_epochs,
                 batch_size=batch_size,
                 path_to_save=path_to_save,
                 lr=lr,
                 n_jobs=n_jobs)

    test_model(char_rnn, val_dataset)


# TODO: add cmd args
if __name__ == '__main__':
    set_seed(0)

    data_path = '/media/Molecules/molecules.csv'

    train_size = 1_000_000
    val_size = 100_000

    path_to_save = "./saved_weights/attempt_1"

    n_epochs = 30
    batch_size = 64
    lr = 1e-3

    cuda_device = 0
    n_jobs = 3

    data = pd.read_csv(data_path, usecols=['SMILES'], nrows=(train_size + val_size), squeeze=True)
    data = data.tolist()

    random.shuffle(data)
    train_data = data[:train_size]
    val_data = data[train_size: (train_size + val_size)]

    train_dataset = SmilesDataset(train_data)
    val_dataset = SmilesDataset(val_data)

    train_model(train_dataset, val_dataset, path_to_save, n_epochs, batch_size, lr, n_jobs, cuda_device)

    char_rnn = CharLSTM(hidden_size=32, num_layers=1, train_dataset=train_dataset, val_dataset=val_dataset)
    char_rnn = char_rnn.to(torch.device('cuda:{device}'.format(device=cuda_device)))
    char_rnn.load_state_dict(torch.load(os.path.join(path_to_save, "epoch_best.pt")))

    test_model(char_rnn, val_dataset)


