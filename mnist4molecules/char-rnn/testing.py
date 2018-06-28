import random
import torch
import pandas as pd

from metrics import diversity, validity, uniqueness
from model import CharLSTM
from utils import set_seed
from dataset import SmilesDataset


def test_model(model, dataset, n_gen=1000, n_other=1000):
    gen_smiles = model.sample_smiles(n_gen)
    other_smiles = random.sample(list(dataset), n_other)

    internal_diversity_val = diversity(gen_smiles)
    external_diversity_val = diversity(gen_smiles, other_smiles)  # TODO: check it
    validity_val = validity(gen_smiles)
    uniqueness_val = uniqueness(gen_smiles)

    print('\n###########################')
    print('Internal Diversity = {}'.format(internal_diversity_val))
    print('External Diversity = {}'.format(external_diversity_val))
    print('Validity = {}'.format(validity_val))
    print('Uniqueness = {}'.format(uniqueness_val))
    print('############################\n')


# TODO: add cmd args
if __name__ == '__main__':
    set_seed(0)

    path_with_weights = "./saved_weights/attempt_3/epoch_best.pt"
    data_path = '/media/Molecules/molecules.csv'
    train_size = 1_000_000
    val_size = 100_000

    data = pd.read_csv(data_path, usecols=['SMILES'], nrows=(train_size + val_size), squeeze=True)
    data = data.tolist()

    random.shuffle(data)
    train_data = data[:train_size]
    val_data = data[train_size: (train_size + val_size)]

    train_dataset = SmilesDataset(train_data)
    val_dataset = SmilesDataset(val_data)

    char_rnn = CharLSTM(train_dataset=train_dataset, val_dataset=val_dataset)
    char_rnn.to(torch.device("cuda:0"))
    char_rnn.load_state_dict(torch.load(path_with_weights))

    test_model(char_rnn, val_dataset)
