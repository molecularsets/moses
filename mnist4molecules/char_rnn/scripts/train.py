import pandas as pd
import torch

from mnist4molecules.char_rnn.config import get_train_parser
from mnist4molecules.char_rnn.datautils import OneHotCorpus
from mnist4molecules.char_rnn.model import CharRNN
from mnist4molecules.char_rnn.trainer import CharRNNTrainer
from mnist4molecules.config import get_config
from mnist4molecules.utils import PandasDataset, get_device, set_logger

if __name__ == '__main__':
    torch.manual_seed(0)

    config = get_config(get_train_parser())
    set_logger(config)

    train = pd.read_csv(config.train_load, usecols=['SMILES'])
    train = PandasDataset(train)

    device = get_device(config)
    corpus = OneHotCorpus(config.batch, device)
    train_dataloader = corpus.fit(train).transform(train)

    model = CharRNN(corpus.vocab, config.hidden, config.num_layers, device).to(device)

    trainer = CharRNNTrainer(config)
    trainer.fit(model, train_dataloader)

    torch.save(config, config.config_save)
    torch.save(corpus.vocab, config.vocab_save)

