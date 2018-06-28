import pandas as pd
import torch

from mnist4molecules.config import get_config
from mnist4molecules.utils import PandasDataset, get_device
from mnist4molecules.vae.config import get_train_parser
from mnist4molecules.vae.corpus import OneHotCorpus
from mnist4molecules.vae.model import VAE
from mnist4molecules.vae.trainer import VAETrainer

if __name__ == '__main__':
    config = get_config(get_train_parser())

    train = pd.read_csv(config.train_load, usecols=['SMILES'])
    train = PandasDataset(train)

    device = get_device(config)
    corpus = OneHotCorpus(config.n_batch, device)
    train = corpus.fit(train).transform(train)

    model = VAE(corpus.vocab, config).to(device)

    trainer = VAETrainer(config)
    trainer.fit(model, train)

    torch.save(model.state_dict(), config.model_save)
    torch.save(config, config.config_save)
    torch.save(corpus.vocab, config.vocab_save)
