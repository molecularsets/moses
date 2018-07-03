import sys
sys.path.insert(0, '..')

import torch

from mnist4molecules.vae.config import get_parser
from mnist4molecules.vae.corpus import OneHotCorpus
from mnist4molecules.vae.model import VAE
from mnist4molecules.vae.trainer import VAETrainer
from utils import add_train_args, read_smiles_csv, set_seed


def main(config):
    set_seed(config.seed)

    train = read_smiles_csv(config.train_load)

    device = torch.device(config.device)

    corpus = OneHotCorpus(config.n_batch, device)
    train = corpus.fit(train).transform(train)

    model = VAE(corpus.vocab, config).to(device)

    trainer = VAETrainer(config)
    trainer.fit(model, train)

    torch.save(model.state_dict(), config.model_save)
    torch.save(config, config.config_save)
    torch.save(corpus.vocab, config.vocab_save)


if __name__ == '__main__':
    parser = add_train_args(get_parser())
    config = parser.parse_known_args()[0]
    main(config)