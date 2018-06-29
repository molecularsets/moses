import sys
sys.path.insert(0, '..')

import torch

from mnist4molecules.char_rnn.config import get_parser
from mnist4molecules.char_rnn.datautils import OneHotCorpus
from mnist4molecules.char_rnn.model import CharRNN
from mnist4molecules.char_rnn.trainer import CharRNNTrainer
from utils import add_train_args, read_smiles_csv, set_seed


def main(config):
    set_seed(config.seed)

    train = read_smiles_csv(config.train_load)

    device = torch.device(config.device)

    corpus = OneHotCorpus(config.batch, device)
    train_dataloader = corpus.fit(train).transform(train)

    model = CharRNN(corpus.vocab, config.hidden, config.num_layers, device)
    model = model.to(device)

    trainer = CharRNNTrainer(config)
    trainer.fit(model, train_dataloader)

    torch.save(model.state_dict(), config.model_save)
    torch.save(config, config.config_save)
    torch.save(corpus.vocab, config.vocab_save)


if __name__ == '__main__':
    parser = add_train_args(get_parser())
    config = parser.parse_known_args()[0]
    main(config)
