import torch

from moses.char_rnn.config import get_parser as char_rnn_parser
from moses.char_rnn.datautils import OneHotCorpus
from moses.char_rnn.model import CharRNN
from moses.char_rnn.trainer import CharRNNTrainer
from moses.script_utils import add_train_args, read_smiles_csv, set_seed


def get_parser():
    return add_train_args(char_rnn_parser())


def main(config):
    set_seed(config.seed)

    train = read_smiles_csv(config.train_load)

    device = torch.device(config.device)

    corpus = OneHotCorpus(config.batch, device)

    train_dataloader = corpus.fit(train).transform(train)

    model = CharRNN(corpus.vocab, config.hidden, config.num_layers, config.dropout, device).to(device)
    trainer = CharRNNTrainer(config)
    trainer.fit(model, train_dataloader)

    torch.save(model.state_dict(), config.model_save)
    torch.save(config, config.config_save)
    torch.save(corpus.vocab, config.vocab_save)


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
