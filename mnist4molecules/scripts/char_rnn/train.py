import torch

from mnist4molecules.char_rnn.config import get_parser
from mnist4molecules.char_rnn.datautils import OneHotCorpus
from mnist4molecules.char_rnn.model import CharRNN
from mnist4molecules.char_rnn.trainer import CharRNNTrainer
from mnist4molecules.script_utils import add_train_args, read_smiles_csv, set_seed


def main(config):
    set_seed(config.seed)

    train = read_smiles_csv(config.train_load)

    print(config.device)

    device = torch.device(config.device)

    corpus = OneHotCorpus(config.batch, device)

    val_dataloader = None
    train_dataloader = corpus.fit(train).transform(train)

    if config.val_load is not None:
        val = read_smiles_csv(config.val_load)
        val_dataloader = corpus.transform(val)

    model = CharRNN(corpus.vocab, config.hidden, config.num_layers, config.dropout, device).to(device)

    # Serialization
    torch.save(config, config.config_save)
    torch.save(corpus.vocab, config.vocab_save)

    trainer = CharRNNTrainer(config)

    if config.val_load is not None:
        trainer.fit(model, (train_dataloader, val_dataloader))
    else:
        trainer.fit(model, train_dataloader)


if __name__ == '__main__':
    parser = add_train_args(get_parser())
    config = parser.parse_known_args()[0]
    main(config)
