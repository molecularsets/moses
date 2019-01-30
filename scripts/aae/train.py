import torch
from random import shuffle

from moses.aae import AAE, AAETrainer, get_parser as aae_parser
from moses.script_utils import add_train_args, read_smiles_csv, read_label_csv, set_seed
from moses.utils import CharVocab


def get_parser():
    return add_train_args(aae_parser())


def main(config):
    set_seed(config.seed)

    train = read_smiles_csv(config.train_load)
    if config.conditional_model:
        labels = read_label_csv(config.train_load)
        config.labels_size = len(labels[0])
        labels = [[int(x) for x in list(l)] for l in labels]
        train_data = [(x, y) for (x,y) in zip(train, labels)]
    else:
        train_data = [(x) for x in train]
    shuffle(train_data)
    train_data = train_data[:500000]
    vocab = CharVocab.from_data(train)
    torch.save(config, config.config_save)
    torch.save(vocab, config.vocab_save)

    device = torch.device(config.device)

    model = AAE(vocab, config)
    model = model.to(device)

    trainer = AAETrainer(config)
    trainer.fit(model, train_data)

    model.to('cpu')
    torch.save(model.state_dict(), config.model_save)


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
