import torch

from mnist4molecules.aae import get_parser, AAE, AAETrainer
from mnist4molecules.script_utils import add_train_args, read_smiles_csv, set_seed
from mnist4molecules.utils import CharVocab


def main(config):
    set_seed(config.seed)

    train = read_smiles_csv(config.train_load)

    vocab = CharVocab.from_data(train)

    device = torch.device(config.device)

    model = AAE(vocab, config)
    model = model.to(device)

    trainer = AAETrainer(config)
    trainer.fit(model, train)

    torch.save(model.state_dict(), config.model_save)
    torch.save(config, config.config_save)
    torch.save(vocab, config.vocab_save)


if __name__ == '__main__':
    parser = add_train_args(get_parser())
    config = parser.parse_known_args()[0]
    main(config)