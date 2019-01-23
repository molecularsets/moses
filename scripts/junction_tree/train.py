import os
import torch
import rdkit

from moses.junction_tree import JTNNVAE, JTreeTrainer, get_parser as junction_tree_parser
from moses.script_utils import add_train_args, read_smiles_csv, set_seed

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

def get_parser():
    return add_train_args(junction_tree_parser())


def main(config):
    set_seed(config.seed)
    device = torch.device(config.device)

    train_data = read_smiles_csv(config.train_load)
    trainer = JTreeTrainer(config)

    if config.vocab_load is not None:
        assert os.path.exists(config.vocab_load), 'vocab_load path doesn\'t exist!'
        vocab = torch.load(config.vocab_load)
    else:
        vocab = trainer.get_vocabulary(train_data)

    model = JTNNVAE(vocab, config).to(device)

    trainer.fit(model, train_data)

    model = model.to('cpu')
    torch.save(model.state_dict(), config.model_save)
    torch.save(config, config.config_save)
    if config.vocab_save is not None:
        torch.save(vocab, config.vocab_save)

if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
