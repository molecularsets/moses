import rdkit
import torch
import torch.nn as nn

from moses.junction_tree.config import get_parser as junction_tree_parser
from moses.junction_tree.datautils import JTreeCorpus
from moses.junction_tree.jtnn.jtnn_vae import JTNNVAE
from moses.junction_tree.trainer import JTreeTrainer
from moses.script_utils import add_train_args, read_smiles_csv, set_seed

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


def get_parser():
    return add_train_args(junction_tree_parser())


def main(config):
    set_seed(config.seed)

    device = torch.device(config.device)

    data = read_smiles_csv(config.train_load)
    corpus = JTreeCorpus(config.n_batch, device)
    train_dataloader = corpus.fit(dataset=data).transform(data)

    model = JTNNVAE(corpus.vocab, config.hidden, config.latent, config.depth)
    model = model.to(device=device)

    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    trainer = JTreeTrainer(config)
    trainer.fit(model, train_dataloader)

    torch.save(model.state_dict(), config.model_save)
    torch.save(corpus.vocab, config.vocab_save)
    torch.save(config, config.config_save)


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
