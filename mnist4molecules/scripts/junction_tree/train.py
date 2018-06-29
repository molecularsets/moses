import sys
sys.path.insert(0, '..')

import rdkit
import torch
import torch.nn as nn

from mnist4molecules.junction_tree.config import get_parser
from mnist4molecules.junction_tree.datautils import JTreeCorpus
from mnist4molecules.junction_tree.jtnn.jtnn_vae import JTNNVAE
from mnist4molecules.junction_tree.trainer import JTreeTrainer
from utils import add_train_args, read_smiles_csv, set_seed


lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


def main(config):
    set_seed(config.seed)

    device = torch.device(config.device)

    data = read_smiles_csv(config.train_load)
    corpus = JTreeCorpus(config.n_batch, device)

    if config.vocab_load is not None:
        vocab = torch.load(config.vocab_load)
        train_dataloader = corpus.fit(vocabulary=vocab).transform(data)
    else:
        train_dataloader = corpus.fit(dataset=data).transform(data)

    model = JTNNVAE(corpus.vocab, config.hidden, config.latent, config.depth)
    model = model.to(device=device)

    # if jt_config.model_load is not None:
    #     model.load_state_dict(torch.load(jt_config.model_load)) # TODO
    # else:

    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    # Serialization
    torch.save(corpus.vocab, config.vocab_save)
    torch.save(config, config.config_save)

    trainer = JTreeTrainer(config)
    trainer.fit(model, train_dataloader)


if __name__ == '__main__':
    parser = add_train_args(get_parser())
    config = parser.parse_known_args()[0]
    main(config)
