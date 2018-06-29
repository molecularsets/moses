import pandas as pd
import rdkit
import torch
import torch.nn as nn

from mnist4molecules.config import get_config
from mnist4molecules.junction_tree.config import get_train_parser
from mnist4molecules.junction_tree.datautils import JTreeCorpus
from mnist4molecules.junction_tree.jtnn.jtnn_vae import JTNNVAE
from mnist4molecules.junction_tree.trainer import JTreeTrainer
from mnist4molecules.utils import PandasDataset, get_device, set_logger

if __name__ == '__main__':

    torch.manual_seed(0)

    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    config = get_config(get_train_parser())
    set_logger(config)

    device = get_device(config)

    data = pd.read_csv(config.train_load, usecols=['SMILES'])
    corpus = JTreeCorpus(config.batch, device)
    train_dataset = PandasDataset(data)

    if config.vocab_load is not None:
        vocab = torch.load(config.vocab_load)
        train_dataloader = corpus.fit(vocabulary=vocab).transform(train_dataset)
    else:
        train_dataloader = corpus.fit(dataset=train_dataset).transform(train_dataset)

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
