import pandas as pd
import rdkit
import torch
import torch.nn as nn

from config import get_train_parser
from datautils import JTreeCorpus
from jtnn.jtnn_vae import JTNNVAE
from mnist4molecules.config import get_config
from mnist4molecules.utils import PandasDataset, get_device
from trainer import JTreeTrainer

torch.manual_seed(0)

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

jt_config = get_config(get_train_parser())

device = get_device(jt_config)

data = pd.read_csv(jt_config.train_load, usecols=['SMILES'])
corpus = JTreeCorpus(jt_config.batch, device)
train_dataset = PandasDataset(data)

if jt_config.vocab_load is not None:
    vocab = torch.load(jt_config.vocab_load)
    train_dataloader = corpus.fit(vocabulary=vocab).transform(train_dataset)
else:
    train_dataloader = corpus.fit(dataset=train_dataset).transform(train_dataset)

model = JTNNVAE(corpus.vocab, jt_config.hidden, jt_config.latent, jt_config.depth)
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
torch.save(corpus.vocab, jt_config.vocab_save)
torch.save(jt_config, jt_config.config_save)

trainer = JTreeTrainer(jt_config)
trainer.fit(model, train_dataloader)
