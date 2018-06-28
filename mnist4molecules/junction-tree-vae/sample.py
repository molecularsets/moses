import pandas as pd
import rdkit
import torch
import tqdm

from config import get_sample_parser
from jtnn.jtnn_vae import JTNNVAE
from mnist4molecules.config import get_config
from mnist4molecules.utils import get_device

torch.manual_seed(0)

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

jt_config = get_config(get_sample_parser())

vocab = torch.load(jt_config.vocab_load)
new_jt_config = torch.load(jt_config.config_load) + jt_config

device = get_device(new_jt_config)

model = JTNNVAE(vocab, new_jt_config.hidden, new_jt_config.latent, new_jt_config.depth)
model.load_state_dict(torch.load(new_jt_config.model_load))
model = model.to(device=device)

gen_smiles = []
for i in tqdm.tqdm(range(new_jt_config.n_samples)):
    gen_smiles.append(model.sample_prior())

df = pd.DataFrame(gen_smiles, columns=['SMILES'])
df.to_csv(new_jt_config.gen_save, index=False)
