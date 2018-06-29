import pandas as pd
import rdkit
import torch
import tqdm

from mnist4molecules.config import get_config
from mnist4molecules.junction_tree.config import get_sample_parser
from mnist4molecules.junction_tree.jtnn.jtnn_vae import JTNNVAE
from mnist4molecules.utils import get_device, set_logger

if __name__ == '__main__':

    torch.manual_seed(0)

    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    config = get_config(get_sample_parser())
    set_logger(config)

    vocab = torch.load(config.vocab_load)
    new_config = torch.load(config.config_load) + config

    device = get_device(new_config)

    model = JTNNVAE(vocab, new_config.hidden, new_config.latent, new_config.depth)
    model.load_state_dict(torch.load(new_config.model_load))
    model = model.to(device=device)

    gen_smiles = []
    for i in tqdm.tqdm(range(new_config.n_samples)):
        gen_smiles.append(model.sample_prior())

    df = pd.DataFrame(gen_smiles, columns=['SMILES'])
    df.to_csv(new_config.gen_save, index=False)
