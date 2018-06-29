import pandas as pd
import torch
import tqdm

from mnist4molecules.config import get_config
from mnist4molecules.utils import get_device, set_logger
from mnist4molecules.vae.config import get_sample_parser
from mnist4molecules.vae.model import VAE

if __name__ == '__main__':
    config = get_config(get_sample_parser())
    set_logger(config)
    config = torch.load(config.config_load) + config  # Right merge
    vocab = torch.load(config.vocab_load)

    device = get_device(config)
    model = VAE(vocab, config)
    model.load_state_dict(torch.load(config.model_load))
    model = model.to(device)

    gen, n = [], config.n_samples
    T = tqdm.tqdm(range(config.n_samples), desc='Generating mols')
    while n > 0:
        x = model.sample(min(n, config.n_batch), config.n_len)[-1]
        mols = vocab.reverse(x)
        n -= len(mols)
        T.update(len(mols))
        T.refresh()
        gen.extend(mols)

    df = pd.DataFrame(gen, columns=['SMILES'])
    df.to_csv(config.gen_save, index=False)
