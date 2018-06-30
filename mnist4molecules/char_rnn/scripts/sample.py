import pandas as pd
import torch
import tqdm

from mnist4molecules.char_rnn.config import get_sample_parser
from mnist4molecules.char_rnn.model import CharRNN
from mnist4molecules.config import get_config
from mnist4molecules.utils import get_device, set_logger

if __name__ == '__main__':
    torch.manual_seed(0)

    config = get_config(get_sample_parser())
    set_logger(config)

    vocab = torch.load(config.vocab_load)
    new_config = torch.load(config.config_load) + config

    device = get_device(new_config)

    model = CharRNN(vocab, new_config.hidden, new_config.num_layers, new_config.dropout, device)
    model.load_state_dict(torch.load(new_config.model_load))
    model = model.to(device=device)
    model.eval()

    gen_smiles = []

    # TODO: n_samples % batch = 0
    for i in tqdm.tqdm(range(new_config.n_samples // new_config.batch)):
        gen_smiles.extend(vocab.reverse(model.sample_smiles(new_config.max_len, new_config.batch)))

    df = pd.DataFrame(gen_smiles, columns=['SMILES'])
    df.to_csv(new_config.gen_save, index=False)
