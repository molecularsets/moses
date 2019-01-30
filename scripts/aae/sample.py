import argparse

import pandas as pd
import numpy as np
import torch
import tqdm
from rdkit import Chem
from rdkit.Chem import MACCSkeys

from moses.aae import AAE
from moses.script_utils import add_sample_args, set_seed


def get_parser():
    return add_sample_args(argparse.ArgumentParser())

def main(config):
    set_seed(config.seed)

    model_config = torch.load(config.config_load)
    model_vocab = torch.load(config.vocab_load)
    model_state = torch.load(config.model_load)

    device = torch.device(config.device)

    model = AAE(model_vocab, model_config)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()

    if model_config.conditional_model:
        test = pd.read_csv(config.label_load,usecols=['fingerprints_center'],squeeze=True).astype(str).tolist()
        labels = [[int(x) for x in list(t)] for t in test]
        labels = np.array(labels)
        labels = torch.FloatTensor(labels).cuda()
    else: labels = None

    samples = []
    n = config.n_samples
    n_labels = config.n_labels
    with tqdm.tqdm(total=config.n_samples*n_labels, desc='Generating samples') as T:
        while n > 0:
            current_samples = model.sample(n_labels, config.max_len, labels)
            samples.append(current_samples)
            n-=1
            T.update(n_labels)

    samples = np.transpose(np.array(samples)).tolist()
    output = open(config.gen_save,'w')
    output.write('SMILES\n')
    for i in range(len(samples)):
        for j in range(len(samples[0])):
            output.write('{0}\n'.format(samples[i][j]))
    output.close()


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
