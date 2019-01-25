import argparse

import pandas as pd
import torch
import tqdm

from moses.script_utils import add_sample_args, set_seed
from moses.cvae.model import VAE


def get_parser():

    parser = add_sample_args(argparse.ArgumentParser())
    parser.add_argument('--condition_load', type=str, default='molhack_test.solution',
                       help='Target conditional input data in csv format to train')
    parser.add_argument('--output_size', type=int, default=10,
                       help='Output size in the condition linear layer')
    return parser

# read fingerrprints
def read_fps_csv(path):
    return pd.read_csv(path,
                       usecols=['fingerprints_center'],
                       squeeze=True).astype(str).tolist()

# convert fingerprints to list
def fps_to_list(fps):
    fps = [list(x) for x in fps]
    for i in tqdm(range(len(fps))):
        fps[i] = [int(x) for x in fps[i]]
    return fps

# check smiles validity
def isValid(smile_str):
    try:
        m = Chem.MolFromSmiles(smile_str)
        if m:
            return 1 
        else:
            return 0
    except:
        return 0

# convert smiles to maccs fingerprints
def smi_to_maccs(smi):
    MACCS_SIZE = 167
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return np.array(MACCSkeys.GenMACCSKeys(mol))
    else:
        return np.zeros(MACCS_SIZE, dtype=int)

# calculate tanimoto similarity
def Tanimoto(l1,l2):
    a = sum(l1)
    b = sum(l2)
    c = sum([l1[i]&l2[i] for i in range(len(l1))])
    return c/(a+b-c)

# calculate score
def calculate_score(smi_file, fps_file, n_samples):
    smiles = pd.read_csv(smi_file)
    smiles = smiles['SMILES'].tolist()
    targets = read_fps_csv(fps_file)
    
    # parameters
    cutoff = 0.4
    top_seleted = 100

    total_sims = []
    top_sims = []
    validity = []

    start = 0
    end = n_samples

    for target in targets:
        
        target_fps = [int(t) for t in target]
        
        smiles_lst = []
        sims = []

        for i in range(start, end):
            # check valid smiles
            if isValid(smiles[i]):
                smi = Chem.MolToSmiles(Chem.MolFromSmiles(smiles[i]))
                smiles_lst.append(smi)

        validity.append(len(smiles_lst))

        # calculate tanimoto similarity
        for i in range(len(smiles_lst)):
            fps = smi_to_maccs(smiles_lst[i])
            sims.append(Tanimoto(fps, target_fps))

        # sort similarity
        sims = sorted(sims, reverse=True)

        # mean of top 100 similarity
        top_sims = sims[:top_seleted]
        top_sims.append(sum(top_sims)/top_seleted)

        # mean of total similarity
        total_sims = sims[top_seleted:]
        total_sims.append(sum(sims)/(n_samples-top_seleted))

        # iteration
        start += n_samples
        end += n_samples
    
    # summary
    print('======= Results =======')
    print('Valid molecules: ' + str(validity))
    print('Mean of valid molecules: ' + str(np.mean(validity)))

    # final score
    print('======= Score =======')
    top_mean = np.mean(top_sims)
    print('Mean tanimoto similarity of top 100 unique molecules: ' + str(top_mean))
    mean = np.mean(total_sims)
    print('Mean tanimoto similarity of unique generated molecules: ' + str(mean))
    
    score = 0.7 * top_mean + 0.3 * mean
    print('The final score: ' + str(score))


def main(config):
    set_seed(config.seed)

    model_config = torch.load(config.config_load)
    model_vocab = torch.load(config.vocab_load)
    model_state = torch.load(config.model_load)

    device = torch.device(config.device)

    # For CUDNN to work properly:
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device.index or 0)

    # target fingerprints
    fps_center = read_fps_csv(config.condition_load)
    fps_center = fps_to_list(list(set(fps_center)))
    fps_center = [torch.tensor(f, dtype=torch.float, device=device) for f in fps_center]
    fps_len = len(fps_center[0])
    fps_num = len(fps_center)

    model = VAE(model_vocab, fps_len, model_config)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()

    # sample numbers
    gen, n = [], config.n_samples
    gen_samples = []

    # sample
    # T = tqdm.tqdm(range(config.n_samples), desc='Generating mols')
    T = tqdm(range(config.n_samples), desc='Generating mols')

    for i in range(fps_num):
        gen, n = [], config.n_samples
        while n > 0:
            x = model.sample(fps_center[i], min(n, config.n_batch), config.max_len)[-1]
            mols = [model_vocab.ids2string(i_x.tolist()) for i_x in x]
            n -= len(mols)
            T.update(len(mols))
            T.refresh()
            gen.extend(mols)

        gen_samples.extend(gen)

    df = pd.DataFrame(gen_samples, columns=['SMILES'])
    df.to_csv(config.gen_save, index=False)

    # tanimoto similarity score and summary
    calculate_score(config.gen_save, config.condition_load, config.n_samples)


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
