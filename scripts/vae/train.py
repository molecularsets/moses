import torch

from moses.script_utils import add_train_args, read_smiles_csv, set_seed
from moses.cvae.config import get_parser as vae_parser
from moses.cvae.corpus import OneHotCorpus
from moses.cvae.model import VAE
from moses.cvae.trainer import VAETrainer


def get_parser():

    parser = add_train_args(vae_parser())

    # conditional generation
    parser.add_argument('--conditional', type=int, default=0,
                       help='Conditional generation mode')
    parser.add_argument('--output_size', type=int, default=10,
                       help='Output size in the condition linear layer')

    return parser

# read fingerprints
def read_fps_csv(path):
    return pd.read_csv(path,
                       usecols=['fingerprints_center'],
                       squeeze=True).astype(str).tolist()

# convert fingerprints to list
def fps_to_list(fps):
    fps = [list(x) for x in fps]
    for i in range(len(fps)):
        fps[i] = [int(x) for x in fps[i]]
    return fps


def main(config):
    set_seed(config.seed)

    train = read_smiles_csv(config.train_load)

    device = torch.device(config.device)

    # For CUDNN to work properly:
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device.index or 0)

    corpus = OneHotCorpus(config.n_batch, device)
    train = corpus.fit(train).transform(train)


    # condition mode
    if config.conditional:
        fps = read_fps_csv(config.train_load)
        fps = fps_to_list(fps)
        fps = [torch.tensor(f, dtype=torch.float, device=device) for f in fps]

        # fingerprints length
        fps_len = len(fps[0])

        # fingerprints dataloader
        fps = corpus.fps_transform(fps)

        # training data
        train = zip(train, fps)
        shuffle(train)
    else:
        fps_len = 0


    model = VAE(corpus.vocab, fps_len, config).to(device)

    trainer = VAETrainer(config)

    torch.save(config, config.config_save)
    torch.save(corpus.vocab, config.vocab_save)
    trainer.fit(model, train, config.conditional)


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
