import torch

from moses.vae.model import VAE
from moses.vae.trainer import VAETrainer
from moses.vae.config import get_parser as vae_parser
from moses.script_utils import add_train_args, read_smiles_csv, set_seed


def get_parser():
    return add_train_args(vae_parser())


def main(config):
    set_seed(config.seed)
    device = torch.device(config.device)
    # For CUDNN to work properly:
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device.index or 0)

    train_data = read_smiles_csv(config.train_load)
    trainer = VAETrainer(config)

    if config.vocab_load is not None:
        assert os.path.exists(config.vocab_load), 'vocab_load path doesn\'t exist!'
        vocab = torch.load(config.vocab_load)
    else:
        vocab = trainer.get_vocabulary(train_data)

    model = VAE(vocab, config).to(device)
    trainer.fit(model, train_data)

    model = model.to('cpu')
    torch.save(model.state_dict(), config.model_save)
    if config.config_save is not None:
        torch.save(config, config.config_save)
    if config.vocab_save is not None:
        torch.save(vocab, config.vocab_save)

if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
