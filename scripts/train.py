import argparse
import os
import sys
import torch
import rdkit

from moses.vae import VAE, VAETrainer, get_parser as vae_parser
from moses.organ import ORGAN, ORGANTrainer, get_parser as organ_parser
from moses.aae import AAE, AAETrainer, get_parser as aae_parser
from moses.char_rnn import CharRNN, CharRNNTrainer, get_parser as char_rnn_parser
from moses.junction_tree import JTNNVAE, JTreeTrainer, get_parser as junction_tree_parser

from moses.script_utils import add_train_args, read_smiles_csv, set_seed

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

MODELS = {}

def add_model(name, class_, trainer_, parser_):
    MODELS[name] = { 'class' : class_,
                     'trainer' : trainer_,
                     'parser' : parser_,}

def init_models():
    add_model('aae', AAE, AAETrainer, aae_parser)
    add_model('char_rnn', CharRNN, CharRNNTrainer, char_rnn_parser)
    add_model('junction_tree', JTNNVAE, JTreeTrainer, junction_tree_parser)
    add_model('vae', VAE, VAETrainer, vae_parser)
    add_model('organ', ORGAN, ORGANTrainer, organ_parser)

def get_model_names():
    return MODELS.keys()

def get_model_trainer(model):
    return MODELS[model]['trainer']

def get_model_class(model):
    return MODELS[model]['class']

def get_model_parser(model):
    return MODELS[model]['parser']

def get_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Models trainer script', description='available models')
    for model in get_model_names():
        add_train_args(get_model_parser(model)(subparsers.add_parser(model)))
    return parser


def main(model, config):
    set_seed(config.seed)
    device = torch.device(config.device)
    # For CUDNN to work properly:
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device.index or 0)

    train_data = read_smiles_csv(config.train_load)
    trainer = get_model_trainer(model)(config)

    if config.vocab_load is not None:
        assert os.path.exists(config.vocab_load), 'vocab_load path doesn\'t exist!'
        vocab = torch.load(config.vocab_load)
    else:
        vocab = trainer.get_vocabulary(train_data)

    model = get_model_class(model)(vocab, config).to(device)
    trainer.fit(model, train_data)

    model = model.to('cpu')
    torch.save(model.state_dict(), config.model_save)
    if config.config_save is not None:
        torch.save(config, config.config_save)
    if config.vocab_save is not None:
        torch.save(vocab, config.vocab_save)

if __name__ == '__main__':
    init_models()
    parser = get_parser()
    config = parser.parse_known_args()[0]
    model = sys.argv[1]
    main(model, config)
