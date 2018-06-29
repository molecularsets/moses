import argparse

from attrdict import AttrDict


def get_base_parser():
    parser = argparse.ArgumentParser()

    # Base
    base_arg = parser.add_argument_group('Base')
    base_arg.add_argument('--device_code',
                          type=int, default=0,
                          help='Device code to run (-1 for cpu)')
    base_arg.add_argument('--log_level',
                          type=str, default='info',
                          choices=[
                              'notset', 'debug', 'info',
                              'warning', 'error', 'critical'
                          ],
                          help='Logging level to use')

    return parser


def get_train_parser():
    parser = get_base_parser()

    # Common
    common_arg = parser.add_argument_group('Common')
    common_arg.add_argument('--train_load',
                            type=str, required=True,
                            help='Input data in csv format to train')
    common_arg.add_argument('--model_save',
                            type=str, default='model.pt',
                            help='Where to save the model')
    common_arg.add_argument('--config_save',
                            type=str, default='config.pt',
                            help='Where to save the config')
    common_arg.add_argument('--vocab_save',
                            type=str, default='vocab.pt',
                            help='Where to save the vocab')

    return parser


def get_sample_parser():
    parser = get_base_parser()

    # Common
    common_arg = parser.add_argument_group('Common')
    common_arg.add_argument('--model_load',
                            type=str, default='model.pt',
                            help='Where to load the model')
    common_arg.add_argument('--config_load',
                            type=str, default='config.pt',
                            help='Where to load the config')
    common_arg.add_argument('--vocab_load',
                            type=str, default='vocab.pt',
                            help='Where to load the vocab')
    common_arg.add_argument('--n_samples',
                            type=int, required=True,
                            help='Number of samples to sample')
    common_arg.add_argument('--gen_save',
                            type=str, default='gen.csv',
                            help='Where to save the gen molecules')

    return parser


def get_config(parser):
    return AttrDict(parser.parse_known_args()[0].__dict__)
