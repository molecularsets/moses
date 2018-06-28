import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument('--embedding-size', type=int, default=16,
                           help='Embedding size in encoder and decoder')
    model_arg.add_argument('--encoder-hidden-size', type=int, default=64,
                           help='Size of hidden state for lstm layers in encoder')
    model_arg.add_argument('--encoder-num-layers', type=int, default=1,
                           help='Number of lstm layers in encoder')
    model_arg.add_argument('--encoder-bidirectional', type=bool, default=True,
                           help='If true to use bidirectional lstm layers in encoder')
    model_arg.add_argument('--encoder-dropout', type=float, default=0,
                           help='Dropout probability for lstm layers in encoder')
    model_arg.add_argument('--decoder-hidden-size', type=int, default=64,
                           help='Size of hidden state for lstm layers in decoder')
    model_arg.add_argument('--decoder-num-layers', type=int, default=1,
                           help='Number of lstm layers in decoder')
    model_arg.add_argument('--decoder-dropout', type=float, default=0,
                           help='Dropout probability for lstm layers in decoder')
    model_arg.add_argument('--latent-size', type=int, default=8,
                           help='Size of latent vectors')
    model_arg.add_argument('--discriminator-layers', nargs='+', type=int, default=[128, 64],
                           help='Numbers of features for linear layers in discriminator')


    train_arg = parser.add_argument_group('Training')
    train_arg.add_argument('--pretrain-epochs', type=int, default=10,
                           help='Number of epochs for autoencoder pretraining')
    train_arg.add_argument('--train-epochs', type=int, default=10,
                           help='Number of epochs for autoencoder training')
    train_arg.add_argument('--batch-size', type=int, default=64,
                           help='Size of batch')
    train_arg.add_argument('--lr', type=float, default=1e-2,
                           help='Learning rate')

    return parser


def get_config():
    parser = get_parser()
    return parser.parse_known_args()[0]