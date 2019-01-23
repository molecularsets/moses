import argparse


def get_parser(parser=None):
    if parser == None:
        parser = argparse.ArgumentParser()

    # Model
    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument('--hidden', type=int, default=450,
                           help='Hidden size')
    model_arg.add_argument('--latent', type=int, default=56,
                           help='Latent size')
    model_arg.add_argument('--depth', type=int, default=3,
                           help='Depth of graph message passing')
    model_arg.add_argument('--kl_start', type=int, default=1,
                           help='Epoch to init KL weight (start from 0)')
    model_arg.add_argument('--kl_w', type=float, default=5e-3,
                           help='KL weight value')

    # Train
    train_arg = parser.add_argument_group('Training')
    train_arg.add_argument('--train_epochs', type=int, default=5,
                           help='Number of epochs for model training')
    train_arg.add_argument('--n_batch', type=int, default=40,
                           help='Size of batch')
    train_arg.add_argument('--lr', type=float, default=1e-3,
                           help='Learning rate')
    train_arg.add_argument('--n_jobs', type=int, default=1,
                           help='Number of threads')
    train_arg.add_argument('--n_workers', type=int, default=1,
                           help='Number of workers for DataLoaders')

    return parser


def get_config():
    parser = get_parser()
    return parser.parse_known_args()[0]
