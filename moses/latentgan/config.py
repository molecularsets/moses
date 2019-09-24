import argparse


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # Model
    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument("--heteroencoder_version", type=str, default='chembl',
                           help="Which heteroencoder model version to use")
    # Train
    train_arg = parser.add_argument_group('Training')

    train_arg.add_argument('--gp', type=int, default=10, help='Gradient Penalty Coefficient')
    train_arg.add_argument('--n_critic', type=int, default=5,
                           help='ratio of discriminator to generator training frequency')
    train_arg.add_argument('--train_epochs', type=int, default=2000,
                           help='Number of epochs for model training')
    train_arg.add_argument('--n_batch', type=int, default=64,
                           help='Size of batch')
    train_arg.add_argument('--lr', type=float, default=0.0002,
                           help='Learning rate')
    train_arg.add_argument('--b1', type=float, default=0.5,
                           help='Adam optimizer parameter beta 1')
    train_arg.add_argument('--b2', type=float, default=0.999,
                           help='Adam optimizer parameter beta 2')
    train_arg.add_argument('--step_size', type=int, default=10,
                           help='Period of learning rate decay')
    train_arg.add_argument('--gamma', type=float, default=1,
                           help='Multiplicative factor of learning rate decay')
    train_arg.add_argument('--n_jobs', type=int, default=1,
                           help='Number of threads')
    train_arg.add_argument('--n_workers', type=int, default=1,
                           help='Number of workers for DataLoaders')

    return parser


def get_config():
    parser = get_parser()
    return parser.parse_known_args()[0]
