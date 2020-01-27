import argparse


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # Model
    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument("--heteroencoder_version",
                           type=str, default='moses',
                           help="Which heteroencoder model version to use")
    # Train
    train_arg = parser.add_argument_group('Training')

    train_arg.add_argument('--gp', type=int, default=10,
                           help='Gradient Penalty Coefficient')
    train_arg.add_argument('--n_critic', type=int, default=5,
                           help='Ratio of discriminator to'
                                ' generator training frequency')
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
    train_arg.add_argument('--latent_vector_dim', type=int, default=512,
                           help='Size of latentgan vector')
    train_arg.add_argument('--gamma', type=float, default=1,
                           help='Multiplicative factor of'
                                ' learning rate decay')
    train_arg.add_argument('--n_jobs', type=int, default=1,
                           help='Number of threads')
    train_arg.add_argument('--n_workers', type=int, default=1,
                           help='Number of workers for DataLoaders')

    # Arguments used if training a new heteroencoder
    heteroencoder_arg = parser.add_argument_group('heteroencoder')

    heteroencoder_arg.add_argument('--heteroencoder_layer_dim', type=int,
                                   default=512,
                                   help='Layer size for heteroencoder '
                                        '(if training new heteroencoder)')
    heteroencoder_arg.add_argument('--heteroencoder_noise_std', type=float,
                                   default=0.1,
                                   help='Noise amplitude for heteroencoder')
    heteroencoder_arg.add_argument('--heteroencoder_dec_layers', type=int,
                                   default=4,
                                   help='Number of decoding layers'
                                        ' for heteroencoder')
    heteroencoder_arg.add_argument('--heteroencoder_batch_size',
                                   type=int, default=128,
                                   help='Batch size for heteroencoder')
    heteroencoder_arg.add_argument('--heteroencoder_epochs', type=int,
                                   default=100,
                                   help='Number of epochs for heteroencoder')
    heteroencoder_arg.add_argument('--heteroencoder_lr', type=float,
                                   default=1e-3,
                                   help='learning rate for heteroencoder')
    heteroencoder_arg.add_argument('--heteroencoder_mini_epochs', type=int,
                                   default=10,
                                   help='How many sub-epochs to '
                                        'split each epoch for heteroencoder')
    heteroencoder_arg.add_argument('--heteroencoder_lr_decay',
                                   default=True, action='store_false',
                                   help='Use learning rate decay '
                                        'for heteroencoder ')
    heteroencoder_arg.add_argument('--heteroencoder_patience', type=int,
                                   default=100,
                                   help='Patience for adaptive learning '
                                        'rate for heteroencoder')
    heteroencoder_arg.add_argument('--heteroencoder_lr_decay_start', type=int,
                                   default=500,
                                   help='Which sub-epoch to start decaying '
                                        'learning rate for heteroencoder ')
    heteroencoder_arg.add_argument('--heteroencoder_save_period', type=int,
                                   default=100,
                                   help='How often in sub-epochs to '
                                        'save model checkpoints for'
                                        ' heteroencoder')

    return parser


def get_config():
    parser = get_parser()
    return parser.parse_known_args()[0]
