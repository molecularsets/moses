import argparse


def get_parser():
    def restricted_float(arg):
        if float(arg) < 0 or float(arg) > 1:
            raise argparse.ArgumentTypeError('{} not in range [0, 1]'.format(arg))

        return arg

    def conv_pair(arg):
        if arg[0] != '(' or arg[-1] != ')':
            raise argparse.ArgumentTypeError('Wrong pair: {}'.format(arg))

        feats, kernel_size = arg[1:-1].split(',')
        feats, kernel_size = int(feats), int(kernel_size)

        return feats, kernel_size

    parser = argparse.ArgumentParser()

    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument('--embedding_size', type=int, default=32,
                           help='Embedding size in generator and discriminator')
    model_arg.add_argument('--hidden_size', type=int, default=512,
                           help='Size of hidden state for lstm layers in generator')
    model_arg.add_argument('--num_layers', type=int, default=2,
                           help='Number of lstm layers in generator')
    model_arg.add_argument('--dropout', type=float, default=0,
                           help='Dropout probability for lstm layers in generator')
    model_arg.add_argument('--discriminator_layers', nargs='+', type=conv_pair,
                           default=[(100, 1), (200, 2), (200, 3), (200, 4), (200, 5), (100, 6),
                                    (100, 7), (100, 8), (100, 9), (100, 10), (160, 15), (160, 20)],
                           help='Numbers of features for convalution layers in discriminator')
    model_arg.add_argument('--discriminator_dropout', type=float, default=0,
                           help='Dropout probability for discriminator')
    model_arg.add_argument('--reward_weight', type=restricted_float, default=0.7,
                           help='Reward weight for policy gradient training')

    train_arg = parser.add_argument_group('Training')
    train_arg.add_argument('--generator_pretrain_epochs', type=int, default=300,
                           help='Number of epochs for generator pretraining')
    train_arg.add_argument('--discriminator_pretrain_epochs', type=int, default=300,
                           help='Number of epochs for discriminator pretraining')
    train_arg.add_argument('--pg_iters', type=int, default=1000,
                           help='Number of inerations for policy gradient training')
    train_arg.add_argument('--n_batch', type=int, default=64,
                           help='Size of batch')
    train_arg.add_argument('--lr', type=float, default=1e-4,
                           help='Learning rate')
    train_arg.add_argument('--n_jobs', type=int, default=8, help='Number of threads')
    train_arg.add_argument('--max_length', type=int, default=100,
                           help='Maximum length for sequence')
    train_arg.add_argument('--rollouts', type=int, default=16,
                           help='Number of rollouts')
    train_arg.add_argument('--generator_updates', type=int, default=1,
                           help='Number of updates of generator per iteration')
    train_arg.add_argument('--discriminator_updates', type=int, default=1,
                           help='Number of updates of discriminator per iteration')
    train_arg.add_argument('--discriminator_epochs', type=int, default=10,
                           help='Number of epochs of discriminator per iteration')

    return parser


def get_config():
    parser = get_parser()
    return parser.parse_known_args()[0]
