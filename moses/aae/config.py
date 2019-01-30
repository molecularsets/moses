import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument('--conditional_model', type=bool, default=False,
                           help='If true to learn given conditions')
    model_arg.add_argument('--embedding_size', type=int, default=32,
                           help='Embedding size in encoder and decoder')
    model_arg.add_argument('--encoder_hidden_size', type=int, default=380,
                           help='Size of hidden state for lstm layers in encoder')
    model_arg.add_argument('--encoder_num_layers', type=int, default=1,
                           help='Number of lstm layers in encoder')
    model_arg.add_argument('--encoder_bidirectional', type=bool, default=True,
                           help='If true to use bidirectional lstm layers in encoder')
    model_arg.add_argument('--encoder_dropout', type=float, default=0,
                           help='Dropout probability for lstm layers in encoder')
    model_arg.add_argument('--decoder_hidden_size', type=int, default=640,
                           help='Size of hidden state for lstm layers in decoder')
    model_arg.add_argument('--decoder_num_layers', type=int, default=2,
                           help='Number of lstm layers in decoder')
    model_arg.add_argument('--decoder_dropout', type=float, default=0,
                           help='Dropout probability for lstm layers in decoder')
    model_arg.add_argument('--latent_size', type=int, default=640,
                           help='Size of latent vectors')
    model_arg.add_argument('--discriminator_layers', nargs='+', type=int, default=[640, 256],
                           help='Numbers of features for linear layers in discriminator')
    model_arg.add_argument('--labels_size', type=int, default=167,
                           help='Size of labels')
    model_arg.add_argument('--labels_embedding_size', type=int, default=50,
                           help='Size of labels after embedding')

    train_arg = parser.add_argument_group('Training')
    train_arg.add_argument('--pretrain_epochs', type=int, default=0,
                           help='Number of epochs for autoencoder pretraining')
    train_arg.add_argument('--train_epochs', type=int, default=20,
                           help='Number of epochs for autoencoder training')
    train_arg.add_argument('--n_batch', type=int, default=128,
                           help='Size of batch')
    train_arg.add_argument('--lr', type=float, default=1e-3,
                           help='Learning rate')
    train_arg.add_argument('--step_size', type=float, default=50,
                           help='Period of learning rate decay')
    train_arg.add_argument('--gamma', type=float, default=0.5,
                           help='Multiplicative factor of learning rate decay')
    train_arg.add_argument('--n_jobs', type=int, default=1,
                           help='Number of threads')
    train_arg.add_argument('--d_threshold', type=float, default=0.5,
                           help='If loss of discriminator lower than the threshold, discriminator stop improving')
    train_arg.add_argument('--g_threshold', type=float, default=0.8,
                           help='If loss of discriminator is higher than the threshold, generator stop improving')

    return parser


def get_config():
    parser = get_parser()
    return parser.parse_known_args()[0]
