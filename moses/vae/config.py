import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    # Model
    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument('--q_cell',
                           type=str, default='gru', choices=['gru'],
                           help='Encoder rnn cell type')
    model_arg.add_argument('--q_bidir',
                           type=bool, default=True,
                           help='If to add second direction to encoder')
    model_arg.add_argument('--q_d_h',
                           type=int, default=128,
                           help='Encoder h dimensionality')
    model_arg.add_argument('--q_n_layers',
                           type=int, default=1,
                           help='Encoder number of layers')
    model_arg.add_argument('--q_dropout',
                           type=float, default=0.0,
                           help='Encoder layers dropout')
    model_arg.add_argument('--d_cell',
                           type=str, default='gru', choices=['gru'],
                           help='Decoder rnn cell type')
    model_arg.add_argument('--d_n_layers',
                           type=int, default=3,
                           help='Decoder number of layers')
    model_arg.add_argument('--d_dropout',
                           type=float, default=0.2,
                           help='Decoder layers dropout')
    model_arg.add_argument('--d_z',
                           type=int, default=128,
                           help='Latent vector dimensionality')
    model_arg.add_argument('--d_d_h',
                           type=int, default=512,
                           help='Latent vector dimensionality')
    model_arg.add_argument('--freeze_embeddings',
                           type=bool, default=False,
                           help='If to freeze embeddings while training')

    # Train
    train_arg = parser.add_argument_group('Train')
    train_arg.add_argument('--n_batch',
                           type=int, default=128,
                           help='Batch size')
    train_arg.add_argument('--grad_clipping',
                           type=int, default=50,
                           help='Gradients clipping size')
    train_arg.add_argument('--kl_start',
                           type=int, default=0,
                           help='Epoch to start change kl weight from')
    train_arg.add_argument('--kl_w_start',
                           type=float, default=1,
                           help='Initial kl weight value')
    train_arg.add_argument('--kl_w_end',
                           type=float, default=1,
                           help='Maximum kl weight value')
    train_arg.add_argument('--lr_start',
                           type=float, default=3 * 1e-4,
                           help='Initial lr value')
    train_arg.add_argument('--lr_n_period',
                           type=int, default=50,
                           help='Epochs before first restart in SGDR')
    train_arg.add_argument('--lr_n_restarts',
                           type=int, default=1,
                           help='Number of restarts in SGDR')
    train_arg.add_argument('--lr_n_mult',
                           type=int, default=1,
                           help='Mult coefficient after restart in SGDR')
    train_arg.add_argument('--lr_end',
                           type=float, default=3 * 1e-4,
                           help='Maximum lr weight value')
    train_arg.add_argument('--n_last',
                           type=int, default=1000,
                           help='Number of iters to smooth loss calc')
    train_arg.add_argument('--n_jobs', type=int, default=1,
                           help='Number of threads')
    return parser


def get_config():
    parser = get_parser()
    return parser.parse_known_args()[0]
