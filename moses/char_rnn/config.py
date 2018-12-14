import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    # Model
    model_arg = parser.add_argument_group('Model')

    model_arg.add_argument("--val_load", type=str, help="Input data in csv format to validation")
    model_arg.add_argument("--num_layers", type=int, default=3, help="Number of LSTM layers")
    model_arg.add_argument("--hidden", type=int, default=600, help="Hidden size")
    model_arg.add_argument("--dropout", type=float, default=0.2, help="dropout between LSTM layers except for last")
    model_arg.add_argument("--batch", type=int, default=64, help="Batch size")
    model_arg.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    model_arg.add_argument('--lr', type=float, default=0.001, help='Initial lr value')
    model_arg.add_argument('--step_size', type=float, default=50, help='Period of learning rate decay')
    model_arg.add_argument('--gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay')
    model_arg.add_argument('--n_jobs', type=int, default=1, help='Number of threads')
    return parser


def get_config():
    parser = get_parser()
    return parser.parse_known_args()[0]
