import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    # Model
    model_arg = parser.add_argument_group('Model')

    model_arg.add_argument("--hidden", type=int, default=32, help="Hidden size")
    model_arg.add_argument("--num_layers", type=int, default=1, help="Number of LSTM layers")
    model_arg.add_argument("--batch", type=int, default=40, help="Batch size")
    model_arg.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    model_arg.add_argument('--lr', type=float, default=1e-3, help='Initial lr value')

    return parser


def get_config():
    parser = get_parser()
    return parser.parse_known_args()[0]
