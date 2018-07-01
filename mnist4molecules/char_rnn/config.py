from mnist4molecules.config import get_sample_parser as get_common_sample_parser
from mnist4molecules.config import get_train_parser as get_common_train_parser


def get_train_parser():
    parser = get_common_train_parser()

    # Model
    model_arg = parser.add_argument_group('Model')

    model_arg.add_argument("--val_load", type=str, help="Input data in csv format to validation")
    model_arg.add_argument("--num_layers", type=int, default=3, help="Number of LSTM layers")
    model_arg.add_argument("--hidden", type=int, default=600, help="Hidden size")
    model_arg.add_argument("--dropout", type=float, default=0.2, help="dropout between LSTM layers except for last")
    model_arg.add_argument("--batch", type=int, default=64, help="Batch size")
    model_arg.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    model_arg.add_argument('--lr', type=float, default=0.001, help='Initial lr value')

    return parser


def get_sample_parser():
    parser = get_common_sample_parser()

    # Model
    model_arg = parser.add_argument_group('Model')

    model_arg.add_argument("--max_len", type=int, default=100, help="Max of length of SMILES")
    model_arg.add_argument("--batch", type=int, default=50, help="Batch size")

    return parser
