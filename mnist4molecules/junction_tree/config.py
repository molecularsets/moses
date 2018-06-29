from mnist4molecules.config import get_sample_parser as get_common_sample_parser
from mnist4molecules.config import get_train_parser as get_common_train_parser


def get_train_parser():
    parser = get_common_train_parser()

    # Model
    model_arg = parser.add_argument_group('Model')

    # TODO: check if user uses vocab_load and not rewrite vocabulary
    model_arg.add_argument("--vocab_load", type=str, help="Where to load vocabulary otherwise it will be evaluated")
    model_arg.add_argument("--hidden", type=int, default=450, help="Hidden size")
    model_arg.add_argument("--latent", type=int, default=56, help="Latent size")
    model_arg.add_argument("--depth", type=int, default=3, help="Depth of graph message passing")
    model_arg.add_argument("--batch", type=int, default=40, help="Batch size")
    model_arg.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    model_arg.add_argument('--kl_start', type=int, default=1, help='Epoch to init KL weight (start from 0)')
    model_arg.add_argument('--kl_w', type=float, default=0.005, help='KL weight value')
    model_arg.add_argument('--lr', type=float, default=1e-3, help='Initial lr value')

    return parser


def get_sample_parser():
    parser = get_common_sample_parser()
    return parser
