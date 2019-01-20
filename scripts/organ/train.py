import torch
import rdkit

from moses.organ import ORGAN, ORGANTrainer, MetricsReward, get_parser as organ_parser
from moses.script_utils import add_train_args, read_smiles_csv, set_seed

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


def get_parser():
    parser = add_train_args(organ_parser())

    parser.add_argument('--n_ref_subsample', type=int, default=500,
                        help='Number of reference molecules (sampling from training data)')
    parser.add_argument('--addition_rewards', nargs='+', type=str,
                        choices=MetricsReward.supported_metrics, default=[],
                        help='Adding of addition rewards')

    return parser


def main(config):
    set_seed(config.seed)
    device = torch.device(config.device)

    train_data = read_smiles_csv(config.train_load)
    trainer = ORGANTrainer(config)

    if config.vocab_load is not None:
        assert os.path.exists(config.vocab_load), 'vocab_load path doesn\'t exist!'
        vocab = torch.load(config.vocab_load)
    else:
        vocab = trainer.get_vocabulary(train_data)

    model = ORGAN(vocab, config).to(device)

    trainer.fit(model, train_data)

    model = model.to('cpu')
    torch.save(model.state_dict(), config.model_save)
    torch.save(config, config.config_save)
    if config.vocab_save is not None:
        torch.save(vocab, config.vocab_save)

if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
