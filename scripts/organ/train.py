import torch
import rdkit

from moses.organ import ORGAN, ORGANTrainer, get_parser as organ_parser
from moses.script_utils import add_train_args, read_smiles_csv, set_seed, MetricsReward
from moses.utils import CharVocab
from multiprocessing import Pool

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


def get_parser():
    parser = add_train_args(organ_parser())

    parser.add_argument('--n_ref_subsample', type=int, default=500,
                        help='Number of reference molecules (sampling from training data)')
    parser.add_argument('--addition_rewards', nargs='+', type=str,
                        choices=MetricsReward.supported_metrics, default=[],
                        help='Adding of addition rewards')

    # conditional generation
    parser.add_argument('--conditional', type=int, default=0,
                       help='Conditional generation mode')
    parser.add_argument('--output_size', type=int, default=10,
                       help='Output size in the condition linear layer')

    return parser

# read fingerrprints
def read_fps_csv(path):
    return pd.read_csv(path,
                       usecols=['fingerprints_center'],
                       squeeze=True).astype(str).tolist()

# convert fingerprints to list
def fps_to_list(fps):
    fps = [list(x) for x in fps]
    for i in tqdm(range(len(fps))):
        fps[i] = [int(x) for x in fps[i]]
    return fps


def main(config):
    set_seed(config.seed)

    train = read_smiles_csv(config.train_load)
    vocab = CharVocab.from_data(train)
    torch.save(vocab, config.vocab_save)
    torch.save(config, config.config_save)
    device = torch.device(config.device)

    # condition mode
    if config.conditional:
        fps = read_fps_csv(config.train_load)
        fps = fps_to_list(fps)
        fps = [torch.tensor(f, dtype=torch.float, device=device) for f in fps]
        # fingerprints length
        fps_len = len(fps[0])
    else:
        fps = None
        fps_len = 0

    with Pool(config.n_jobs) as pool:
        reward_func = MetricsReward(train, config.n_ref_subsample, config.rollouts, pool, config.addition_rewards)
        model = ORGAN(vocab, config, fps_len, reward_func)
        model = model.to(device)

        trainer = ORGANTrainer(config)
        trainer.fit(model, train, fps)

    torch.save(model.state_dict(), config.model_save)


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
