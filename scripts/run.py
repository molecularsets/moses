import argparse
import os
import pandas as pd
import importlib.util


def get_models():
    return ['aae', 'char_rnn', 'junction_tree', 'organ', 'vae']


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


for model in get_models():
    globals()[model + '_train'] = load_module('train', os.path.join('.', model, 'train.py'))
    globals()[model + '_sample'] = load_module('sample', os.path.join('.', model, 'sample.py'))

download_dataset = load_module('download_dataset', './download_dataset.py')
eval = load_module('eval', './metrics/eval.py')


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='all', choices=['all'] + get_models(),
                        help='Which model to run')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory for datasets')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--n_samples', type=int, default=30000,
                        help='Number of samples to sample')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of threads')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run: "cpu" or "cuda:<device number>"')

    return parser


def get_trainer(model):
    mapper = {'aae': aae_train, # noqa
              'char_rnn': char_rnn_train, # noqa
              'junction_tree': junction_tree_train, # noqa
              'organ': organ_train, # noqa
              'vae': vae_train} # noqa

    return mapper[model]


def get_sampler(model):
    mapper = {'aae': aae_sample, # noqa
              'char_rnn': char_rnn_sample, # noqa
              'junction_tree': junction_tree_sample, # noqa
              'organ': organ_sample, # noqa
              'vae': vae_sample} # noqa

    return mapper[model]


def train_model(config, model, train_path):
    model_path = os.path.join(config.checkpoint_dir, model + '_model.pt')
    config_path = os.path.join(config.checkpoint_dir, model + '_config.pt')
    vocab_path = os.path.join(config.checkpoint_dir, model + '_vocab.pt')

    if os.path.exists(model_path) and \
            os.path.exists(config_path) and \
            os.path.exists(vocab_path):
        return

    trainer = get_trainer(model)
    trainer_parser = trainer.get_parser()
    trainer_config = trainer_parser.parse_args(['--device', config.device,
                                                '--train_load', train_path,
                                                '--model_save', model_path,
                                                '--config_save', config_path,
                                                '--vocab_save', vocab_path])
    trainer.main(trainer_config)


def sample_from_model(config, model):
    sampler = get_sampler(model)
    sampler_parser = sampler.get_parser()
    sampler_config = sampler_parser.parse_args(['--device', config.device,
                                                '--model_load', os.path.join(config.checkpoint_dir,
                                                                             model + '_model.pt'),
                                                '--config_load', os.path.join(config.checkpoint_dir,
                                                                              model + '_config.pt'),
                                                '--vocab_load', os.path.join(config.checkpoint_dir,
                                                                             model + '_vocab.pt'),
                                                '--gen_save', os.path.join(config.data_dir, model + '_generated.csv'),
                                                '--n_samples', str(config.n_samples)])
    sampler.main(sampler_config)


def eval_metrics(config, model, ref_path):
    eval_parser = eval.get_parser()
    eval_config = eval_parser.parse_args(['--ref_path', ref_path,
                                          '--gen_path', os.path.join(config.data_dir, model + '_generated.csv'),
                                          '--n_jobs', str(config.n_jobs)])
    metrics = eval.main(eval_config, print_metrics=False)

    return metrics


def main(config):
    if not os.path.exists(config.data_dir):
        os.mkdir(config.data_dir)

    if not os.path.exists(config.checkpoint_dir):
        os.mkdir(config.checkpoint_dir)

    train_path = os.path.join(config.data_dir, 'train.csv')
    test_path = os.path.join(config.data_dir, 'test.csv')
    test_scaffolds_path = os.path.join(config.data_dir, 'test_scaffolds.csv')

    if not os.path.exists(train_path) or \
            not os.path.exists(test_path) or \
            os.path.exists(test_scaffolds_path):
        downloading_config = download_dataset.get_parser()
        downloading_config = downloading_config.parse_args(['--output_dir', config.data_dir])
        download_dataset.main(downloading_config)

    models = get_models() if config.model == 'all' else [config.model]
    for model in models:
        train_model(config, model, train_path)
        sample_from_model(config, model)

    metrics = []
    for model in models:
        model_metrics = eval_metrics(config, model, test_path)
        model_metrics.update({'model': model + '_test'})
        metrics.append(model_metrics)

        model_metrics = eval_metrics(config, model, test_scaffolds_path)
        model_metrics.update({'model': model + '_test_scaffolds'})
        metrics.append(model_metrics)

    table = pd.DataFrame(metrics)
    table.to_csv('metrics.csv', index=False)


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
