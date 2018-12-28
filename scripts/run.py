import argparse
import os
import pandas as pd
import importlib.util
import sys


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

split_dataset = load_module('split_dataset', './split_dataset.py')
eval_script = load_module('eval', './metrics/eval.py')


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='all', choices=['all'] + get_models(),
                        help='Which model to run')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Directory for datasets')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--n_samples', type=int, default=30000,
                        help='Number of samples to sample')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of threads')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU index (-1 for cpu)')
    parser.add_argument('--metrics', type=str, default='metrics.csv',
                        help='Path to output file with metrics')
    parser.add_argument('--train_size', type=int, default=None,
                        help='Size of training dataset')
    parser.add_argument('--test_size', type=int, default=None,
                        help='Size of testing dataset')
    parser.add_argument('--experiment_suff', type=str, default='',
                        help='Experiment suffix to break ambiguity')
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
    model_path = os.path.join(config.checkpoint_dir, model + config.experiment_suff + '_model.pt')
    config_path = os.path.join(config.checkpoint_dir, model + config.experiment_suff + '_config.pt')
    vocab_path = os.path.join(config.checkpoint_dir, model + config.experiment_suff + '_vocab.pt')

    if os.path.exists(model_path) and \
            os.path.exists(config_path) and \
            os.path.exists(vocab_path):
        return

    trainer = get_trainer(model)
    trainer_parser = trainer.get_parser()
    device = f'cuda:{config.gpu}' if config.gpu >= 0 else 'cpu'
    trainer_config = trainer_parser.parse_known_args(sys.argv+[
                                                     '--device', device,
                                                     '--train_load', train_path,
                                                     '--model_save', model_path,
                                                     '--config_save', config_path,
                                                     '--vocab_save', vocab_path,
                                                     '--n_jobs', str(config.n_jobs)])[0]
    trainer.main(trainer_config)


def sample_from_model(config, model):
    sampler = get_sampler(model)
    sampler_parser = sampler.get_parser()
    model_path = os.path.join(config.checkpoint_dir, model + config.experiment_suff + '_model.pt')
    config_path = os.path.join(config.checkpoint_dir, model + config.experiment_suff + '_config.pt')
    vocab_path = os.path.join(config.checkpoint_dir, model + config.experiment_suff + '_vocab.pt')
    gen_save = os.path.join(config.data_dir, model + config.experiment_suff + '_generated.csv')
    device = f'cuda:{config.gpu}' if config.gpu >= 0 else 'cpu'
    sampler_config = sampler_parser.parse_known_args(sys.argv+[
                                                     '--device', device,
                                                     '--model_load', model_path,
                                                     '--config_load', config_path,
                                                     '--vocab_load', vocab_path,
                                                     '--gen_save', gen_save,
                                                     '--n_samples', str(config.n_samples)])[0]
    sampler.main(sampler_config)


def eval_metrics(config, model, test_path, test_scaffolds_path, ptest_path, ptest_scaffolds_path):
    eval_parser = eval_script.get_parser()
    gen_path = os.path.join(config.data_dir, model + config.experiment_suff + '_generated.csv')
    eval_config = eval_parser.parse_args(['--test_path', test_path,
                                          '--test_scaffolds_path', test_scaffolds_path,
                                          '--ptest_path', ptest_path,
                                          '--ptest_scaffolds_path', ptest_scaffolds_path,
                                          '--gen_path', gen_path,
                                          '--n_jobs', str(config.n_jobs),
                                          '--gpu', str(config.gpu)])
    metrics = eval_script.main(eval_config, print_metrics=False)

    return metrics


def main(config):
    if not os.path.exists(config.data_dir):
        os.mkdir(config.data_dir)

    if not os.path.exists(config.checkpoint_dir):
        os.mkdir(config.checkpoint_dir)

    train_path = os.path.join(config.data_dir, 'train.csv')
    test_path = os.path.join(config.data_dir, 'test.csv')
    test_scaffolds_path = os.path.join(config.data_dir, 'test_scaffolds.csv')
    train_path = os.path.join(config.data_dir, 'train.csv')
    ptest_path = os.path.join(config.data_dir, 'test_stats.npz')
    ptest_scaffolds_path = os.path.join(config.data_dir, 'test_scaffolds_stats.npz')


    if not os.path.exists(train_path) or \
            not os.path.exists(test_path) or \
            not os.path.exists(test_scaffolds_path):
        splitting_config = split_dataset.get_parser()
        conf = ['--output_dir', config.data_dir]
        if config.train_size is not None:
            conf.extend(['--train_size', str(config.train_size)])
        if config.test_size is not None:
            conf.extend(['--test_size', str(config.test_size)])
        splitting_config = splitting_config.parse_args(conf)
        split_dataset.main(splitting_config)

    models = get_models() if config.model == 'all' else [config.model]
    for model in models:
        train_model(config, model, train_path)
        sample_from_model(config, model)

    metrics = []
    for model in models:
        model_metrics = eval_metrics(config, model,
                                     test_path, test_scaffolds_path,
                                     ptest_path, ptest_scaffolds_path)
        model_metrics.update({'model': model})
        metrics.append(model_metrics)

    table = pd.DataFrame(metrics)
    table.to_csv(config.metrics, index=False)


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
