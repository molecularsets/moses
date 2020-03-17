import os
import argparse
import sys
import importlib.util
import pandas as pd

from moses.models_storage import ModelsStorage


def load_module(name, path):
    dirname = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(dirname, path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


MODELS = ModelsStorage()
split_dataset = load_module('split_dataset', 'split_dataset.py')
eval_script = load_module('eval', 'eval.py')
trainer_script = load_module('train', 'train.py')
sampler_script = load_module('sample', 'sample.py')


def get_model_path(config, model):
    return os.path.join(
        config.checkpoint_dir, model + config.experiment_suff + '_model.pt'
    )


def get_log_path(config, model):
    return os.path.join(
        config.checkpoint_dir, model + config.experiment_suff + '_log.txt'
    )


def get_config_path(config, model):
    return os.path.join(
        config.checkpoint_dir, model + config.experiment_suff + '_config.pt'
    )


def get_vocab_path(config, model):
    return os.path.join(
        config.checkpoint_dir, model + config.experiment_suff + '_vocab.pt'
    )


def get_generation_path(config, model):
    return os.path.join(
        config.checkpoint_dir,
        model + config.experiment_suff + '_generated.csv'
    )


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='all',
                        choices=['all'] + MODELS.get_model_names(),
                        help='Which model to run')
    parser.add_argument('--test_path',
                        type=str, required=False,
                        help='Path to test molecules csv')
    parser.add_argument('--test_scaffolds_path',
                        type=str, required=False,
                        help='Path to scaffold test molecules csv')
    parser.add_argument('--train_path',
                        type=str, required=False,
                        help='Path to train molecules csv')
    parser.add_argument('--ptest_path',
                        type=str, required=False,
                        help='Path to precalculated test npz')
    parser.add_argument('--ptest_scaffolds_path',
                        type=str, required=False,
                        help='Path to precalculated scaffold test npz')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--n_samples', type=int, default=30000,
                        help='Number of samples to sample')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of threads')
    parser.add_argument('--device', type=str, default='cpu',
                        help='GPU device index in form `cuda:N` (or `cpu`)')
    parser.add_argument('--metrics', type=str, default='metrics.csv',
                        help='Path to output file with metrics')
    parser.add_argument('--train_size', type=int, default=None,
                        help='Size of training dataset')
    parser.add_argument('--test_size', type=int, default=None,
                        help='Size of testing dataset')
    parser.add_argument('--experiment_suff', type=str, default='',
                        help='Experiment suffix to break ambiguity')
    return parser


def train_model(config, model, train_path, test_path):
    print('Training...')
    model_path = get_model_path(config, model)
    config_path = get_config_path(config, model)
    vocab_path = get_vocab_path(config, model)
    log_path = get_log_path(config, model)

    if os.path.exists(model_path) and \
            os.path.exists(config_path) and \
            os.path.exists(vocab_path):
        return

    trainer_parser = trainer_script.get_parser()
    args = [
        '--device', config.device,
        '--model_save', model_path,
        '--config_save', config_path,
        '--vocab_save', vocab_path,
        '--log_file', log_path,
        '--n_jobs', str(config.n_jobs)
    ]
    if train_path is not None:
        args.extend(['--train_load', train_path])
    if test_path is not None:
        args.extend(['--val_load', test_path])

    trainer_config = trainer_parser.parse_known_args(
         [model] + sys.argv[1:] + args
    )[0]
    trainer_script.main(model, trainer_config)


def sample_from_model(config, model):
    print('Sampling...')
    model_path = get_model_path(config, model)
    config_path = get_config_path(config, model)
    vocab_path = get_vocab_path(config, model)

    assert os.path.exists(model_path), (
        "Can't find model path for sampling: '{}'".format(model_path)
    )
    assert os.path.exists(config_path), (
        "Can't find config path for sampling: '{}'".format(config_path)
    )
    assert os.path.exists(vocab_path), (
        "Can't find vocab path for sampling: '{}'".format(vocab_path)
    )

    sampler_parser = sampler_script.get_parser()
    sampler_config = sampler_parser.parse_known_args(
        [model] + sys.argv[1:] +
        ['--device', config.device,
         '--model_load', model_path,
         '--config_load', config_path,
         '--vocab_load', vocab_path,
         '--gen_save', get_generation_path(config, model),
         '--n_samples', str(config.n_samples)]
    )[0]
    sampler_script.main(model, sampler_config)


def eval_metrics(config, model, test_path, test_scaffolds_path,
                 ptest_path, ptest_scaffolds_path, train_path):
    print('Computing metrics...')
    eval_parser = eval_script.get_parser()
    args = [
        '--gen_path', get_generation_path(config, model),
        '--n_jobs', str(config.n_jobs),
        '--device', config.device,
    ]
    if test_path is not None:
        args.extend(['--test_path', test_path])
    if test_scaffolds_path is not None:
        args.extend(['--test_scaffolds_path', test_scaffolds_path])
    if ptest_path is not None:
        args.extend(['--ptest_path', ptest_path])
    if ptest_scaffolds_path is not None:
        args.extend(['--ptest_scaffolds_path', ptest_scaffolds_path])
    if train_path is not None:
        args.extend(['--train_path', train_path])

    eval_config = eval_parser.parse_args(args)
    metrics = eval_script.main(eval_config, print_metrics=False)

    return metrics


def main(config):
    if not os.path.exists(config.checkpoint_dir):
        os.mkdir(config.checkpoint_dir)

    train_path = config.train_path
    test_path = config.test_path
    test_scaffolds_path = config.test_scaffolds_path
    ptest_path = config.ptest_path
    ptest_scaffolds_path = config.ptest_scaffolds_path

    models = (MODELS.get_model_names()
              if config.model == 'all'
              else [config.model])
    for model in models:
        train_model(config, model, train_path, test_path)
        sample_from_model(config, model)

    for model in models:
        model_metrics = eval_metrics(config, model,
                                     test_path, test_scaffolds_path,
                                     ptest_path, ptest_scaffolds_path,
                                     train_path)
        table = pd.DataFrame([model_metrics]).T
        if len(models) == 1:
            metrics_path = ''.join(
                os.path.splitext(config.metrics)[:-1])+f'_{model}.csv'
        else:
            metrics_path = config.metrics
        table.to_csv(metrics_path, header=False)


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
