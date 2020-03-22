import pickle
import numpy as np
from pomegranate import HiddenMarkovModel, DiscreteDistribution

import moses


class HMM:
    def __init__(self, n_components=200,
                 epochs=100,
                 batches_per_epoch=100,
                 seed=0, verbose=False,
                 n_jobs=1):
        """
        Creates a Hidden Markov Model

        Arguments:
            n_components: numebr of states in HMM
            epochs: number of iterations to train model for
            batches_per_epoch: number of batches for minibatch training
            seed: seed for initializing the model
            verbose: if True, will log training statistics
            n_jobs: number of threads for training HMM model
        """
        self.n_components = n_components
        self.epochs = epochs
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.fitted = False

    def fit(self, data):
        """
        Fits a model---learns transition and emission probabilities

        Arguments:
            data: list of SMILES
        """
        list_data = [list(smiles) for smiles in data]
        self.model = HiddenMarkovModel.from_samples(
            DiscreteDistribution, n_components=self.n_components,
            end_state=True, X=list_data,
            init='kmeans||', verbose=self.verbose, n_jobs=self.n_jobs,
            max_iterations=self.epochs,
            batches_per_epoch=self.batches_per_epoch,
            random_state=self.seed
        )
        self.fitted = True
        return self

    def save(self, path):
        """
        Saves a model using pickle

        Arguments:
            path: path to .pkl file for saving
        """
        if not self.fitted:
            raise RuntimeError("Can't save empty model."
                               " Fit the model first")
        json = self.model.to_json()
        with open(path, "wb") as f:
            pickle.dump({
                'model': json,
                'n_components': self.n_components,
                'epochs': self.epochs,
                'batches_per_epoch': self.batches_per_epoch,
                'verbose': self.verbose,
            }, f)

    @classmethod
    def load(cls, path):
        """
        Loads saved model

        Arguments:
            path: path to saved .pkl file

        Returns:
            Loaded HMM
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        hmm = data['model']
        del data['model']
        model = cls(**data)
        model.model = HiddenMarkovModel.from_json(hmm)
        model.fitted = True
        return model

    def generate_one(self):
        """
        Generates a SMILES string using a trained HMM

        Retruns:
            SMILES string
        """
        return ''.join(self.model.sample())


def reproduce(seed, samples_path=None, metrics_path=None,
              n_jobs=1, device='cpu', verbose=False,
              samples=30000):
    data = moses.get_dataset('train')[:100000]
    if verbose:
        print("Training...")
    model = HMM(n_jobs=n_jobs, seed=seed, verbose=verbose)
    model.fit(data)
    np.random.seed(seed)
    if verbose:
        print(f"Sampling for seed {seed}")
    np.random.seed(seed)
    samples = [model.generate_one()
               for _ in range(samples)]
    if samples_path is not None:
        with open(samples_path, 'w') as f:
            f.write('SMILES\n')
            for sample in samples:
                f.write(sample+'\n')
    if verbose:
        print(f"Computing metrics for seed {seed}")
    metrics = moses.get_all_metrics(
        samples, n_jobs=n_jobs, device=device)
    if metrics_path is not None:
        with open(samples_path, 'w') as f:
            for key, value in metrics.items():
                f.write("%s,%f\n" % (key, value))
    return samples, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "Reproduce HMM experiment for one seed (~24h with n_jobs=32)")
    parser.add_argument(
        '--n_jobs', type=int, required=False,
        default=1, help='Number of threads for computing metrics')
    parser.add_argument(
        '--device', type=str, required=False,
        default='cpu', help='Device for computing metrics')
    parser.add_argument(
        '--samples', type=int, required=False,
        default=30000, help='Number of samples for metrics')
    parser.add_argument(
        '--metrics_path', type=str, required=False,
        default='.', help='Path to save metrics')
    parser.add_argument(
        '--seed', type=int, required=False,
        default=1, help='Random seed')
    parser.add_argument(
        '--model_save', type=str, required=False,
        help='File for saving the model')

    args = parser.parse_known_args()[0]
    reproduce(
        seed=args.seed, metrics_path=args.model_save,
        n_jobs=args.n_jobs, device=args.device,
        verbose=True, samples=args.samples
    )
