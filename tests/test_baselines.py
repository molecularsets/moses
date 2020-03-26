import unittest
import tempfile
import numpy as np

import moses
from moses.baselines import CombinatorialGenerator
from moses.baselines import NGram
from moses.baselines import HMM


class test_baselines(unittest.TestCase):
    def setUp(self):
        self.train = moses.get_dataset('train')

    def test_hmm(self):
        model = HMM(n_components=5, seed=1)
        model.fit(self.train[:10])
        np.random.seed(1)
        sample_original = model.generate_one()
        with tempfile.NamedTemporaryFile() as f:
            model.save(f.name)
            model = HMM.load(f.name)
        np.random.seed(1)
        sample_loaded = model.generate_one()
        self.assertEqual(
            sample_original, sample_loaded,
            "Samples before and after saving differ"
        )

    def test_combinatorial(self):
        model = CombinatorialGenerator()
        model.fit(self.train[:10])
        sample_original = model.generate_one(1)
        with tempfile.NamedTemporaryFile() as f:
            model.save(f.name)
            model = CombinatorialGenerator.load(f.name)
        sample_loaded = model.generate_one(1)
        self.assertEqual(
            sample_original, sample_loaded,
            "Samples before and after saving differ"
        )

    def test_ngram(self):
        model_1 = NGram(1)
        model_1.fit(self.train[:1000])
        model_2 = NGram(2)
        model_2.fit(self.train[:1000])
        np.random.seed(0)
        sample_1 = model_1.generate_one(context_len=1)
        np.random.seed(0)
        sample_2 = model_2.generate_one(context_len=1)
        with tempfile.NamedTemporaryFile() as f:
            model_1.save(f.name)
            model_l = NGram.load(f.name)
        np.random.seed(0)
        sample_l = model_l.generate_one(context_len=1)
        self.assertEqual(
            sample_1, sample_2,
            "Samples with the same context from two models"
        )
        self.assertEqual(
            sample_1, sample_l,
            "Samples before and after saving differ"
        )


if __name__ == '__main__':
    unittest.main()
