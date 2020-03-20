import unittest
import tempfile
import numpy as np

import moses
from moses.baselines import HMM, CombinatorialGenerator


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


if __name__ == '__main__':
    unittest.main()
