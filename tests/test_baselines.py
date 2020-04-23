import unittest
import tempfile
import numpy as np

import moses
from moses.baselines import CombinatorialGenerator
from moses.baselines import NGram
from moses.baselines import HMM
from moses.metrics.utils import fragmenter


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
        self.assertEqual(
            self.train[0],
            'CCCS(=O)c1ccc2[nH]c(=NC(=O)OC)[nH]c2c1',
            "Train set is different: %s" % self.train[0]
        )
        fragments = fragmenter('CCCS(=O)c1ccc2[nH]c(=NC(=O)OC)[nH]c2c1')
        self.assertEqual(
            fragments,
            ['[1*]C(=O)N=c1[nH]c2ccc(S(=O)CCC)cc2[nH]1', '[3*]OC'],
            "Fragmenter is not working properly: %s" % str(fragments)
        )
        model = CombinatorialGenerator()
        model.fit(self.train[:100])
        self.assertEqual(
            model.fragment_counts.shape,
            (156, 5),
            "Model was not fitted properly"
        )
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
