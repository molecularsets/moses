import unittest
import tempfile

import moses
from moses.baselines import HMM, CombinatorialGenerator


class test_baselines(unittest.TestCase):
    def setUp(self):
        self.train = moses.get_dataset('train')

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
