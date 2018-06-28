import unittest

import numpy as np
from metrics import *
from rdkit import Chem


class test_metrics(unittest.TestCase):
    def setUp(self):
        self.ref = ['Oc1ccccc1-c1cccc2cnccc12',
                    'COc1cccc(NC(=O)Cc2coc3ccc(OC)cc23)c1']
        self.gen = ['CNC', 'Oc1ccccc1-c1cccc2cnccc12', 'INVALID', 'CCCP',
                    'Cc1noc(C)c1CN(C)C(=O)Nc1cc(F)cc(F)c1']
        self.target = {'valid': 0.8,
                       'unique@3': 1.0,
                       'FCD': 28.246311814725665,
                       'morgan': 0.31525856535881758,
                       'fragments': 0.3,
                       'scaffolds': 0.5,
                       'internal_diversity': 0.7189187309761661,
                       'filters': 0.75}

    def test_get_all_metrics(self):
        metrics = get_all_metrics(self.ref, self.gen, k=3)
        print(metrics)
        for metric in self.target:
            assert np.allclose(metrics[metric], self.target[
                metric]), "Metric `{}` value does not match expected value".format(
                metric)

    def test_get_all_metrics_multiprocess(self):
        metrics = get_all_metrics(self.ref, self.gen, k=3, n_jobs=2)
        for metric in self.target:
            assert np.allclose(metrics[metric], self.target[
                metric]), "Metric `{}` value does not match expected value".format(
                metric)

    def test_similarity_prototype(self):
        fingerprint_similarity(['Oc1ccccc1-c1cccc2cnccc12'] * 100,
                               ['Cc1noc(C)c1CN(C)C(=O)Nc1cc(F)cc(F)c1'] * 2000)

    def test_valid_unique(self):
        mols = ['CCNC', 'CCC', 'INVALID', 'CCC']
        assert np.allclose(fraction_valid(mols), 3 / 4), "Failed valid"
        assert np.allclose(fraction_unique(mols, check_validity=False),
                           3 / 4), "Failed unique"
        assert np.allclose(fraction_unique(mols, k=2), 1), "Failed unique"
        mols = [Chem.MolFromSmiles(x) for x in mols]
        assert np.allclose(fraction_valid(mols), 3 / 4), "Failed valid"
        assert np.allclose(fraction_unique(mols, check_validity=False),
                           3 / 4), "Failed unique"
        assert np.allclose(fraction_unique(mols, k=2), 1), "Failed unique"


if __name__ == '__main__':
    unittest.main()
