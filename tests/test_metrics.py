import unittest
import warnings
import numpy as np
from rdkit import Chem

from moses.metrics import get_all_metrics, fraction_valid, fraction_unique
from moses.utils import disable_rdkit_log, enable_rdkit_log


class test_metrics(unittest.TestCase):
    def setUp(self):
        self.test = ['Oc1ccccc1-c1cccc2cnccc12',
                     'COc1cccc(NC(=O)Cc2coc3ccc(OC)cc23)c1']
        self.test_sf = ['COCc1nnc(NC(=O)COc2ccc(C(C)(C)C)cc2)s1',
                        'O=C(C1CC2C=CC1C2)N1CCOc2ccccc21',
                        'Nc1c(Br)cccc1C(=O)Nc1ccncn1']
        self.gen = ['CNC', 'Oc1ccccc1-c1cccc2cnccc12',
                    'INVALID', 'CCCP',
                    'Cc1noc(C)c1CN(C)C(=O)Nc1cc(F)cc(F)c1',
                    'Cc1nc(NCc2ccccc2)no1-c1ccccc1']
        self.target = {'valid': 2/3,
                       'unique@3': 1.0,
                       'FCD/Test': 52.58371754126664,
                       'SNN/Test': 0.3152585653588176,
                       'Frag/Test': 0.3,
                       'Scaf/Test': 0.5,
                       'IntDiv': 0.7189187309761661,
                       'Filters': 0.75,
                       'logP': 1.63229,
                       'SA': 0.5238335295121783,
                       'QED': 0.20370891752648637,
                       'weight': 106.87}

    def test_get_all_metrics(self):
        metrics = get_all_metrics(gen=self.gen,
                                  test=self.test, k=3)
        fail = set()
        for metric in self.target:
            if not np.allclose(metrics[metric], self.target[metric]):
                warnings.warn(
                    "Metric `{}` value does not match expected "
                    "value. Got {}, expected {}".format(metric,
                                                        metrics[metric],
                                                        self.target[metric])
                )
                fail.add(metric)
        assert len(fail) == 0, f"Some metrics didn't pass tests: {fail}"

    def test_get_all_metrics_multiprocess(self):
        metrics = get_all_metrics(gen=self.gen,
                                  test=self.test, k=3, n_jobs=2)
        fail = set()
        for metric in self.target:
            if not np.allclose(metrics[metric], self.target[metric]):
                warnings.warn(
                    "Metric `{}` value does not match expected "
                    "value. Got {}, expected {}".format(metric,
                                                        metrics[metric],
                                                        self.target[metric])
                )
                fail.add(metric)
        assert len(fail) == 0, f"Some metrics didn't pass tests: {fail}"

    def test_get_all_metrics_scaffold(self):
        get_all_metrics(gen=self.gen,
                        test=self.test, test_scaffolds=self.test_sf,
                        k=3, n_jobs=2)

    def test_valid_unique(self):
        disable_rdkit_log()
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
        enable_rdkit_log()


if __name__ == '__main__':
    unittest.main()
