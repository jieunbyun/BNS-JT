import unittest
import importlib
import numpy as np

from BNS_JT.cpm import Cpm

np.set_printoptions(precision=3)
#pd.set_option.display_precision = 3

class Test1(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        variables = [1, 2, 3]
        numChild = 1
        C = np.array([[2, 2, 3], [2, 1, 2], [1, 1, 1]])
        p = [1, 1, 1]

        cls.kwargs = {'variables': variables,
                      'numChild': numChild,
                      'C': C,
                      'p': p}

    def test_init(self):

        _ = Cpm(**self.kwargs)

    def test_variables1(self):

        f_variables = [1, 2]
        with self.assertRaises(AssertionError):
            _ = Cpm(**{'variables': f_variables,
                       'numChild': self.kwargs['numChild'],
                       'C': self.kwargs['C'],
                       'p': self.kwargs['p']})

    def test_variables2(self):

        f_variables = [1, 2, 3, 4]
        with self.assertRaises(AssertionError):
            _ = Cpm(**{'variables': f_variables,
                       'numChild': self.kwargs['numChild'],
                       'C': self.kwargs['C'],
                       'p': self.kwargs['p']})

    def test_variables3(self):

        f_variables = ['x', 2, 3]
        with self.assertRaises(AssertionError):
            _ = Cpm(**{'variables': f_variables,
                       'numChild': self.kwargs['numChild'],
                       'C': self.kwargs['C'],
                       'p': self.kwargs['p']})

    def test_numChild(self):

        f_numChild = 3
        with self.assertRaises(AssertionError):
            _ = Cpm(**{'variables': self.kwargs['variables'],
                       'numChild': f_numChild,
                       'C': self.kwargs['C'],
                       'p': self.kwargs['p']})



if __name__=='__main__':
    unittest.main()

