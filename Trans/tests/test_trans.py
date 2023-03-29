import unittest
import numpy as np
import pandas as pd

#from BNS_JT.variable import Variable
from Trans.trans import get_arcs_length

np.set_printoptions(precision=3)
#pd.set_option.display_precision = 3

class Test(unittest.TestCase):
    """
    @classmethod
    def setUpClass(cls):

        B = np.array([[1, 0], [0, 1], [1, 1]])
        value = ['survival', 'fail']

        cls.kwargs = {'B': B,
                      'value': value,
                      }
    """
    def test_get_arcs_length(self):

        node_coords = {1: [-2, 3],
                       2: [-2, -3],
                       3: [2, -2],
                       4: [1, 1],
                       5: [0, 0]}

        arcs = {1: [1, 2],
                2: [1,5],
                3: [2,5],
                4: [3,4],
                5: [3,5],
                6: [4,5]}

        result = get_arcs_length(arcs, node_coords)

        expected = {1: 6.0,
                    2: 3.6056,
                    3: 3.6056,
                    4: 3.1623,
                    5: 2.8284,
                    6: 1.4142}

        pd.testing.assert_series_equal(pd.Series(result), pd.Series(expected), rtol=1.0e-3)

    """
    def test_B1(self):

        f_B = [[1, 2], [0, 1], [1, 1]]
        with self.assertRaises(AssertionError):
            _ = Variable(**{'B': f_B,
                         'value': self.kwargs['value']})

    def test_B2(self):

        f_B = [[1, 2]]
        with self.assertRaises(AssertionError):
            _ = Variable(**{'B': f_B,
                         'value': self.kwargs['value']})
    """
if __name__=='__main__':
    unittest.main()

