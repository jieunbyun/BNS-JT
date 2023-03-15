import unittest
import importlib
import numpy as np

from BNS_JT.variable import Variable

np.set_printoptions(precision=3)
#pd.set_option.display_precision = 3

class Test_Varaible(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        B = np.array([[1, 0], [0, 1], [1, 1]])
        value = ['survival', 'fail']

        cls.kwargs = {'B': B,
                      'value': value,
                      }

    def test_init(self):
        a = Variable(**self.kwargs)

        self.assertTrue(isinstance(a, Variable))
        np.testing.assert_array_equal(a.B, self.kwargs['B'])
        np.testing.assert_array_equal(a.value, self.kwargs['value'])

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

if __name__=='__main__':
    unittest.main()

