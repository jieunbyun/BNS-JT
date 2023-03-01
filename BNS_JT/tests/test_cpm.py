import unittest
import importlib
import numpy as np

from BNS_JT.cpm import Cpm

np.set_printoptions(precision=3)
#pd.set_option.display_precision = 3

class Test_Cpm(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        variables = [3, 2, 1]
        numChild = 1
        C = np.array([[2, 2, 3], [2, 1, 2], [1, 1, 1]])
        p = [1, 1, 1]

        cls.kwargs = {'variables': variables,
                      'numChild': numChild,
                      'C': C,
                      'p': p}

    def test_init(self):
        a = Cpm(**self.kwargs)

        self.assertTrue(isinstance(a, Cpm))
        self.assertEqual(a.variables, self.kwargs['variables'])
        self.assertEqual(a.numChild, self.kwargs['numChild'])
        np.testing.assert_array_equal(a.C, self.kwargs['C'])

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

class Test_Sum(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        variables = [3, 1, 2]
        numChild = 1
        C = np.array([[2, 2, 3], [2, 1, 2], [1, 1, 1]])
        p = [1, 1, 1]

        cls.kwargs = {'variables': variables,
                      'numChild': numChild,
                      'C': C,
                      'p': p}

    def test_get_varsRemain0(self):
        '''
        sumFlag = 0
        '''
        sumVars = 2
        expected = np.array([1, 2])
        Msys = Cpm(**self.kwargs)
        #result = get_varsRemain(Msys, sumVars, 0)

        #self.assertEqual(expected, result)


 
if __name__=='__main__':
    unittest.main()

