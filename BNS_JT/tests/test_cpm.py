import unittest
import importlib
import numpy as np

from BNS_JT.cpm import Cpm, ismember, isCompatible, get_value_given_condn
from BNS_JT.variable import Variable

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

    def test_init1(self):
        a = Cpm(**self.kwargs)

        self.assertTrue(isinstance(a, Cpm))
        self.assertEqual(a.variables, self.kwargs['variables'])
        self.assertEqual(a.numChild, self.kwargs['numChild'])
        np.testing.assert_array_equal(a.C, self.kwargs['C'])

    def test_init2(self):
        # using list for P
        a = Cpm(variables=[1], numChild=1, C=np.array([1, 2]), p=[0.9, 0.1])
        self.assertTrue(isinstance(a, Cpm))

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

        f_numChild = 4
        with self.assertRaises(AssertionError):
            _ = Cpm(**{'variables': self.kwargs['variables'],
                       'numChild': f_numChild,
                       'C': self.kwargs['C'],
                       'p': self.kwargs['p']})


class Test_isCompatible(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.M = {}
        cls.vars_ = {}

        cls.M[1] = Cpm(variables=[1], numChild=1, C = np.array([1, 2]).T, p = np.array([0.9, 0.1]).T)
        cls.M[2] = Cpm(variables=[2, 1], numChild=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]), p = np.array([0.99, 0.01, 0.9, 0.1]).T)
        cls.M[3] = Cpm(variables=[3, 1], numChild=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]), p = np.array([0.95, 0.05, 0.85, 0.15]).T)
        cls.M[4] = Cpm(variables=[4, 1], numChild=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]), p = np.array([0.99, 0.01, 0.9, 0.1]).T)
        cls.M[5] = Cpm(variables=[5, 2, 3, 4], numChild=1, C = np.array([[2, 3, 3, 2], [1, 1, 3, 1], [1, 2, 1, 1], [2, 2, 2, 1]]), p = np.array([1, 1, 1, 1]).T)

        cls.vars_[1] = Variable(B=np.eye(2), value=['Mild', 'Severe'])
        cls.vars_[2] = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]), value=['Survive', 'Fail'])
        cls.vars_[3] = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]), value=['Survive', 'Fail'])
        cls.vars_[4] = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]), value=['Survive', 'Fail'])
        cls.vars_[5] = Variable(B=np.array([[1, 0], [0, 1]]), value=['Survive', 'Fail'])

    def test_ismember1(self):

        checkVars = [1]
        variables = [2, 1]

        idxInCheckVars = ismember(checkVars, variables)

        self.assertEqual([1], idxInCheckVars)

    def test_ismember2(self):

        A = [5, 3, 4, 2]
        B = [2, 4, 4, 4, 6, 8]

        # MATLAB: [0, 0, 2, 1] => [False, False, 1, 0]
        result = ismember(A, B)
        self.assertEqual([False, False, 1, 0], result)

        result = ismember(B, A)
        self.assertEqual([3, 2, 2, 2, False, False], result)


    def test_ismember3(self):

        A = np.array([5, 3, 4, 2])
        B = np.array([2, 4, 4, 4, 6, 8])
        # MATLAB: [0, 0, 2, 1] => [False, False, 1, 0]

        expected = [False, False, 1, 0]
        result = ismember(A, B)

        self.assertEqual(expected, result)

    def test_get_value_given_condn1(self):

        A = [1, 2, 3, 4]
        condn = [0, False, 1, 3]

        self.assertEqual([1, 3, 4], get_value_given_condn(A, condn))

    def test_get_value_given_condn2(self):

        A = [1, 2, 3]
        condn = [0, False, 1, 3]

        with self.assertRaises(AssertionError):
            get_value_given_condn(A, condn)

    def test_isCompatible1(self):

        # M[2]
        C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]])  #M[2].C
        variables = [2, 1]
        checkVars = [1]
        checkStates = [1]
        vInfo = self.vars_

        result = isCompatible(C, variables, checkVars, checkStates, vInfo)
        expected = np.array([[1, 1, 0, 0]]).T
        np.testing.assert_array_equal(expected, result)

    def test_isCompatible2(self):

        # M[5]
        C = np.array([[2, 3, 3, 2], [1, 1, 3, 1], [1, 2, 1, 1], [2, 2, 2, 1]])
        variables = [5, 2, 3, 4]
        checkVars = [3, 4]
        checkStates = [1, 1]
        vInfo = self.vars_

        result = isCompatible(C, variables, checkVars, checkStates, vInfo)
        expected = np.array([[0, 1, 1, 0]]).T
        np.testing.assert_array_equal(expected, result)

    def test_isCompatible3(self):

        #M[1]
        C = np.array([[1, 2]]).T
        variables = [1]
        checkVars = [3, 4]
        checkStates = [1, 1]
        vInfo = self.vars_

        result = isCompatible(C, variables, checkVars, checkStates, vInfo)
        expected = np.array([[1, 1]]).T
        np.testing.assert_array_equal(expected, result)


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

