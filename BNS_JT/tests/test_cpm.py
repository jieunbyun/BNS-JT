import unittest
import importlib
import numpy as np

from BNS_JT.cpm import Cpm, ismember, isCompatible, get_value_given_condn, getCpmSubset, isCompatibleCpm, flip, addNewStates, condition
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

    def test_ismember4(self):

        A = np.array([np.ones(4), np.zeros(4)]).T
        B = np.array([[1, 0], [0, 1], [1, 1]])

        expected = [0, 0, 0, 0]
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

    def test_isCompatible4(self):

        C = np.array([[1,1,1,1,1],
             [2,1,1,1,1],
             [1,2,1,1,1],
             [2,2,2,1,1],
             [1,1,1,2,1],
             [2,1,1,2,1],
             [1,2,1,2,1],
             [2,2,2,2,1],
             [1,1,2,1,2],
             [2,1,2,1,2],
             [1,2,2,1,2],
             [2,2,2,1,2],
             [1,1,2,2,2],
             [2,1,2,2,2],
             [1,2,2,2,2],
             [2,2,2,2,2]])
        variables = [2, 3, 5, 1, 4]
        checkVars = np.array([2, 1])
        checkStates = np.array([1, 1])
        vars_ = self.vars_

        result = isCompatible(C, variables, checkVars, checkStates, vars_)
        expected = np.array([[1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0]]).T
        #np.testing.assert_array_equal(expected, result)

        idx = ismember(checkVars, variables)
        # should be one less than the Matlab result
        self.assertEqual(idx, [0, 3])

        checkVars = get_value_given_condn(checkVars, idx)
        self.assertEqual(checkVars, [2, 1])

        checkStates = get_value_given_condn(checkStates, idx)
        self.assertEqual(checkStates, [1, 1])

        C1_common = C1_common = C[:, idx].copy()
        compatFlag = np.ones(shape=(C.shape[0], 1), dtype=bool)
        B = self.vars_[checkVars[0]].B
        C1 = C1_common[:, 0][np.newaxis, :]
        #x1_old = [B[k-1, :] for k in C1][0]
        x1 = [B[k-1, :] for k in C1[:, compatFlag.flatten()]][0]
        x2 = B[checkStates[0]-1, :]
        compatCheck = (np.sum(x1 * x2, axis=1) > 0)[:, np.newaxis]

        expected = np.array([[1,0],
                            [0,1],
                            [1,0],
                            [0,1],
                            [1,0],
                            [0,1],
                            [1,0],
                            [0,1],
                            [1,0],
                            [0,1],
                            [1,0],
                            [0,1],
                            [1,0],
                            [0,1],
                            [1,0],
                            [0,1]])
        np.testing.assert_array_equal(x1, expected)
        np.testing.assert_array_equal(x2, [1, 0])

        expected = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]).T
        np.testing.assert_array_equal(compatCheck, expected)

        compatFlag[:len(compatCheck)] = np.logical_and(compatFlag[:len(compatCheck)], compatCheck)

        # i = 1
        B = self.vars_[checkVars[1]].B
        C1 = C1_common[:, 1][np.newaxis, :]
        #x1_old = [B[k-1, :] for k in C1][0]
        x1 = [B[k-1, :] for k in C1[:, compatFlag.flatten()]][0]
        x2 = B[checkStates[1]-1, :]
        compatCheck = (np.sum(x1 * x2, axis=1) > 0)[:, np.newaxis]

        expected = np.array([[1, 1, 0, 0, 1, 1, 0, 0]]).T
        np.testing.assert_array_equal(compatCheck, expected)
        compatFlag[np.where(compatFlag > 0)[0][:len(compatCheck)]] = compatCheck
        np.testing.assert_array_equal(compatFlag, np.array([[1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]]).T)

    def test_getCpmSubset1(self):


        # M[5]
        rowIndex = [0]  # 1 -> 0
        result = getCpmSubset(self.M[5], rowIndex)

        np.testing.assert_array_equal(result.C, np.array([[2, 3, 3, 2]]))
        np.testing.assert_array_equal(result.p, [[1]])

    def test_getCpmSubset2(self):

        # M[5]
        rowIndex = [1, 2, 3]  # [2, 3, 4] -> 0
        result = getCpmSubset(self.M[5], rowIndex, 0)

        np.testing.assert_array_equal(result.C, np.array([[2, 3, 3, 2]]))
        np.testing.assert_array_equal(result.p, [[1]])

    def test_isCompatibleCpm1(self):

        # M[5]
        rowIndex = [0]  # 1 -> 0
        M_sys_select = getCpmSubset(self.M[5], rowIndex)
        result = isCompatibleCpm(self.M[3], M_sys_select, vInfo=self.vars_)
        expected = np.array([1, 1, 1, 1])[:, np.newaxis]
        np.testing.assert_array_equal(result, expected)

    def test_isCompatibleCpm2(self):

        # M[5]
        rowIndex = [0]  # 1 -> 0
        M_sys_select = getCpmSubset(self.M[5], rowIndex)

        result = isCompatibleCpm(self.M[4], M_sys_select, vInfo=self.vars_)
        expected = np.array([0, 1, 0, 1])[:, np.newaxis]
        np.testing.assert_array_equal(result, expected)

    def test_isCompatibleCpm3(self):

        # M[5]
        rowIndex = [0]  # 1 -> 0
        M_sys_select = getCpmSubset(self.M[5], rowIndex)

        result = isCompatibleCpm(self.M[1], M_sys_select, vInfo=self.vars_)
        expected = np.array([1, 1])[:, np.newaxis]
        np.testing.assert_array_equal(result, expected)

    @unittest.skip('NYI')
    def test_product(self):

        M1 = self.M[2]
        M2 = self.M[3]
        vInfo = self.vars_

        result = product(M1, M2, vInfo)

    def test_condition0(self):

        C = np.array([[1,1,1,1,1],
             [2,1,1,1,1],
             [1,2,1,1,1],
             [2,2,2,1,1],
             [1,1,1,2,1],
             [2,1,1,2,1],
             [1,2,1,2,1],
             [2,2,2,2,1],
             [1,1,2,1,2],
             [2,1,2,1,2],
             [1,2,2,1,2],
             [2,2,2,1,2],
             [1,1,2,2,2],
             [2,1,2,2,2],
             [1,2,2,2,2],
             [2,2,2,2,2]])
        p = np.array([0.9405, 0.0095, 0.0495, 0.0005, 0.7650, 0.0850, 0.1350, 0.0150, 0.9405, 0.0095, 0.0495, 0.0005, 0.7650, 0.0850, 0.1350, 0.0150])
        Mx = Cpm(variables=[2, 3, 5, 1, 4], numChild=3, C = C, p = p.T)
        condVars = np.array([2])
        condStates = np.array([1])
        vars_ = self.vars_

        compatFlag = isCompatible(Mx.C, Mx.variables, condVars, condStates, vars_)
        expected = np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])[:, np.newaxis]

        np.testing.assert_array_equal(expected, compatFlag)
        Ccompat = Mx.C[compatFlag.flatten(), :]
        expected = np.array([[1,1,1,1,1],
                            [1,2,1,1,1],
                            [1,1,1,2,1],
                            [1,2,1,2,1],
                            [1,1,2,1,2],
                            [1,2,2,1,2],
                            [1,1,2,2,2],
                            [1,2,2,2,2]])
        np.testing.assert_array_equal(expected, Ccompat)

        idxInC = np.array(ismember(condVars, Mx.variables))
        np.testing.assert_array_equal(idxInC, [0])  # matlab 1 though

        idxInCondVars = ismember(Mx.variables, condVars)
        np.testing.assert_array_equal(idxInCondVars, [0, False, False, False, False])  # matlab 1 though

        not_idxInCondVars = flip(idxInCondVars)
        self.assertEqual(not_idxInCondVars, [False, True, True, True, True])
        Ccond = np.zeros_like(Ccompat)
        Ccond[:, not_idxInCondVars] = get_value_given_condn(Ccompat, not_idxInCondVars)
        #np.testing.assert_array_equal(Ccond_, Ccompat[:, 1:])
        #Ccond[:, new_cond] = Ccond_
        expected = np.append(np.zeros((Ccompat.shape[0], 1)), Ccompat[:, 1:], axis=1)
        np.testing.assert_array_equal(Ccond, expected)

        _condVars = condVars[idxInC >= 0]
        _condStates = condStates[idxInC >= 0]
        _idxInC = idxInC[idxInC >= 0]
        self.assertEqual(_condVars, np.array([2]))
        self.assertEqual(_condStates, np.array([1]))
        self.assertEqual(_idxInC, np.array([0]))

        B = vars_[_condVars[0]].B
        np.testing.assert_array_equal(B, np.array([[1, 0], [0, 1], [1, 1]]))

        # FIXIT: index or not
        _Ccompat = Ccompat[:, _idxInC].copy() - 1
        #np.testing.assert_array_equal(_Ccompat, np.ones((8, 1)))
        np.testing.assert_array_equal(_Ccompat, np.zeros((8, 1)))

        expected = np.array([np.ones(8), np.zeros(8)]).T
        np.testing.assert_array_equal(B[_Ccompat.flatten(), :], expected)
        # FIXIT: index or not
        np.testing.assert_array_equal(B[_condStates - 1, :], np.array([[1, 0]]))
        #np.testing.assert_array_equal(B[_condStates, :], np.array([1, 0]))
        compatCheck_mv = B[_Ccompat.flatten(), :] * B[_condStates - 1, :]
        np.testing.assert_array_equal(compatCheck_mv, expected)

        B = addNewStates(compatCheck_mv, B)
        vars_[_condVars[0]].B = B

        # FIXIT: index or not
        Ccond[:, _idxInC[0]] = [x+1 for x in ismember(compatCheck_mv, B)]

        # Need to confirm whether 
        expected = np.array([[1,1,1,1,1],
                            [1,2,1,1,1],
                            [1,1,1,2,1],
                            [1,2,1,2,1],
                            [1,1,2,1,2],
                            [1,2,2,1,2],
                            [1,1,2,2,2],
                            [1,2,2,2,2]])
        np.testing.assert_array_equal(Ccond, expected)

        # Mx.p
        expected = np.array([[0.9405,0.0495,0.7650,0.1350,0.9405,0.0495,0.7650,0.1350]]).T
        np.testing.assert_array_equal(Mx.p[compatFlag][:, np.newaxis], expected)

    def test_condition1(self):

        C = np.array([[1,1,1,1,1],
             [2,1,1,1,1],
             [1,2,1,1,1],
             [2,2,2,1,1],
             [1,1,1,2,1],
             [2,1,1,2,1],
             [1,2,1,2,1],
             [2,2,2,2,1],
             [1,1,2,1,2],
             [2,1,2,1,2],
             [1,2,2,1,2],
             [2,2,2,1,2],
             [1,1,2,2,2],
             [2,1,2,2,2],
             [1,2,2,2,2],
             [2,2,2,2,2]])
        p = np.array([0.9405, 0.0095, 0.0495, 0.0005, 0.7650, 0.0850, 0.1350, 0.0150, 0.9405, 0.0095, 0.0495, 0.0005, 0.7650, 0.0850, 0.1350, 0.0150])
        Mx = Cpm(variables=[2, 3, 5, 1, 4], numChild=3, C = C, p = p.T)
        condVars = np.array([2])
        condStates = np.array([1])
        vars_ = self.vars_

        M_n, vars_n = condition([Mx], condVars, condStates, vars_)

        self.assertEqual(M_n[0].variables, [2, 3, 5, 1, 4])
        self.assertEqual(M_n[0].numChild, 3)
        expected = np.array([[1,1,1,1,1],
                            [1,2,1,1,1],
                            [1,1,1,2,1],
                            [1,2,1,2,1],
                            [1,1,2,1,2],
                            [1,2,2,1,2],
                            [1,1,2,2,2],
                            [1,2,2,2,2]])
        np.testing.assert_array_equal(M_n[0].C, expected)

        expected = np.array([[0.9405,0.0495,0.7650,0.1350,0.9405,0.0495,0.7650,0.1350]]).T
        np.testing.assert_array_equal(M_n[0].p, expected)


    def test_condition2(self):

        C = np.array([[1,1,1,1,1],
             [2,1,1,1,1],
             [1,2,1,1,1],
             [2,2,2,1,1],
             [1,1,1,2,1],
             [2,1,1,2,1],
             [1,2,1,2,1],
             [2,2,2,2,1],
             [1,1,2,1,2],
             [2,1,2,1,2],
             [1,2,2,1,2],
             [2,2,2,1,2],
             [1,1,2,2,2],
             [2,1,2,2,2],
             [1,2,2,2,2],
             [2,2,2,2,2]])
        p = np.array([0.9405, 0.0095, 0.0495, 0.0005, 0.7650, 0.0850, 0.1350, 0.0150, 0.9405, 0.0095, 0.0495, 0.0005, 0.7650, 0.0850, 0.1350, 0.0150])
        Mx = Cpm(variables=[2, 3, 5, 1, 4], numChild=3, C = C, p = p.T)
        condVars = np.array([1])
        condStates = np.array([1])
        vars_ = self.vars_

        M_n, vars_n = condition([Mx], condVars, condStates, vars_)

        self.assertEqual(M_n[0].variables, [2, 3, 5, 1, 4])
        self.assertEqual(M_n[0].numChild, 3)
        expected = np.array([[1,1,1,1,1],
                            [2,1,1,1,1],
                            [1,2,1,1,1],
                            [2,2,2,1,1],
                            [1,1,2,1,2],
                            [2,1,2,1,2],
                            [1,2,2,1,2],
                            [2,2,2,1,2]])

        np.testing.assert_array_equal(M_n[0].C, expected)

        expected = np.array([[0.9405,0.0095,0.0495,0.0005,0.9405,0.0095,0.0495,0.0005]]).T
        np.testing.assert_array_equal(M_n[0].p, expected)

    def test_condition3(self):

        C = np.array([[1,1,1,1,1],
             [2,1,1,1,1],
             [1,2,1,1,1],
             [2,2,2,1,1],
             [1,1,1,2,1],
             [2,1,1,2,1],
             [1,2,1,2,1],
             [2,2,2,2,1],
             [1,1,2,1,2],
             [2,1,2,1,2],
             [1,2,2,1,2],
             [2,2,2,1,2],
             [1,1,2,2,2],
             [2,1,2,2,2],
             [1,2,2,2,2],
             [2,2,2,2,2]])
        p = np.array([0.9405, 0.0095, 0.0495, 0.0005, 0.7650, 0.0850, 0.1350, 0.0150, 0.9405, 0.0095, 0.0495, 0.0005, 0.7650, 0.0850, 0.1350, 0.0150])
        Mx = Cpm(variables=[2, 3, 5, 1, 4], numChild=3, C = C, p = p.T)
        condVars = np.array([2, 1])
        condStates = np.array([1, 1])
        vars_ = self.vars_

        M_n, vars_n = condition([Mx], condVars, condStates, vars_)

        self.assertEqual(M_n[0].variables, [2, 3, 5, 1, 4])
        self.assertEqual(M_n[0].numChild, 3)
        expected = np.array([[1,1,1,1,1],
                            [1,2,1,1,1],
                            [1,1,2,1,2],
                            [1,2,2,1,2]])

        np.testing.assert_array_equal(M_n[0].C, expected)

        expected = np.array([[0.9405,0.0495,0.9405,0.0495]]).T
        np.testing.assert_array_equal(M_n[0].p, expected)

    @unittest.skip('NW')
    def test_condition4(self):

        C = np.array([[1,1,1,1,1],
             [2,1,1,1,1],
             [1,2,1,1,1],
             [2,2,2,1,1],
             [1,1,1,2,1],
             [2,1,1,2,1],
             [1,2,1,2,1],
             [2,2,2,2,1],
             [1,1,2,1,2],
             [2,1,2,1,2],
             [1,2,2,1,2],
             [2,2,2,1,2],
             [1,1,2,2,2],
             [2,1,2,2,2],
             [1,2,2,2,2],
             [2,2,2,2,2]])
        p = np.array([0.9405, 0.0095, 0.0495, 0.0005, 0.7650, 0.0850, 0.1350, 0.0150, 0.9405, 0.0095, 0.0495, 0.0005, 0.7650, 0.0850, 0.1350, 0.0150])
        Mx = Cpm(variables=[2, 3, 5, 1, 4], numChild=3, C = C, p = p.T)
        condVars = np.array([2, 3])
        condStates = np.array([1, 1])
        vars_ = self.vars_

        M_n, vars_n = condition([Mx], condVars, condStates, vars_)

        self.assertEqual(M_n[0].variables, [2, 3, 5, 1, 4])
        self.assertEqual(M_n[0].numChild, 3)
        expected = np.array([[2,1,1,2],
                             [1,1,1,1]])

        np.testing.assert_array_equal(M_n[0].C, expected)

        expected = np.array([[1, 1]]).T
        np.testing.assert_array_equal(M_n[0].p, expected)

    def test_addNewStates(self):
        states = np.array([np.ones(8), np.zeros(8)]).T
        B = np.array([[1, 0], [0, 1], [1, 1]])

        newStateCheck = ismember(states, B)
        expected = [0, 0, 0, 0, 0, 0, 0, 0]
        self.assertEqual(newStateCheck, expected)

        newStateCheck = flip(newStateCheck)
        np.testing.assert_array_equal(newStateCheck, np.zeros_like(newStateCheck, dtype=bool))
        newState = states[newStateCheck, :]
        np.testing.assert_array_equal(newState, np.empty(shape=(0, 2)))
        #B = np.append(B, newState, axis=1)

        result = addNewStates(states, B)
        np.testing.assert_array_equal(result, B)

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

    @unittest.skip('NYI')
    def test_get_varsRemain0(self):
        #sumFlag = 0
        sumVars = 2
        expected = np.array([1, 2])
        Msys = Cpm(**self.kwargs)
        #result = get_varsRemain(Msys, sumVars, 0)

        #self.assertEqual(expected, result)

if __name__=='__main__':
    unittest.main()

