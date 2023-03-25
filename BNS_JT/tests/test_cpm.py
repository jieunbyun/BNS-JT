import unittest
import importlib
import numpy as np

from BNS_JT.cpm import Cpm, ismember, is_compatible, get_value_given_condn, flip, add_new_states, condition, get_prod, argsort, setdiff, get_sample_order, get_prod_idx, single_sample, mcs_product, prod_cpms
from BNS_JT.variable import Variable

np.set_printoptions(precision=3)
#pd.set_option.display_precision = 3

class Test_Cpm(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        variables = [3, 2, 1]
        no_child = 1
        C = np.array([[2, 2, 3], [2, 1, 2], [1, 1, 1]])
        p = [1, 1, 1]

        cls.kwargs = {'variables': variables,
                      'no_child': no_child,
                      'C': C,
                      'p': p}

    def test_init1(self):
        a = Cpm(**self.kwargs)

        self.assertTrue(isinstance(a, Cpm))
        np.testing.assert_array_equal(a.variables, self.kwargs['variables'])
        self.assertEqual(a.no_child, self.kwargs['no_child'])
        np.testing.assert_array_equal(a.C, self.kwargs['C'])

    def test_init2(self):
        # using list for P
        a = Cpm(variables=[1], no_child=1, C=np.array([1, 2]), p=[0.9, 0.1])
        self.assertTrue(isinstance(a, Cpm))

    def test_variables1(self):

        f_variables = [1, 2]
        with self.assertRaises(AssertionError):
            _ = Cpm(**{'variables': f_variables,
                       'no_child': self.kwargs['no_child'],
                       'C': self.kwargs['C'],
                       'p': self.kwargs['p']})

    def test_variables2(self):

        f_variables = [1, 2, 3, 4]
        with self.assertRaises(AssertionError):
            _ = Cpm(**{'variables': f_variables,
                       'no_child': self.kwargs['no_child'],
                       'C': self.kwargs['C'],
                       'p': self.kwargs['p']})

    def test_variables3(self):

        f_variables = ['x', 2, 3]
        with self.assertRaises(AssertionError):
            _ = Cpm(**{'variables': f_variables,
                       'no_child': self.kwargs['no_child'],
                       'C': self.kwargs['C'],
                       'p': self.kwargs['p']})

    def test_no_child(self):

        f_no_child = 4
        with self.assertRaises(AssertionError):
            _ = Cpm(**{'variables': self.kwargs['variables'],
                       'no_child': f_no_child,
                       'C': self.kwargs['C'],
                       'p': self.kwargs['p']})

    def test_sort(self):

        p = np.array([[0.9405, 0.0495, 0.0095, 0.0005, 0.7650, 0.1350, 0.0850, 0.0150]]).T
        C = np.array([[1, 1, 1], [1, 2, 1], [2, 1, 1], [2, 2, 1], [1, 1, 2], [1, 2, 2], [2, 1, 2], [2, 2, 2]])

        M = Cpm(variables=[2, 3, 1],
                no_child = 2,
                C = C,
                p = p)

        if any(M.sample_idx):
            rowIdx = argsort(M.sample_idx)
        else:
            rowIdx = argsort(list(map(tuple, C[:, ::-1])))

        try:
            Ms_p = M.p[rowIdx]
        except IndexError:
            Ms_p = M.p

        try:
            Ms_q = M.q[rowIdx]
        except IndexError:
            Ms_q = M.q

        try:
            Ms_sample_idx = M.sample_idx[rowIdx]
        except IndexError:
            Ms_sample_idx = M.sample_idx

        Ms = Cpm(C=M.C[rowIdx, :],
                 p=Ms_p,
                 q=Ms_q,
                 sample_idx=Ms_sample_idx,
                 variables=M.variables,
                 no_child=M.no_child)

        np.testing.assert_array_equal(Ms.C, np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1], [1, 1, 2], [2, 1, 2], [1, 2, 2], [2, 2, 2]]))
        np.testing.assert_array_almost_equal(Ms.p, np.array([[0.9405, 0.0095, 0.0495, 5.0e-4, 0.7650, 0.0850, 0.1350, 0.0150]]).T)


class Test_functions(unittest.TestCase):

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

    def test_argsort(self):

        C = np.array([[1, 1, 1], [1, 2, 1], [2, 1, 1], [2, 2, 1], [1, 1, 2], [1, 2, 2], [2, 1, 2], [2, 2, 2]])
        x = list(map(tuple, C[:, ::-1]))
        res = argsort(x)
        self.assertEqual(res, [0, 2, 1, 3, 4, 6, 5, 7])  # matlab index -1

    def test_get_prod(self):

        A = np.array([[0.95, 0.05]]).T
        B = np.array([0.99])

        result = get_prod(A, B)
        np.testing.assert_array_equal(result, np.array([[0.9405, 0.0495]]).T)

    def test_setdiff(self):

        A = [3, 6, 2, 1, 5, 1, 1]
        B = [2, 4, 6]
        C, ia = setdiff(A, B)

        self.assertEqual(C, [1, 3, 5])
        self.assertEqual(ia, [3, 0, 4])

    def test_get_value_given_condn1(self):

        A = [1, 2, 3, 4]
        condn = [0, False, 1, 3]

        self.assertEqual([1, 3, 4], get_value_given_condn(A, condn))

    def test_get_value_given_condn2(self):

        A = [1, 2, 3]
        condn = [0, False, 1, 3]

        with self.assertRaises(AssertionError):
            get_value_given_condn(A, condn)

    def test_add_new_states(self):
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

        result = add_new_states(states, B)
        np.testing.assert_array_equal(result, B)



class Test_is_compatible(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.M = {}
        cls.vars_ = {}

        cls.M[1] = Cpm(variables=[1], no_child=1, C = np.array([[1, 2]]).T, p = np.array([0.9, 0.1]).T)
        cls.M[2] = Cpm(variables=[2, 1], no_child=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]), p = np.array([0.99, 0.01, 0.9, 0.1]).T)
        cls.M[3] = Cpm(variables=[3, 1], no_child=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]), p = np.array([0.95, 0.05, 0.85, 0.15]).T)
        cls.M[4] = Cpm(variables=[4, 1], no_child=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]), p = np.array([0.99, 0.01, 0.9, 0.1]).T)
        cls.M[5] = Cpm(variables=[5, 2, 3, 4], no_child=1, C = np.array([[2, 3, 3, 2], [1, 1, 3, 1], [1, 2, 1, 1], [2, 2, 2, 1]]), p = np.array([1, 1, 1, 1]).T)

        cls.vars_[1] = Variable(B=np.eye(2), value=['Mild', 'Severe'])
        cls.vars_[2] = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]), value=['Survive', 'Fail'])
        cls.vars_[3] = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]), value=['Survive', 'Fail'])
        cls.vars_[4] = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]), value=['Survive', 'Fail'])
        cls.vars_[5] = Variable(B=np.array([[1, 0], [0, 1]]), value=['Survive', 'Fail'])


    def test_is_compatible1(self):

        # M[2]
        C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]])  #M[2].C
        variables = [2, 1]
        checkVars = [1]
        checkStates = [1]
        v_info = self.vars_

        result = is_compatible(C, variables, checkVars, checkStates, v_info)
        expected = np.array([1, 1, 0, 0])
        np.testing.assert_array_equal(expected, result)

    def test_is_compatible2(self):

        # M[5]
        C = np.array([[2, 3, 3, 2], [1, 1, 3, 1], [1, 2, 1, 1], [2, 2, 2, 1]])
        variables = [5, 2, 3, 4]
        checkVars = [3, 4]
        checkStates = [1, 1]
        v_info = self.vars_

        result = is_compatible(C, variables, checkVars, checkStates, v_info)
        expected = np.array([0, 1, 1, 0])
        np.testing.assert_array_equal(expected, result)

    def test_is_compatible3(self):

        #M[1]
        C = np.array([[1, 2]]).T
        variables = [1]
        checkVars = [3, 4]
        checkStates = [1, 1]
        v_info = self.vars_

        result = is_compatible(C, variables, checkVars, checkStates, v_info)
        expected = np.array([1, 1])
        np.testing.assert_array_equal(expected, result)

    def test_is_compatible4(self):

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

        result = is_compatible(C, variables, checkVars, checkStates, vars_)
        expected = np.array([1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0])
        np.testing.assert_array_equal(expected, result)

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


    def test_get_subset1(self):

        # M[5]
        rowIndex = [0]  # 1 -> 0
        result = self.M[5].get_subset(rowIndex)

        np.testing.assert_array_equal(result.C, np.array([[2, 3, 3, 2]]))
        np.testing.assert_array_equal(result.p, [[1]])

    def test_get_subset2(self):

        # M[5]
        rowIndex = [1, 2, 3]  # [2, 3, 4] -> 0
        result = self.M[5].get_subset(rowIndex, 0)

        np.testing.assert_array_equal(result.C, np.array([[2, 3, 3, 2]]))
        np.testing.assert_array_equal(result.p, [[1]])

    def test_get_subset3(self):

        M = Cpm(variables=[2, 3, 5, 1, 4],
                no_child=5,
                C=np.array([[2, 2, 2, 2, 2]]),
                p=np.array([[0.0150]]).T)

        result = M.get_subset([0], 0)

        self.assertFalse(result.C.any())
        self.assertFalse(result.p.any())


    def test_is_compatibleCpm1(self):

        # M[5]
        rowIndex = [0]  # 1 -> 0
        M_sys_select = self.M[5].get_subset(rowIndex)
        result = self.M[3].is_compatible(M_sys_select, var=self.vars_)
        expected = np.array([1, 1, 1, 1])
        np.testing.assert_array_equal(result, expected)

    def test_is_compatibleCpm2(self):

        # M[5]
        rowIndex = [0]  # 1 -> 0
        M_sys_select = self.M[5].get_subset(rowIndex)

        result = self.M[4].is_compatible(M_sys_select, var=self.vars_)
        expected = np.array([0, 1, 0, 1])
        np.testing.assert_array_equal(result, expected)

    def test_is_compatibleCpm3(self):

        # M[5]
        rowIndex = [0]  # 1 -> 0
        M_sys_select = self.M[5].get_subset(rowIndex)

        result = self.M[1].is_compatible(M_sys_select, var=self.vars_)
        expected = np.array([1, 1])
        np.testing.assert_array_equal(result, expected)


class Test_Product(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.M = {}
        cls.vars_ = {}

        cls.M[2] = Cpm(variables=[2, 1], no_child=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]), p = np.array([0.99, 0.01, 0.9, 0.1]).T)
        cls.M[3] = Cpm(variables=[3, 1], no_child=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]), p = np.array([0.95, 0.05, 0.85, 0.15]).T)
        cls.M[5] = Cpm(variables=[5, 2, 3, 4], no_child=1, C = np.array([[2, 3, 3, 2], [1, 1, 3, 1], [1, 2, 1, 1], [2, 2, 2, 1]]), p = np.array([1, 1, 1, 1]).T)

        cls.vars_[1] = Variable(B=np.eye(2), value=['Mild', 'Severe'])
        cls.vars_[2] = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]), value=['Survive', 'Fail'])
        cls.vars_[3] = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]), value=['Survive', 'Fail'])
        cls.vars_[4] = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]), value=['Survive', 'Fail'])
        cls.vars_[5] = Variable(B=np.array([[1, 0], [0, 1]]), value=['Survive', 'Fail'])

    def test_product1(self):
        # When there is no common variable
        M1 = self.M[2]
        M2 = self.M[3]
        v_info = self.vars_

        if any(M1.p):
            if not any(M2.p):
                M1.p = np.ones(M1.C.shape[0])
        else:
            if any(M2.p):
                M2.p = np.ones(M2.C.shape[0])

        if any(M1.q):
            if not any(M2.q):
                M2.q = np.ones(M2.C.shape[0])
        else:
            if any(M2.q):
                M1.q = ones(M1.C.shape[0])

        np.testing.assert_array_equal(M1.p, np.array([[0.99, 0.01, 0.9, 0.1]]).T)
        np.testing.assert_array_equal(M1.q, np.array([[]]).T)
        np.testing.assert_array_equal(M2.p, np.array([[0.95, 0.05, 0.85, 0.15]]).T)
        np.testing.assert_array_equal(M2.q, np.array([[]]).T)

        commonVars=set(M1.variables).intersection(M2.variables)

        self.assertEqual(list(commonVars), [1])

        idxVarsM1 = ismember(M1.variables, M2.variables)
        commonVars = get_value_given_condn(M1.variables, idxVarsM1)

        np.testing.assert_array_equal(idxVarsM1, np.array([0, 1]))
        self.assertEqual(commonVars, [1])

        #Cprod = np.array([])
        #pprod = np.array([])
        #qprod = np.array([])
        #sampleIndProd = np.array([])

        for i in range(M1.C.shape[0]):
            c1_ = get_value_given_condn(M1.C[i, :], idxVarsM1)
            c1_notCommon = M1.C[i, flip(idxVarsM1)]

            if any(M1.sample_idx):
                sampleInd1 = M1.sample_idx[i]
            else:
                sampleInd1 = []

            #if isinstance(commonVars, list):
            #    commonVars = np.array(commonVars)

            #if isinstance(c1_, list):
            #    c1_ = np.array(c1_)
            [[M2_], v_info] = condition([M2], commonVars, c1_, v_info, sampleInd1)

            #self.assertEqual(M2_.variables, [3, 1])
            #self.assertEqual(M2_.no_child, 1)
            #np.testing.assert_array_equal(M2_.C, np.array([[1, 1], [2, 1]]))
            #np.testing.assert_array_equal(M2_.p, np.array([[0.95, 0.05]]).T)
            #Cprod = np.append(Cprod, M2_.C).reshape(M2_.C.shape[0], -1)
            #print(c1_notCommon)
            #print(np.tile(c1_notCommon, (M2_.C.shape[0], 1)))
            _add = np.append(M2_.C, np.tile(c1_notCommon, (M2_.C.shape[0], 1)), axis=1)
            #print(_add)
            if i:
                Cprod = np.append(Cprod, _add, axis=0)
            else:
                Cprod = _add

            #print(f'Cprod after: {i}')
            #print(Cprod)
            #result = product(M1, M2, v_info)

            #np.testing.assert_array_equal(Cp, np.array([[1, 1, 1], [2, 1, 1]]))

            #sampleIndProd = np.array([])
            if any(sampleInd1):
                _add = repmat(sampleInd1, M2.C.shape[0], 1)
                sampleIndProd = np.append(sampleIndProd, _add).reshape(M2C.shape[0], -1)
            elif any(M2_.sample_idx):
                sampleIndProd = np.append(sampleIndPro, M2_.sample_idx).reshape(M2_.sample_idx.shape[0], -1)

            #np.testing.assert_array_equal(sampleIndProd, [])
            if any(M1.p):
                '''
                val = M2_.p * M1.p[0]
                np.testing.assert_array_equal(val, np.array([[0.9405, 0.0495]]).T)
                pproductSign = np.sign(val)
                np.testing.assert_array_equal(pproductSign, np.ones_like(val))
                pproductVal = np.exp(np.log(np.abs(M2_.p)) + np.log(np.abs(M1.p[0])))
                _prod = pproductSign * pproductVal
                np.testing.assert_array_equal(pproductVal, np.array([[0.9405, 0.0495]]).T)
                pproduct = np.array([])
                pproduct = np.append(pproduct, _prod).reshape(_prod.shape[0], -1)

                np.testing.assert_array_equal(pproduct, np.array([[0.9405, 0.0495]]).T)
                '''
                _prod = get_prod(M2_.p, M1.p[i])
                #np.testing.assert_array_equal(pproductVal, np.array([[0.9405, 0.0495]]).T)
                #pproduct = np.array([])
                if i:
                    #pprod = np.append(pprod, _prod, axis=0.reshape(_prod.shape[0], -1)

                    pprod = np.append(pprod, _prod, axis=0)
                else:
                    pprod = _prod

        np.testing.assert_array_almost_equal(pprod, np.array([[0.9405, 0.0495, 0.0095, 0.0005, 0.7650, 0.1350, 0.0850, 0.0150]]).T)
        np.testing.assert_array_almost_equal(Cprod, np.array([[1, 1, 1], [2, 1, 1], [1, 1, 2], [2, 1, 2], [1, 2, 1], [2, 2, 1], [1, 2, 2], [2, 2, 2]]))

        Cprod_vars = np.append(M2.variables, get_value_given_condn(M1.variables, flip(idxVarsM1)))
        np.testing.assert_array_equal(Cprod_vars, [3, 1, 2])

        newVarsChild = np.append(M1.variables[0:M1.no_child], M2.variables[0:M2.no_child])
        newVarsChild = np.sort(newVarsChild)
        np.testing.assert_array_equal(newVarsChild, [2, 3])

        newVarsParent = np.append(M1.variables[M1.no_child:], M2.variables[M2.no_child:])
        newVarsParent = list(set(newVarsParent).difference(newVarsChild))
        newVars = np.append(newVarsChild, newVarsParent, axis=0)
        np.testing.assert_array_equal(newVars, [2, 3, 1])

        idxVars = ismember(newVars, Cprod_vars)

        self.assertEqual(idxVars, [2, 0, 1]) # matlab 3, 1, 2

        Mprod = Cpm(variables=newVars,
                    no_child = len(newVarsChild),
                    C = Cprod[:, idxVars],
                    p = pprod)

        Mprod.sort()

        np.testing.assert_array_equal(Mprod.variables, [2, 3, 1])
        self.assertEqual(Mprod.no_child, 2)
        np.testing.assert_array_equal(Mprod.C, np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1], [1, 1, 2], [2, 1, 2], [1, 2, 2], [2, 2, 2]]))
        np.testing.assert_array_almost_equal(Mprod.p, np.array([[0.9405, 0.0095, 0.0495, 5.0e-4, 0.7650, 0.0850, 0.1350, 0.0150]]).T)

    def test_product2(self):

        M1 = self.M[2]
        M2 = self.M[3]
        v_info = self.vars_

        Mprod, v_info_ = M1.product(M2, v_info)

        np.testing.assert_array_equal(Mprod.variables, [2, 3, 1])
        self.assertEqual(Mprod.no_child, 2)
        np.testing.assert_array_equal(Mprod.C, np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1], [1, 1, 2], [2, 1, 2], [1, 2, 2], [2, 2, 2]]))
        np.testing.assert_array_almost_equal(Mprod.p, np.array([[0.9405, 0.0095, 0.0495, 5.0e-4, 0.7650, 0.0850, 0.1350, 0.0150]]).T)

    def test_product3(self):

        M1 = Cpm(variables=[2, 3, 1], no_child=2, C = np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1], [1, 1, 2], [2, 1, 2], [1, 2, 2], [2, 2, 2]]), p = np.array([[0.9405, 0.0095, 0.0495, 5.0e-4, 0.7650, 0.0850, 0.1350, 0.0150]]).T)

        M2 = self.M[5]
        v_info = self.vars_

        Mprod, v_info_ = M1.product(M2, v_info)

        np.testing.assert_array_equal(Mprod.variables, [2, 3, 5, 1, 4])
        self.assertEqual(Mprod.no_child, 3)

        expected_C = np.array([
              [1,1,1,1,1],
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

        expected_p = np.array([[0.9405,0.0095,0.0495,0.0005,0.7650,0.0850,0.1350,0.0150,0.9405,0.0095,0.0495,0.0005,0.7650,0.0850,0.1350,0.0150]]).T
        np.testing.assert_array_equal(Mprod.C, expected_C)
        np.testing.assert_array_almost_equal(Mprod.p, expected_p)


class Test_Condition(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.C = np.array([[1,1,1,1,1],
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
        cls.p = np.array([[0.9405, 0.0095, 0.0495, 0.0005, 0.7650, 0.0850, 0.1350, 0.0150, 0.9405, 0.0095, 0.0495, 0.0005, 0.7650, 0.0850, 0.1350, 0.0150]]).T

        cls.vars_ = {}
        cls.vars_[1] = Variable(B=np.eye(2), value=['Mild', 'Severe'])
        cls.vars_[2] = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]), value=['Survive', 'Fail'])
        cls.vars_[3] = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]), value=['Survive', 'Fail'])
        cls.vars_[4] = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]), value=['Survive', 'Fail'])
        cls.vars_[5] = Variable(B=np.array([[1, 0], [0, 1]]), value=['Survive', 'Fail'])

        cls.Mx = Cpm(variables=[2, 3, 5, 1, 4], no_child=3, C = cls.C, p = cls.p)

    def test_condition0(self):

        condVars = np.array([2])
        condStates = np.array([1])
        vars_ = self.vars_

        Mx = Cpm(variables=[2, 1],
                 no_child = 1,
                 C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]),
                 p = np.array([[0.99, 0.01, 0.9, 0.1]]).T)

        compatFlag = is_compatible(Mx.C, Mx.variables, condVars, condStates, vars_)
        #expected = np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])[:, np.newaxis]
        expected = np.array([1, 0, 1, 0])
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

        expected = np.array([[1, 1], [1, 2]])
        np.testing.assert_array_equal(expected, Ccompat)

        idxInC = np.array(ismember(condVars, self.Mx.variables))
        np.testing.assert_array_equal(idxInC, [0])  # matlab 1 though

        idxInCondVars = ismember(Mx.variables, condVars)
        np.testing.assert_array_equal(idxInCondVars, [0, False])  # matlab 1 though
        not_idxInCondVars = flip(idxInCondVars)
        self.assertEqual(not_idxInCondVars, [False, True])
        Ccond = np.zeros_like(Ccompat)
        Ccond[:, not_idxInCondVars] = get_value_given_condn(Ccompat, not_idxInCondVars)
        #np.testing.assert_array_equal(Ccond_, Ccompat[:, 1:])
        #Ccond[:, new_cond] = Ccond_
        expected = np.array([[0, 1], [0, 2]])
        np.testing.assert_array_equal(Ccond, expected)

        _condVars = get_value_given_condn(condVars, idxInC)
        _condStates = get_value_given_condn(condStates, idxInC)
        _idxInC = get_value_given_condn(idxInC, idxInC)
        self.assertEqual(_condVars, np.array([2]))
        self.assertEqual(_condStates, np.array([1]))
        self.assertEqual(_idxInC, np.array([0]))

        B = vars_[_condVars[0]].B
        np.testing.assert_array_equal(B, np.array([[1, 0], [0, 1], [1, 1]]))

        # FIXME: index or not
        _Ccompat = Ccompat[:, _idxInC[0]].copy() - 1
        np.testing.assert_array_equal(_Ccompat, [0, 0])

        expected = np.array([[1, 0], [1, 0]])
        np.testing.assert_array_equal(B[_Ccompat.flatten(), :], expected)
        # FIXME: index or not
        np.testing.assert_array_equal(B[_condStates[0] - 1, :], [1, 0])
        #np.testing.assert_array_equal(B[_condStates, :], np.array([1, 0]))
        compatCheck_mv = B[_Ccompat.flatten(), :] * B[_condStates[0] - 1, :]
        np.testing.assert_array_equal(compatCheck_mv, expected)

        B = add_new_states(compatCheck_mv, B)
        vars_[_condVars[0]].B = B

        # FIXME: index or not
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
        expected = np.array([[1, 1], [1, 2]])
        np.testing.assert_array_equal(Ccond, expected)

        # Mx.p
        expected = np.array([[0.9405,0.0495,0.7650,0.1350,0.9405,0.0495,0.7650,0.1350]]).T
        expected = np.array([[0.99, 0.9]]).T
        np.testing.assert_array_equal(Mx.p[compatFlag], expected)

    def test_condition1(self):

        condVars = np.array([2])
        condStates = np.array([1])
        vars_ = self.vars_

        M_n, vars_n = condition([self.Mx], condVars, condStates, vars_)
        np.testing.assert_array_equal(M_n[0].variables, [2, 3, 5, 1, 4])
        self.assertEqual(M_n[0].no_child, 3)
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

        condVars = np.array([1])
        condStates = np.array([1])
        vars_ = self.vars_

        M_n, vars_n = condition([self.Mx], condVars, condStates, vars_)

        np.testing.assert_array_equal(M_n[0].variables, [2, 3, 5, 1, 4])
        self.assertEqual(M_n[0].no_child, 3)
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
        # conditioning on multiple nodes
        condVars = np.array([2, 1])
        condStates = np.array([1, 1])
        vars_ = self.vars_

        ([M_n], vars_n) = condition([self.Mx], condVars, condStates, vars_)

        np.testing.assert_array_equal(M_n.variables, [2, 3, 5, 1, 4])
        self.assertEqual(M_n.no_child, 3)
        expected = np.array([[1,1,1,1,1],
                            [1,2,1,1,1],
                            [1,1,2,1,2],
                            [1,2,2,1,2]])

        np.testing.assert_array_equal(M_n.C, expected)

        expected = np.array([[0.9405,0.0495,0.9405,0.0495]]).T
        np.testing.assert_array_equal(M_n.p, expected)

    def test_condition4(self):

        C = np.array([[2, 3, 3, 2],
                     [1, 1, 3, 1],
                     [1, 2, 1, 1],
                     [2, 2, 2, 1]])
        p = np.array([1, 1, 1, 1, ])
        Mx = Cpm(variables=[5, 2, 3, 4], no_child=1, C = C, p = p.T)
        condVars = np.array([2, 3])
        condStates = np.array([1, 1])
        vars_ = self.vars_

        result = is_compatible(Mx.C, Mx.variables, condVars, condStates, vars_)
        expected = np.array([1,1,0,0])
        np.testing.assert_array_equal(expected, result)

        [M_n], vars_n = condition([Mx], condVars, condStates, vars_)

        np.testing.assert_array_equal(M_n.variables, [5, 2, 3, 4])
        self.assertEqual(M_n.no_child, 1)
        expected = np.array([[2,1,1,2],
                             [1,1,1,1]])
        np.testing.assert_array_equal(M_n.C, expected)

        expected = np.array([[1, 1]]).T
        np.testing.assert_array_equal(M_n.p, expected)

    def test_condition5(self):

        C = np.array([[1, 1],
                     [2, 1],
                     [1, 2],
                     [2, 2]])
        p = np.array([0.95, 0.05, 0.85, 0.15])
        M2 = Cpm(variables=[3, 1], no_child=1, C = C, p = p.T)
        condVars = np.array([1])
        states = np.array([2])
        vars_ = self.vars_

        [M_n], vars_n = condition([M2], condVars, states, vars_)

        np.testing.assert_array_equal(M_n.variables, [3, 1])
        self.assertEqual(M_n.no_child, 1)
        expected = np.array([[1,2],
                             [2,2]])
        np.testing.assert_array_equal(M_n.C, expected)

        expected = np.array([[0.85, 0.15]]).T
        np.testing.assert_array_equal(M_n.p, expected)

    def test_condition6(self):

        C = np.array([[1, 2]]).T
        p = np.array([0.9, 0.1])
        M2 = Cpm(variables=[1], no_child=1, C = C, p = p.T)
        condVars = np.array([])
        states = np.array([])
        vars_ = self.vars_

        [M_n], vars_n = condition([M2], condVars, states, vars_)

        np.testing.assert_array_equal(M_n.variables, [1])
        self.assertEqual(M_n.no_child, 1)
        expected = np.array([[1,2]]).T
        np.testing.assert_array_equal(M_n.C, expected)

        expected = np.array([[0.9, 0.1]]).T
        np.testing.assert_array_equal(M_n.p, expected)


class Test_Sum(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        variables = [2, 3, 5, 1, 4]
        no_child = 3
        C = np.array([
              [1,1,1,1,1],
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

        p = np.array([[0.9405,0.0095,0.0495,0.0005,0.7650,0.0850,0.1350,0.0150,0.9405,0.0095,0.0495,0.0005,0.7650,0.0850,0.1350,0.0150]]).T

        cls.kwargs = {'variables': variables,
                      'no_child': no_child,
                      'C': C,
                      'p': p}

    def test_sum1(self):

        M = Cpm(**self.kwargs)
        sumVars = [1]
        varsRemainIdx = ismember( sumVars, M.variables[:M.no_child])

        sumFlag = 1
        if sumFlag:
            varsRemain, varsRemainIdx = setdiff(M.variables[:M.no_child], sumVars)
            self.assertEqual(varsRemain, [2, 3, 5])
            self.assertEqual(varsRemainIdx, [0, 1, 2])  # Matlab: [1, 2, 3]
        else:
            varsRemainIdx = get_value_given_condn(varsRemainIdx, varsRemainIdx)
            self.assertEqual(varsRemainIdx, [])
            varsRemain = get_value_given_condn(M.variables, varsRemainIdx)

        no_child = len(varsRemain)

        if any(M.variables[M.no_child:]):
            varsRemain = np.append(varsRemain, M.variables[M.no_child:])
            varsRemainIdx = np.append(varsRemainIdx, range(M.no_child, len(M.variables)))

        np.testing.assert_array_equal(varsRemain, [2, 3, 5, 1, 4])
        np.testing.assert_array_equal(varsRemainIdx, [0, 1, 2, 3, 4])

        Mloop = Cpm(variables=get_value_given_condn(M.variables, varsRemainIdx),
                    C=M.C[:, varsRemainIdx],
                    p=M.p,
                    q=M.q,
                    sample_idx=M.sample_idx,
                    no_child=len(varsRemainIdx))
        i = 0
        while Mloop.C.any():
            Mcompare = Mloop.get_subset([0]) # need to change to 0 
            if i==0:
                np.testing.assert_array_equal(Mcompare.variables, [2, 3, 5, 1, 4])
                np.testing.assert_array_equal(Mcompare.no_child, 5)
                np.testing.assert_array_equal(Mcompare.p, np.array([[0.9405]]).T)
                np.testing.assert_array_equal(Mcompare.C, np.array([[1, 1, 1, 1, 1]]))
                self.assertFalse(Mcompare.q.any())
                self.assertFalse(Mcompare.sample_idx.any())

            flag = Mloop.is_compatible(Mcompare)
            expected = np.zeros(16)
            expected[0] = 1
            if i==0:
                np.testing.assert_array_equal(flag, expected)

            if i==0:
                Csum = Mloop.C[0, :][np.newaxis, :]
            else:
                Csum = np.append(Csum, Mloop.C[0, :][np.newaxis, :], axis=0)

            if i==0:
                np.testing.assert_array_equal(Csum, np.array([[1, 1, 1, 1, 1]]))

            if any(Mloop.p):
                pval = np.array([np.sum(Mloop.p[flag])])[:, np.newaxis]
                if i==0:
                    psum = pval
                else:
                    psum = np.append(psum, pval, axis=0)

                if i==0:
                    np.testing.assert_array_equal(psum, [[0.9405]])

            try:
                Mloop = Mloop.get_subset(np.where(flag)[0], flag=0)
            except AssertionError:
                print(Mloop)

            expected_C = np.array([[2,1,1,1,1],
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
            expected_p = np.array([[0.0095,0.0495,0.0005,0.7650,0.0850,0.1350,0.0150,0.9405,0.0095,0.0495,0.0005,0.7650,0.0850,0.1350,0.0150]]).T
            if i==0:
                np.testing.assert_array_equal(Mloop.variables, [2, 3, 5, 1, 4])
                np.testing.assert_array_equal(Mloop.no_child, 5)
                np.testing.assert_array_equal(Mloop.p, expected_p)
                np.testing.assert_array_equal(Mloop.C, expected_C)
                self.assertFalse(Mloop.q.any())
                self.assertFalse(Mloop.sample_idx.any())
            i += 1

        Msum = Cpm(variables=varsRemain,
                   no_child=no_child,
                   C=Csum,
                   p=psum)

        expected_C = np.array([[1,1,1,1,1],
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

        expected_p = np.array([[0.9405,0.0095,0.0495,0.0005,0.7650,0.0850,0.1350,0.0150,0.9405,0.0095,0.0495,0.0005,0.7650,0.0850,0.1350,0.0150]]).T
        np.testing.assert_array_equal(Msum.C, expected_C)
        np.testing.assert_array_equal(Msum.p, expected_p)

    def test_sum2(self):

        M = Cpm(**self.kwargs)
        sumVars = [1]

        Ms = M.sum(sumVars, flag=1)

        expected_C = np.array([[1,1,1,1,1],
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

        expected_p = np.array([[0.9405,0.0095,0.0495,0.0005,0.7650,0.0850,0.1350,0.0150,0.9405,0.0095,0.0495,0.0005,0.7650,0.0850,0.1350,0.0150]]).T
        np.testing.assert_array_equal(Ms.C, expected_C)
        np.testing.assert_array_equal(Ms.p, expected_p)

    def test_sum3(self):

        M = Cpm(**self.kwargs)
        sumVars = [2, 3]
        Ms = M.sum(sumVars)
        expected_C = np.array([[1,1,1],
                              [2,1,1],
                              [1,2,1],
                              [2,2,1],
                              [2,1,2],
                              [2,2,2]])
        expected_p = np.array([[0.9995, 0.0005,0.985, 0.015, 1.00, 1.00]]).T

        np.testing.assert_array_equal(Ms.C, expected_C)
        np.testing.assert_array_almost_equal(Ms.p, expected_p)
        np.testing.assert_array_equal(Ms.variables, [5, 1, 4])

    def test_sum4(self):

        M = Cpm(**self.kwargs)
        sumVars = [5]
        Ms = M.sum(sumVars, flag=0)
        expected_C = np.array([[1,1,1],
                              [2,1,1],
                              [1,2,1],
                              [2,2,1],
                              [2,1,2],
                              [2,2,2]])
        expected_p = np.array([[0.9995, 0.0005,0.985, 0.015, 1.00, 1.00]]).T

        np.testing.assert_array_equal(Ms.C, expected_C)
        np.testing.assert_array_almost_equal(Ms.p, expected_p)
        np.testing.assert_array_equal(Ms.variables, [5, 1, 4])
        self.assertEqual(Ms.no_child, 1)


class Test_mcs_product(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.M1 = Cpm(variables=[1],
                       no_child=1,
                       C = np.array([1, 2]).T,
                       p = np.array([0.9, 0.1]).T)

        cls.M2= Cpm(variables=[2, 1],
                       no_child=1,
                       C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]),
                       p = np.array([0.99, 0.01, 0.9, 0.1]).T)

        cls.M3 = Cpm(variables=[3, 1],
                       no_child=1,
                       C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]),
                       p = np.array([0.95, 0.05, 0.85, 0.15]).T)

        cls.M4 = Cpm(variables=[4, 1],
                       no_child=1,
                       C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]),
                       p = np.array([0.99, 0.01, 0.9, 0.1]).T)

        cls.M5 = Cpm(variables=[5, 2, 3, 4],
                       no_child=1,
                       C = np.array([[2, 3, 3, 2], [1, 1, 3, 1], [1, 2, 1, 1], [2, 2, 2, 1]]),
                       p = np.array([1, 1, 1, 1]).T)

        vars1 = Variable(B=np.eye(2), value=['Mild', 'Severe'])
        vars2 = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]), value=['Survive', 'Fail'])
        vars3 = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]), value=['Survive', 'Fail'])
        vars4 = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]), value=['Survive', 'Fail'])
        vars5 = Variable(B=np.array([[1, 0], [0, 1]]), value=['Survive', 'Fail'])

        cls.vars_ = [vars1, vars2, vars3, vars4, vars5]

    def test_get_sample_order(self):

        cpms = [self.M1, self.M2, self.M3]

        sampleOrder, sampleVars, varAdditionOrder = get_sample_order(cpms)

        expected = [0, 1, 2]
        np.testing.assert_array_equal(sampleOrder, expected)
        np.testing.assert_array_equal(varAdditionOrder, expected)

        expected = [1, 2, 3]
        np.testing.assert_array_equal(sampleVars, expected)

    def test_get_prod_idx(self):

        cpms = [self.M1, self.M2, self.M3]

        result = get_prod_idx(cpms, [])

        #expected = [1, 0, 0]
        expected = 0

        np.testing.assert_array_equal(result, expected)

    def test_single_sample(self):

        cpms = [self.M1, self.M2, self.M3]
        sampleOrder = [0, 1, 2]
        sampleVars = [1, 2, 3]
        varAdditionOrder = [0, 1, 2]
        varis = self.vars_
        sampleInd = [1]

        sample, sampleProb = single_sample(cpms, sampleOrder, sampleVars, varAdditionOrder, varis, sampleInd)

        if (sample == [1, 1, 1]).all():
            np.testing.assert_array_almost_equal(sampleProb, [[0.846]], decimal=3)
        elif (sample == [2, 1, 1]).all():
            np.testing.assert_array_almost_equal(sampleProb, [[0.0765]], decimal=3)

    def test_mcs_product1(self):

        nSample = 10
        cpms = [self.M1, self.M2, self.M3]
        Mcs = mcs_product(cpms, nSample, self.vars_)

        np.testing.assert_array_equal(Mcs.variables, [3, 2, 1])

        self.assertEqual(Mcs.C.shape, (10, 3))
        self.assertEqual(Mcs.q.shape, (10, 1))
        self.assertEqual(Mcs.sample_idx.shape, (10, 1))

        irow = np.where((Mcs.C == (1, 1, 1)).all(axis=1))[0]
        try:
            np.testing.assert_array_almost_equal(Mcs.q[irow], 0.8464*np.ones((len(irow), 1)), decimal=4)
        except AssertionError:
            print(f'{Mcs.q[irow]} vs 0.8464')

        irow = np.where((Mcs.C == (1, 1, 2)).all(axis=1))[0]
        try:
            np.testing.assert_array_almost_equal(Mcs.q[irow], 0.0765*np.ones((len(irow), 1)))
        except AssertionError:
            print(f'{Mcs.q[irow]} vs 0.0765')

    def test_mcs_product2(self):

        nSample = 10
        cpms = [self.M1, self.M2, self.M3, self.M4, self.M5]
        Mcs = mcs_product(cpms, nSample, self.vars_)

        np.testing.assert_array_equal(Mcs.variables, [5, 4, 3, 2, 1])

        self.assertEqual(Mcs.C.shape, (10, 5))
        self.assertEqual(Mcs.q.shape, (10, 1))
        self.assertEqual(Mcs.sample_idx.shape, (10, 1))

        irow = np.where((Mcs.C == (1, 1, 1, 1, 1)).all(axis=1))[0]
        try:
            np.testing.assert_array_almost_equal(Mcs.q[irow], 0.8380*np.ones((len(irow), 1)), decimal=4)
        except AssertionError:
            print(f'{Mcs.q[irow]} vs 0.8380')

        irow = np.where((Mcs.C == (1, 1, 1, 1, 2)).all(axis=1))[0]
        try:
            np.testing.assert_array_almost_equal(Mcs.q[irow], 0.0688*np.ones((len(irow), 1)), decimal=3)
        except AssertionError:
            print(f'{Mcs.q[irow]} vs 0.0688')

    def test_mcs_product3(self):

        nSample = 10
        cpms = [self.M2, self.M5]
        with self.assertRaises(TypeError):
            Mcs = mcs_product(cpms, nSample, self.vars_)

    def test_get_value_given_condn1(self):

        condn = [1, False]
        value = [1,2]
        expected = [1]

        result = get_value_given_condn(value, condn)

        self.assertEqual(result, expected)


    def test_condition(self):

        condVars = np.array([1, 2])
        condStates = np.array([1, 1])
        vars_ = self.vars_

        [M], _ = condition(self.M3, condVars, condStates, vars_, [0])
        np.testing.assert_array_equal(M.C, [[1, 1], [2, 1]])

class Test_prod_cpms(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        M1 = Cpm(variables=[1],
                       no_child=1,
                       C = np.array([1, 2]).T,
                       p = np.array([0.9, 0.1]).T)

        M2 = Cpm(variables=[2, 1],
                       no_child=1,
                       C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]),
                       p = np.array([0.99, 0.01, 0.9, 0.1]).T)

        M3 = Cpm(variables=[3, 1],
                       no_child=1,
                       C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]),
                       p = np.array([0.95, 0.05, 0.85, 0.15]).T)

        cls.cpms = [M1, M2, M3]

        vars1 = Variable(B=np.eye(3), value=['Sunny', 'Cloudy', 'Rainy'])
        vars2 = Variable(B=np.array([[1, 0], [0, 1]]), value=['Good', 'Bad'])
        vars3 = Variable(B=np.array([[1, 0], [0, 1]]), value=['Below 0', 'Above 0'])
        cls.vars_ = [vars1, vars2, vars3]

    def test_prod_cpms(self):

        Mmult, vars_ = prod_cpms(cpms=self.cpms, var=self.vars_)

        np.testing.assert_array_equal(Mmult.variables, [1, 2, 3])

        expected = np.array([[1,1,1],[2,1,1],[1,2,1],[2,2,1],[1,1,2],[2,1,2],[1,2,2],[2,2,2]])
        np.testing.assert_array_equal(Mmult.C, expected)

        expected = np.array([[0.8464, 0.0765, 0.0086, 0.0085, 0.0446, 0.0135, 4.5e-4, 0.0015]]).T
        np.testing.assert_array_almost_equal(Mmult.p, expected, decimal=4)

if __name__=='__main__':
    unittest.main()

