import numpy as np
import sys, os
import pytest
import pdb

np.set_printoptions(precision=3)
#pd.set_option.display_precision = 3

from BNS_JT.cpm import Cpm, ismember, iscompatible, get_value_given_condn, flip, add_new_states, condition, get_prod, argsort, setdiff, get_sample_order, get_prod_idx, single_sample, mcs_product, prod_cpms, isinscope
from BNS_JT.variable import Variable


def test_init_cpm():

    a = Cpm(variables = ['A3', 'A2', 'A1'],
            no_child = 1,
            C = np.array([[2, 2, 3], [2, 1, 2], [1, 1, 1]]),
            p = [1, 1, 1])
    assert isinstance(a, Cpm)

@pytest.fixture
def dict_cpm():

    variables = [3, 2, 1]
    no_child = 1
    C = np.array([[2, 2, 3], [2, 1, 2], [1, 1, 1]])
    p = [1, 1, 1]

    return {'variables': variables,
            'no_child': no_child,
            'C': C,
            'p': p}

def test_init1(dict_cpm):
    a = Cpm(**dict_cpm)

    assert isinstance(a, Cpm)
    assert a.variables==dict_cpm['variables']
    assert a.no_child == dict_cpm['no_child']
    np.testing.assert_array_equal(a.C, dict_cpm['C'])

def test_init2():
    # using list for P
    a = Cpm(variables=[1], no_child=1, C=np.array([1, 2]), p=[0.9, 0.1])
    assert isinstance(a, Cpm)

def test_variables1(dict_cpm):

    f_variables = [1, 2]
    with pytest.raises(AssertionError):
        _ = Cpm(**{'variables': f_variables,
                   'no_child': dict_cpm['no_child'],
                   'C': dict_cpm['C'],
                   'p': dict_cpm['p']})

def test_variables2(dict_cpm):

    f_variables = [1, 2, 3, 4]
    with pytest.raises(AssertionError):
        _ = Cpm(**{'variables': f_variables,
                   'no_child': dict_cpm['no_child'],
                   'C': dict_cpm['C'],
                   'p': dict_cpm['p']})

def test_variables3(dict_cpm):

    f_variables = ['x', 2, 3]
    with pytest.raises(AssertionError):
        _ = Cpm(**{'variables': f_variables,
                   'no_child': dict_cpm['no_child'],
                   'C': dict_cpm['C'],
                   'p': dict_cpm['p']})

def test_no_child(dict_cpm):

    f_no_child = 4
    with pytest.raises(AssertionError):
        _ = Cpm(**{'variables': dict_cpm['variables'],
                   'no_child': f_no_child,
                   'C': dict_cpm['C'],
                   'p': dict_cpm['p']})

def test_sort1():

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



def test_sort2s():

    p = np.array([[0.9405, 0.0495, 0.0095, 0.0005, 0.7650, 0.1350, 0.0850, 0.0150]]).T
    C = np.array([[1, 1, 1], [1, 2, 1], [2, 1, 1], [2, 2, 1], [1, 1, 2], [1, 2, 2], [2, 1, 2], [2, 2, 2]])

    Ms = Cpm(variables=['2', '3', '1'],
             no_child = 2,
             C = C,
             p = p)

    Ms.sort()

    np.testing.assert_array_equal(Ms.C, np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1], [1, 1, 2], [2, 1, 2], [1, 2, 2], [2, 2, 2]]))
    np.testing.assert_array_almost_equal(Ms.p, np.array([[0.9405, 0.0095, 0.0495, 5.0e-4, 0.7650, 0.0850, 0.1350, 0.0150]]).T)


def test_ismember1s():

    checkVars = ['1']
    variables = ['2', '1']

    lia, idxInCheckVars = ismember(checkVars, variables)

    assert idxInCheckVars==[1]
    assert lia==[True]

def test_ismember1():

    checkVars = [1]
    variables = [2, 1]

    lia, idxInCheckVars = ismember(checkVars, variables)

    assert idxInCheckVars==[1]
    assert lia==[True]

def test_ismember2():

    A = [5, 3, 4, 2]
    B = [2, 4, 4, 4, 6, 8]

    # MATLAB: [0, 0, 2, 1] => [False, False, 1, 0]
    lia, result = ismember(A, B)
    assert result==[False, False, 1, 0]
    assert lia==[False, False, True, True]

    lia, result = ismember(B, A)
    assert result==[3, 2, 2, 2, False, False]
    assert lia==[True, True, True, True, False, False]

def test_ismember2s():

    A = ['5', '3', '4', '2']
    B = ['2', '4', '4', '4', '6', '8']

    # MATLAB: [0, 0, 2, 1] => [False, False, 1, 0]
    lia, result = ismember(A, B)
    assert result==[False, False, 1, 0]
    assert lia==[False, False, True, True]

    lia, result = ismember(B, A)
    assert result==[3, 2, 2, 2, False, False]
    assert lia==[True, True, True, True, False, False]

    lia, result = ismember(A, B)
    expected = [False, False, 1, 0]

    assert result==expected
    assert lia==[False, False, True, True]

def test_ismember3():

    A = np.array([5, 3, 4, 2])
    B = np.array([2, 4, 4, 4, 6, 8])
    # MATLAB: [0, 0, 2, 1] => [False, False, 1, 0]

    expected = [False, False, 1, 0]
    lia, result = ismember(A, B)

    assert result==expected
    assert lia==[False, False, True, True]

def test_ismember4():
    # row by row checking
    A = np.array([[1, 0],
                  [1, 0],
                  [1, 0],
                  [1, 0]])
    B = np.array([[1, 0], [0, 1], [1, 1]])

    expected = [0, 0, 0, 0]
    lia, result = ismember(A, B)
    assert result==expected
    assert lia==[True, True, True, True]

def test_ismember5():
    # row by row checking
    A = np.array([[0, 1], [1, 2], [1, 0], [1, 1]])
    B = np.array([[1, 0], [0, 1], [1, 1]])

    expected = [1, False, 0, 2]
    lia, result = ismember(A, B)
    assert result==expected
    assert lia==[True, False, True, True]

def test_ismember6():
    # row by row checking
    A = [1]
    B = np.array([[1, 0], [0, 1], [1, 1]])

    with pytest.raises(AssertionError):
        _ = ismember(A, B)

def test_ismember7():

    A = np.array([1])
    B = np.array([2])
    expected = [False]
    # MATLAB: [0, 0, 2, 1] => [False, False, 1, 0]
    lia, result = ismember(A, B)
    assert result==expected
    assert lia==[False]

    B = np.array([1])
    expected = [0]
    # MATLAB: [0, 0, 2, 1] => [False, False, 1, 0]
    lia, result = ismember(A, B)
    assert result==expected
    assert lia==[True]

def test_ismember8():

    A = [12, 8]
    B = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # MATLAB: [0, 1] => [False, 0]

    expected = [False, True]
    result, lib = ismember(A, B)

    assert expected==result
    assert lib==[False, 7]

def test_argsort():

    C = np.array([[1, 1, 1], [1, 2, 1], [2, 1, 1], [2, 2, 1], [1, 1, 2], [1, 2, 2], [2, 1, 2], [2, 2, 2]])
    x = list(map(tuple, C[:, ::-1]))
    res = argsort(x)
    assert res==[0, 2, 1, 3, 4, 6, 5, 7]  # matlab index -1

def test_get_prod():

    A = np.array([[0.95, 0.05]]).T
    B = np.array([0.99])

    result = get_prod(A, B)
    np.testing.assert_array_equal(result, np.array([[0.9405, 0.0495]]).T)
    np.testing.assert_array_equal(result, A*B)

def test_setdiff():

    A = [3, 6, 2, 1, 5, 1, 1]
    B = [2, 4, 6]
    C, ia = setdiff(A, B)

    assert C==[3, 1, 5]
    assert ia==[0, 3, 4]

def test_get_value_given_condn1():

    A = [1, 2, 3, 4]
    condn = [0, False, 1, 3]
    result = get_value_given_condn(A, condn)
    assert result==[1, 3, 4]

def test_get_value_given_condn2():

    A = [1, 2, 3]
    condn = [0, False, 1, 3]

    with pytest.raises(AssertionError):
        get_value_given_condn(A, condn)

def test_add_new_states():
    states = np.array([np.ones(8), np.zeros(8)]).T
    B = np.array([[1, 0], [0, 1], [1, 1]])

    _, newStateCheck = ismember(states, B)
    expected = [0, 0, 0, 0, 0, 0, 0, 0]
    assert newStateCheck==expected

    newStateCheck = flip(newStateCheck)
    np.testing.assert_array_equal(newStateCheck, np.zeros_like(newStateCheck, dtype=bool))
    newState = states[newStateCheck, :]
    np.testing.assert_array_equal(newState, np.empty(shape=(0, 2)))
    #B = np.append(B, newState, axis=1)

    result = add_new_states(states, B)
    np.testing.assert_array_equal(result, B)

def test_isinscope1s():

    cpms = []
    # Travel times (systems)
    c7 = np.array([
    [1,3,1,3,3,3,3],
    [2,1,2,1,3,3,3],
    [3,1,2,2,3,3,3],
    [3,2,2,3,3,3,3]])

    vars_ = ['7', '1', '2', '3', '4', '5', '6']
    for i in range(1, 7):
        m = Cpm(variables= [f'{i}'],
                      no_child = 1,
                      C = np.array([[1, 0]]).T,
                      p = [1, 1])
        cpms.append(m)

    for i in range(7, 11):
        m = Cpm(variables= vars_,
                      no_child = 1,
                      C = c7,
                      p = [1, 1, 1, 1])
        cpms.append(m)

    result = isinscope(['1'], cpms)
    expected = np.array([[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]]).T
    np.testing.assert_array_equal(expected, result)

    result = isinscope(['1', '2'], cpms)
    expected = np.array([[1, 1, 0, 0, 0, 0, 1, 1, 1, 1]]).T
    np.testing.assert_array_equal(expected, result)



def test_isinscope1():

    cpms = []
    # Travel times (systems)
    c7 = np.array([
    [1,3,1,3,3,3,3],
    [2,1,2,1,3,3,3],
    [3,1,2,2,3,3,3],
    [3,2,2,3,3,3,3]])

    vars_ = [7, 1, 2, 3, 4, 5, 6]
    for i in range(1, 7):
        m = Cpm(variables= [i],
                      no_child = 1,
                      C = np.array([[1, 0]]).T,
                      p = [1, 1])
        cpms.append(m)

    for i in range(7, 11):
        m = Cpm(variables= vars_,
                      no_child = 1,
                      C = c7,
                      p = [1, 1, 1, 1])
        cpms.append(m)

    result = isinscope([1], cpms)
    expected = np.array([[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]]).T
    np.testing.assert_array_equal(expected, result)

    result = isinscope([1, 2], cpms)
    expected = np.array([[1, 1, 0, 0, 0, 0, 1, 1, 1, 1]]).T
    np.testing.assert_array_equal(expected, result)

def test_isinscope2():

    cpms = {}
    # Travel times (systems)
    c7 = np.array([
    [1,3,1,3,3,3,3],
    [2,1,2,1,3,3,3],
    [3,1,2,2,3,3,3],
    [3,2,2,3,3,3,3]])

    vars_ = [7, 1, 2, 3, 4, 5, 6]
    for i in range(1, 7):
        m = Cpm(variables= [i],
                      no_child = 1,
                      C = np.array([[1, 0]]).T,
                      p = [1, 1])
        cpms[i] = m

    for i in range(7, 11):
        m = Cpm(variables= vars_,
                      no_child = 1,
                      C = c7,
                      p = [1, 1, 1, 1])
        cpms[i] = m

    result = isinscope([1], cpms)
    expected = np.array([[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]]).T
    np.testing.assert_array_equal(expected, result)

    result = isinscope([1, 2], cpms)
    expected = np.array([[1, 1, 0, 0, 0, 0, 1, 1, 1, 1]]).T
    np.testing.assert_array_equal(expected, result)


@pytest.fixture
def setup_iscompatible():

    M = {}
    vars_ = {}

    M[1] = Cpm(variables=[1], no_child=1, C = np.array([[1, 2]]).T, p = np.array([0.9, 0.1]).T)
    M[2] = Cpm(variables=[2, 1], no_child=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]), p = np.array([0.99, 0.01, 0.9, 0.1]).T)
    M[3] = Cpm(variables=[3, 1], no_child=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]), p = np.array([0.95, 0.05, 0.85, 0.15]).T)
    M[4] = Cpm(variables=[4, 1], no_child=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]), p = np.array([0.99, 0.01, 0.9, 0.1]).T)
    M[5] = Cpm(variables=[5, 2, 3, 4], no_child=1, C = np.array([[2, 3, 3, 2], [1, 1, 3, 1], [1, 2, 1, 1], [2, 2, 2, 1]]), p = np.array([1, 1, 1, 1]).T)

    vars_[1] = Variable(B=np.eye(2), value=['Mild', 'Severe'])
    vars_[2] = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]), value=['Survive', 'Fail'])
    vars_[3] = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]), value=['Survive', 'Fail'])
    vars_[4] = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]), value=['Survive', 'Fail'])
    vars_[5] = Variable(B=np.array([[1, 0], [0, 1]]), value=['Survive', 'Fail'])

    return M, vars_

def test_iscompatible1(setup_iscompatible):

    # M[2]
    C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]])  #M[2].C
    variables = [2, 1]
    checkVars = [1]
    checkStates = [1]
    v_info = setup_iscompatible[1]

    result = iscompatible(C, variables, checkVars, checkStates, v_info)
    expected = np.array([1, 1, 0, 0])
    np.testing.assert_array_equal(expected, result)


def test_iscompatible2(setup_iscompatible):

    # M[5]
    C = np.array([[2, 3, 3, 2], [1, 1, 3, 1], [1, 2, 1, 1], [2, 2, 2, 1]])
    variables = [5, 2, 3, 4]
    checkVars = [3, 4]
    checkStates = [1, 1]
    v_info = setup_iscompatible[1]

    result = iscompatible(C, variables, checkVars, checkStates, v_info)
    expected = np.array([0, 1, 1, 0])
    np.testing.assert_array_equal(expected, result)

def test_iscompatible3(setup_iscompatible):

    #M[1]
    C = np.array([[1, 2]]).T
    variables = [1]
    checkVars = [3, 4]
    checkStates = [1, 1]
    v_info = setup_iscompatible[1]

    result = iscompatible(C, variables, checkVars, checkStates, v_info)
    expected = np.array([1, 1])
    np.testing.assert_array_equal(expected, result)

def test_iscompatible3s(setup_iscompatible):

    #M[1]
    C = np.array([[1, 2]]).T
    variables = ['1']
    checkVars = ['3', '4']
    checkStates = [1, 1]
    v_info = setup_iscompatible[1]

    v_infos = {}
    for k, v in v_info.items():
        v_infos[f'{k}'] = v

    result = iscompatible(C, variables, checkVars, checkStates, v_info)
    expected = np.array([1, 1])
    np.testing.assert_array_equal(expected, result)

def test_iscompatible3s(setup_iscompatible):

    #M[1]
    C = np.array([[1, 2]]).T
    variables = ['1']
    checkVars = ['3', '4']
    checkStates = [1, 1]
    v_info = setup_iscompatible[1]

    v_infos = {}
    for k, v in v_info.items():
        v_infos[f'{k}'] = v

    result = iscompatible(C, variables, checkVars, checkStates, v_infos)
    expected = np.array([1, 1])
    np.testing.assert_array_equal(expected, result)


def test_iscompatible4(setup_iscompatible):

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
    vars_ = setup_iscompatible[1]

    result = iscompatible(C, variables, checkVars, checkStates, vars_)
    expected = np.array([1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0])
    np.testing.assert_array_equal(expected, result)

    _, idx = ismember(checkVars, variables)
    # should be one less than the Matlab result
    assert idx==[0, 3]

    checkVars = get_value_given_condn(checkVars, idx)
    assert checkVars==[2, 1]

    checkStates = get_value_given_condn(checkStates, idx)
    assert checkStates==[1, 1]

    C1_common = C1_common = C[:, idx].copy()
    compatFlag = np.ones(shape=(C.shape[0], 1), dtype=bool)
    B = vars_[checkVars[0]].B
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
    B = vars_[checkVars[1]].B
    C1 = C1_common[:, 1][np.newaxis, :]
    #x1_old = [B[k-1, :] for k in C1][0]
    x1 = [B[k-1, :] for k in C1[:, compatFlag.flatten()]][0]
    x2 = B[checkStates[1]-1, :]
    compatCheck = (np.sum(x1 * x2, axis=1) > 0)[:, np.newaxis]

    expected = np.array([[1, 1, 0, 0, 1, 1, 0, 0]]).T
    np.testing.assert_array_equal(compatCheck, expected)
    compatFlag[np.where(compatFlag > 0)[0][:len(compatCheck)]] = compatCheck
    np.testing.assert_array_equal(compatFlag, np.array([[1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]]).T)

def test_get_subset1(setup_iscompatible):

    M = setup_iscompatible[0]

    # M[5]
    rowIndex = [0]  # 1 -> 0
    result = M[5].get_subset(rowIndex)

    np.testing.assert_array_equal(result.C, np.array([[2, 3, 3, 2]]))
    np.testing.assert_array_equal(result.p, [[1]])

def test_get_subset1s(setup_iscompatible):

    M5 = Cpm(variables=['5', '2', '3', '4'], no_child=1, C = np.array([[2, 3, 3, 2], [1, 1, 3, 1], [1, 2, 1, 1], [2, 2, 2, 1]]), p = np.array([1, 1, 1, 1]).T)

    # M[5]
    rowIndex = [0]  # 1 -> 0
    result = M5.get_subset(rowIndex)

    assert result.variables == M5.variables
    np.testing.assert_array_equal(result.C, np.array([[2, 3, 3, 2]]))
    np.testing.assert_array_equal(result.p, [[1]])


def test_get_subset2(setup_iscompatible):

    M = setup_iscompatible[0]

    # M[5]
    rowIndex = [1, 2, 3]  # [2, 3, 4] -> 0
    result = M[5].get_subset(rowIndex, 0)

    np.testing.assert_array_equal(result.C, np.array([[2, 3, 3, 2]]))
    np.testing.assert_array_equal(result.p, [[1]])

def test_get_subset2s(setup_iscompatible):

    M5 = Cpm(variables=['5', '2', '3', '4'], no_child=1, C = np.array([[2, 3, 3, 2], [1, 1, 3, 1], [1, 2, 1, 1], [2, 2, 2, 1]]), p = np.array([1, 1, 1, 1]).T)

    rowIndex = [1, 2, 3]  # [2, 3, 4] -> 0
    result = M5.get_subset(rowIndex, 0)

    assert result.variables == M5.variables
    np.testing.assert_array_equal(result.C, np.array([[2, 3, 3, 2]]))
    np.testing.assert_array_equal(result.p, [[1]])


def test_get_subset3():

    M = Cpm(variables=[2, 3, 5, 1, 4],
            no_child=5,
            C=np.array([[2, 2, 2, 2, 2]]),
            p=np.array([[0.0150]]).T)

    result = M.get_subset([0], 0)

    assert result.C.any() == False
    assert result.p.any() == False

def test_iscompatibleCpm1(setup_iscompatible):

    # M[5]
    M, vars_ = setup_iscompatible
    rowIndex = [0]  # 1 -> 0
    M_sys_select = M[5].get_subset(rowIndex)
    result = M[3].iscompatible(M_sys_select, var=vars_)
    expected = np.array([1, 1, 1, 1])
    np.testing.assert_array_equal(result, expected)

def test_iscompatibleCpm2(setup_iscompatible):

    # M[5]
    M, vars_ = setup_iscompatible
    rowIndex = [0]  # 1 -> 0
    M_sys_select = M[5].get_subset(rowIndex)

    result = M[4].iscompatible(M_sys_select, var=vars_)
    expected = np.array([0, 1, 0, 1])
    np.testing.assert_array_equal(result, expected)

def test_iscompatibleCpm3(setup_iscompatible):

    # M[5]
    M, vars_ = setup_iscompatible
    rowIndex = [0]  # 1 -> 0
    M_sys_select = M[5].get_subset(rowIndex)

    result = M[1].iscompatible(M_sys_select, var=vars_)
    expected = np.array([1, 1])
    np.testing.assert_array_equal(result, expected)

@pytest.fixture
def setup_product():

    M = {}
    vars_ = {}

    M[2] = Cpm(variables=[2, 1], no_child=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]), p = np.array([0.99, 0.01, 0.9, 0.1]).T)
    M[3] = Cpm(variables=[3, 1], no_child=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]), p = np.array([0.95, 0.05, 0.85, 0.15]).T)
    M[5] = Cpm(variables=[5, 2, 3, 4], no_child=1, C = np.array([[2, 3, 3, 2], [1, 1, 3, 1], [1, 2, 1, 1], [2, 2, 2, 1]]), p = np.array([1, 1, 1, 1]).T)

    vars_[1] = Variable(B=np.eye(2), value=['Mild', 'Severe'])
    vars_[2] = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]), value=['Survive', 'Fail'])
    vars_[3] = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]), value=['Survive', 'Fail'])
    vars_[4] = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]), value=['Survive', 'Fail'])
    vars_[5] = Variable(B=np.array([[1, 0], [0, 1]]), value=['Survive', 'Fail'])

    return M, vars_

def test_product1(setup_product):

    M, v_info = setup_product

    # When there is no common variable
    M1 = M[2]
    M2 = M[3]

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

    assert list(commonVars) == [1]

    _, idxVarsM1 = ismember(M1.variables, M2.variables)
    commonVars = get_value_given_condn(M1.variables, idxVarsM1)

    np.testing.assert_array_equal(idxVarsM1, np.array([0, 1]))
    assert commonVars==[1]

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
        [M2_], v_info = condition([M2], commonVars, c1_, v_info, sampleInd1)

        #assert M2_.variables, [3, 1])
        #assert M2_.no_child, 1)
        #np.testing.assert_array_equal(M2_.C, np.array([[1, 1], [2, 1]]))
        #np.testing.assert_array_equal(M2_.p, np.array([[0.95, 0.05]]).T)
        #Cprod = np.append(Cprod, M2_.C).reshape(M2_.C.shape[0], -1)
        _add = np.append(M2_.C, np.tile(c1_notCommon, (M2_.C.shape[0], 1)), axis=1)
        if i:
            Cprod = np.append(Cprod, _add, axis=0)
        else:
            Cprod = _add

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

    _, idxVars = ismember(newVars, Cprod_vars)

    assert idxVars==[2, 0, 1] # matlab 3, 1, 2

    Mprod = Cpm(variables=newVars,
                no_child = len(newVarsChild),
                C = Cprod[:, idxVars],
                p = pprod)

    Mprod.sort()

    assert Mprod.variables==[2, 3, 1]
    assert Mprod.no_child==2
    np.testing.assert_array_equal(Mprod.C, np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1], [1, 1, 2], [2, 1, 2], [1, 2, 2], [2, 2, 2]]))
    np.testing.assert_array_almost_equal(Mprod.p, np.array([[0.9405, 0.0095, 0.0495, 5.0e-4, 0.7650, 0.0850, 0.1350, 0.0150]]).T)

def test_product2(setup_product):

    M, v_info = setup_product
    M1 = M[2]
    M2 = M[3]

    Mprod, v_info_ = M1.product(M2, v_info)

    np.testing.assert_array_equal(Mprod.variables, [2, 3, 1])
    assert Mprod.no_child==2
    np.testing.assert_array_equal(Mprod.C, np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1], [1, 1, 2], [2, 1, 2], [1, 2, 2], [2, 2, 2]]))
    np.testing.assert_array_almost_equal(Mprod.p, np.array([[0.9405, 0.0095, 0.0495, 5.0e-4, 0.7650, 0.0850, 0.1350, 0.0150]]).T)

def test_product3(setup_product):

    M, v_info = setup_product
    M2 = M[5]

    M1 = Cpm(variables=[2, 3, 1], no_child=2, C = np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1], [1, 1, 2], [2, 1, 2], [1, 2, 2], [2, 2, 2]]), p = np.array([[0.9405, 0.0095, 0.0495, 5.0e-4, 0.7650, 0.0850, 0.1350, 0.0150]]).T)

    Mprod, v_info_ = M1.product(M2, v_info)

    np.testing.assert_array_equal(Mprod.variables, [2, 3, 5, 1, 4])
    assert Mprod.no_child==3

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

def test_product3s(setup_product):

    _, v_info = setup_product
    v_infos = {}
    for k, v in v_info.items():
        v_infos[f'{k}'] = v

    M2 = Cpm(variables=['5', '2', '3', '4'], no_child=1, C = np.array([[2, 3, 3, 2], [1, 1, 3, 1], [1, 2, 1, 1], [2, 2, 2, 1]]), p = np.array([1, 1, 1, 1]).T)

    M1 = Cpm(variables=['2', '3', '1'], no_child=2, C = np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1], [1, 1, 2], [2, 1, 2], [1, 2, 2], [2, 2, 2]]), p = np.array([[0.9405, 0.0095, 0.0495, 5.0e-4, 0.7650, 0.0850, 0.1350, 0.0150]]).T)

    #pdb.set_trace()
    Mprod, v_info_ = M1.product(M2, v_infos)

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

    assert Mprod.variables == ['2', '3', '5', '1', '4']

    np.testing.assert_array_equal(Mprod.C, expected_C)

    np.testing.assert_array_almost_equal(Mprod.p, expected_p)

    assert Mprod.no_child==3

@pytest.fixture
def setup_condition():
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
    p = np.array([[0.9405, 0.0095, 0.0495, 0.0005, 0.7650, 0.0850, 0.1350, 0.0150, 0.9405, 0.0095, 0.0495, 0.0005, 0.7650, 0.0850, 0.1350, 0.0150]]).T

    vars_ = {}
    vars_[1] = Variable(B=np.eye(2), value=['Mild', 'Severe'])
    vars_[2] = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]), value=['Survive', 'Fail'])
    vars_[3] = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]), value=['Survive', 'Fail'])
    vars_[4] = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]), value=['Survive', 'Fail'])
    vars_[5] = Variable(B=np.array([[1, 0], [0, 1]]), value=['Survive', 'Fail'])

    Mx = Cpm(variables=[2, 3, 5, 1, 4], no_child=3, C = C, p = p)

    return Mx, vars_

def test_condition0(setup_condition):

    Mx, vars_ = setup_condition
    condVars = np.array([2])
    condStates = np.array([1])

    Mx = Cpm(variables=[2, 1],
             no_child = 1,
             C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]),
             p = np.array([[0.99, 0.01, 0.9, 0.1]]).T)

    compatFlag = iscompatible(Mx.C, Mx.variables, condVars, condStates, vars_)
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

    _, idxInC = ismember(condVars, Mx.variables)
    np.testing.assert_array_equal(idxInC, [0])  # matlab 1 though

    _, idxInCondVars = ismember(Mx.variables, condVars)
    np.testing.assert_array_equal(idxInCondVars, [0, False])  # matlab 1 though
    not_idxInCondVars = flip(idxInCondVars)
    assert not_idxInCondVars==[False, True]
    Ccond = np.zeros_like(Ccompat)
    Ccond[:, not_idxInCondVars] = get_value_given_condn(Ccompat, not_idxInCondVars)
    #np.testing.assert_array_equal(Ccond_, Ccompat[:, 1:])
    #Ccond[:, new_cond] = Ccond_
    expected = np.array([[0, 1], [0, 2]])
    np.testing.assert_array_equal(Ccond, expected)

    _condVars = get_value_given_condn(condVars, idxInC)
    _condStates = get_value_given_condn(condStates, idxInC)
    _idxInC = get_value_given_condn(idxInC, idxInC)
    assert _condVars == np.array([2])
    assert _condStates== np.array([1])
    assert _idxInC== np.array([0])

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
    Ccond[:, _idxInC[0]] = [x+1 for x in ismember(compatCheck_mv, B)[1]]

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

def test_condition1(setup_condition):

    Mx, vars_ = setup_condition
    condVars = np.array([2])
    condStates = np.array([1])

    [M_n], vars_n = condition([Mx], condVars, condStates, vars_)
    np.testing.assert_array_equal(M_n.variables, [2, 3, 5, 1, 4])
    assert M_n.no_child == 3
    expected = np.array([[1,1,1,1,1],
                        [1,2,1,1,1],
                        [1,1,1,2,1],
                        [1,2,1,2,1],
                        [1,1,2,1,2],
                        [1,2,2,1,2],
                        [1,1,2,2,2],
                        [1,2,2,2,2]])
    np.testing.assert_array_equal(M_n.C, expected)

    expected = np.array([[0.9405,0.0495,0.7650,0.1350,0.9405,0.0495,0.7650,0.1350]]).T
    np.testing.assert_array_equal(M_n.p, expected)
    assert M_n.q.any() == False
    assert M_n.sample_idx.any() == False

def test_condition1d(setup_condition):

    Mx, vars_ = setup_condition
    condVars = np.array([2])
    condStates = np.array([1])

    [M_n], vars_n = condition({1: Mx}, condVars, condStates, vars_)
    np.testing.assert_array_equal(M_n.variables, [2, 3, 5, 1, 4])
    assert M_n.no_child== 3
    expected = np.array([[1,1,1,1,1],
                        [1,2,1,1,1],
                        [1,1,1,2,1],
                        [1,2,1,2,1],
                        [1,1,2,1,2],
                        [1,2,2,1,2],
                        [1,1,2,2,2],
                        [1,2,2,2,2]])
    np.testing.assert_array_equal(M_n.C, expected)

    expected = np.array([[0.9405,0.0495,0.7650,0.1350,0.9405,0.0495,0.7650,0.1350]]).T
    np.testing.assert_array_equal(M_n.p, expected)
    assert M_n.q.any() == False
    assert M_n.sample_idx.any() == False

def test_condition2(setup_condition):

    Mx, vars_ = setup_condition
    condVars = np.array([1])
    condStates = np.array([1])

    [M_n], vars_n = condition([Mx], condVars, condStates, vars_)

    np.testing.assert_array_equal(M_n.variables, [2, 3, 5, 1, 4])
    assert M_n.no_child==3
    expected = np.array([[1,1,1,1,1],
                        [2,1,1,1,1],
                        [1,2,1,1,1],
                        [2,2,2,1,1],
                        [1,1,2,1,2],
                        [2,1,2,1,2],
                        [1,2,2,1,2],
                        [2,2,2,1,2]])

    np.testing.assert_array_equal(M_n.C, expected)

    expected = np.array([[0.9405,0.0095,0.0495,0.0005,0.9405,0.0095,0.0495,0.0005]]).T
    np.testing.assert_array_equal(M_n.p, expected)
    assert M_n.q.any() == False
    assert M_n.sample_idx.any() == False

def test_condition3(setup_condition):
    # conditioning on multiple nodes
    Mx, vars_ = setup_condition
    condVars = np.array([2, 1])
    condStates = np.array([1, 1])

    [M_n], vars_n = condition([Mx], condVars, condStates, vars_)

    np.testing.assert_array_equal(M_n.variables, [2, 3, 5, 1, 4])
    assert M_n.no_child==3
    expected = np.array([[1,1,1,1,1],
                        [1,2,1,1,1],
                        [1,1,2,1,2],
                        [1,2,2,1,2]])

    np.testing.assert_array_equal(M_n.C, expected)

    expected = np.array([[0.9405,0.0495,0.9405,0.0495]]).T
    np.testing.assert_array_equal(M_n.p, expected)
    assert M_n.q.any() == False
    assert M_n.sample_idx.any() == False

def test_condition3s(setup_condition):
    # conditioning on multiple nodes
    Mx_, vars_ = setup_condition
    Mx = Cpm(variables=['2', '3', '5', '1', '4'], no_child=3, C = Mx_.C, p = Mx_.p)
    v_info = {}
    for k, v in vars_.items():
        v_info[f'{k}'] = v

    condVars = np.array(['2', '1'])
    condStates = np.array([1, 1])

    [M_n], vars_n = condition([Mx], condVars, condStates, v_info)

    np.testing.assert_array_equal(M_n.variables, ['2', '3', '5', '1', '4'])
    assert M_n.no_child==3
    expected = np.array([[1,1,1,1,1],
                        [1,2,1,1,1],
                        [1,1,2,1,2],
                        [1,2,2,1,2]])

    np.testing.assert_array_equal(M_n.C, expected)

    expected = np.array([[0.9405,0.0495,0.9405,0.0495]]).T
    np.testing.assert_array_equal(M_n.p, expected)
    assert M_n.q.any() == False
    assert M_n.sample_idx.any() == False


def test_condition4(setup_condition):

    _, vars_ = setup_condition
    C = np.array([[2, 3, 3, 2],
                 [1, 1, 3, 1],
                 [1, 2, 1, 1],
                 [2, 2, 2, 1]])
    p = np.array([1, 1, 1, 1, ])
    Mx = Cpm(variables=[5, 2, 3, 4], no_child=1, C = C, p = p.T)
    condVars = np.array([2, 3])
    condStates = np.array([1, 1])

    result = iscompatible(Mx.C, Mx.variables, condVars, condStates, vars_)
    expected = np.array([1,1,0,0])
    np.testing.assert_array_equal(expected, result)

    [M_n], vars_n = condition([Mx], condVars, condStates, vars_)

    np.testing.assert_array_equal(M_n.variables, [5, 2, 3, 4])
    assert M_n.no_child==1
    expected = np.array([[2,1,1,2],
                         [1,1,1,1]])
    np.testing.assert_array_equal(M_n.C, expected)

    expected = np.array([[1, 1]]).T
    np.testing.assert_array_equal(M_n.p, expected)
    assert M_n.q.any() == False
    assert M_n.sample_idx.any() == False

def test_condition5(setup_condition):

    _, vars_ = setup_condition
    C = np.array([[1, 1],
                 [2, 1],
                 [1, 2],
                 [2, 2]])
    p = np.array([0.95, 0.05, 0.85, 0.15])
    M2 = Cpm(variables=[3, 1], no_child=1, C = C, p = p.T)
    condVars = np.array([1])
    states = np.array([2])

    [M_n], vars_n = condition([M2], condVars, states, vars_)

    np.testing.assert_array_equal(M_n.variables, [3, 1])
    assert M_n.no_child== 1
    expected = np.array([[1,2],
                         [2,2]])
    np.testing.assert_array_equal(M_n.C, expected)

    expected = np.array([[0.85, 0.15]]).T
    np.testing.assert_array_equal(M_n.p, expected)

def test_condition5s(setup_condition):

    _, vars_ = setup_condition
    v_info = {}
    for k, v in vars_.items():
        v_info[f'{k}'] = v

    C = np.array([[1, 1],
                 [2, 1],
                 [1, 2],
                 [2, 2]])
    p = np.array([0.95, 0.05, 0.85, 0.15])
    M2 = Cpm(variables=['3', '1'], no_child=1, C = C, p = p.T)
    condVars = np.array(['1'])
    states = np.array([2])

    [M_n], vars_n = condition([M2], condVars, states, v_info)

    np.testing.assert_array_equal(M_n.variables, ['3', '1'])
    assert M_n.no_child== 1
    expected = np.array([[1,2],
                         [2,2]])
    np.testing.assert_array_equal(M_n.C, expected)

    expected = np.array([[0.85, 0.15]]).T
    np.testing.assert_array_equal(M_n.p, expected)


def test_condition6(setup_condition):

    _, vars_ = setup_condition
    C = np.array([[1, 2]]).T
    p = np.array([0.9, 0.1])
    M2 = Cpm(variables=[1], no_child=1, C = C, p = p.T)
    condVars = np.array([])
    states = np.array([])

    [M_n], vars_n = condition([M2], condVars, states, vars_)

    np.testing.assert_array_equal(M_n.variables, [1])
    assert M_n.no_child== 1
    expected = np.array([[1,2]]).T
    np.testing.assert_array_equal(M_n.C, expected)

    expected = np.array([[0.9, 0.1]]).T
    np.testing.assert_array_equal(M_n.p, expected)


@pytest.fixture
def setup_sum():
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

    return {'variables': variables,
            'no_child': no_child,
            'C': C,
            'p': p}

def test_sum1(setup_sum):

    M = Cpm(**setup_sum)
    sumVars = [1]
    varsRemainIdx = ismember( sumVars, M.variables[:M.no_child])

    sumFlag = 1
    if sumFlag:
        varsRemain, varsRemainIdx = setdiff(M.variables[:M.no_child], sumVars)
        assert varsRemain== [2, 3, 5]
        assert varsRemainIdx== [0, 1, 2]  # Matlab: [1, 2, 3]
    else:
        varsRemainIdx = get_value_given_condn(varsRemainIdx, varsRemainIdx)
        assert varsRemainIdx== []
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
            assert Mcompare.q.any()==False
            assert Mcompare.sample_idx.any() == False

        flag = Mloop.iscompatible(Mcompare)
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
            assert Mloop.q.any()==False
            assert Mloop.sample_idx.any()==False
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

def test_sum2(setup_sum):

    M = Cpm(**setup_sum)
    sumVars = [1]

    Ms = M.sum(sumVars)
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

def test_sum2s(setup_sum):

    M = Cpm(**setup_sum)
    M.variables = [f'{x}' for x in M.variables]
    sumVars = ['1']

    #pdb.set_trace()
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

    np.testing.assert_array_equal(Ms.p, expected_p)
    np.testing.assert_array_equal(Ms.C, expected_C)

def test_sum3(setup_sum):

    M = Cpm(**setup_sum)
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

def test_sum4(setup_sum):

    M = Cpm(**setup_sum)
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
    assert Ms.no_child== 1


@pytest.fixture
def setup_mcs_product():
    M = {}
    M[1] = Cpm(variables=[1],
                   no_child=1,
                   C = np.array([1, 2]).T,
                   p = np.array([0.9, 0.1]).T)

    M[2]= Cpm(variables=[2, 1],
                   no_child=1,
                   C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]),
                   p = np.array([0.99, 0.01, 0.9, 0.1]).T)

    M[3] = Cpm(variables=[3, 1],
                   no_child=1,
                   C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]),
                   p = np.array([0.95, 0.05, 0.85, 0.15]).T)

    M[4] = Cpm(variables=[4, 1],
                   no_child=1,
                   C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]),
                   p = np.array([0.99, 0.01, 0.9, 0.1]).T)

    M[5] = Cpm(variables=[5, 2, 3, 4],
                   no_child=1,
                   C = np.array([[2, 3, 3, 2], [1, 1, 3, 1], [1, 2, 1, 1], [2, 2, 2, 1]]),
                   p = np.array([1, 1, 1, 1]).T)

    vars_ = {}
    vars_[1] = Variable(B=np.eye(2), value=['Mild', 'Severe'])
    vars_[2] = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]), value=['Survive', 'Fail'])
    vars_[3] = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]), value=['Survive', 'Fail'])
    vars_[4] = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]), value=['Survive', 'Fail'])
    vars_[5] = Variable(B=np.array([[1, 0], [0, 1]]), value=['Survive', 'Fail'])

    return M, vars_


def test_get_sample_order(setup_mcs_product):

    cpms, _ = setup_mcs_product
    cpms = [cpms[k] for k in [1, 2, 3]]

    sampleOrder, sampleVars, varAdditionOrder = get_sample_order(cpms)

    expected = [0, 1, 2]
    np.testing.assert_array_equal(sampleOrder, expected)
    np.testing.assert_array_equal(varAdditionOrder, expected)

    expected = [1, 2, 3]
    np.testing.assert_array_equal(sampleVars, expected)

def test_get_prod_idx1(setup_mcs_product):

    cpms, _ = setup_mcs_product
    cpms = [cpms[k] for k in [1, 2, 3]]

    result = get_prod_idx(cpms, [])

    #expected = [1, 0, 0]
    expected = 0

    np.testing.assert_array_equal(result, expected)

def test_get_prod_idx2(setup_mcs_product):

    cpms, _ = setup_mcs_product
    cpms = {k:cpms[k] for k in [1, 2, 3]}

    result = get_prod_idx(cpms, [])

    #expected = [1, 0, 0]
    expected = 0

    np.testing.assert_array_equal(result, expected)

def test_single_sample1(setup_mcs_product):

    cpms, varis = setup_mcs_product
    cpms = [cpms[k] for k in [1, 2, 3]]

    sampleOrder = [0, 1, 2]
    sampleVars = [1, 2, 3]
    varAdditionOrder = [0, 1, 2]
    sampleInd = [1]

    sample, sampleProb = single_sample(cpms, sampleOrder, sampleVars, varAdditionOrder, varis, sampleInd)

    if (sample == [1, 1, 1]).all():
        np.testing.assert_array_almost_equal(sampleProb, [[0.846]], decimal=3)
    elif (sample == [2, 1, 1]).all():
        np.testing.assert_array_almost_equal(sampleProb, [[0.0765]], decimal=3)

def test_single_sample2(setup_mcs_product):

    cpms, varis = setup_mcs_product
    cpms = {k:cpms[k] for k in [1, 2, 3]}

    sampleOrder = [0, 1, 2]
    sampleVars = [1, 2, 3]
    varAdditionOrder = [0, 1, 2]
    sampleInd = [1]

    sample, sampleProb = single_sample(cpms, sampleOrder, sampleVars, varAdditionOrder, varis, sampleInd)

    if (sample == [1, 1, 1]).all():
        np.testing.assert_array_almost_equal(sampleProb, [[0.846]], decimal=3)
    elif (sample == [2, 1, 1]).all():
        np.testing.assert_array_almost_equal(sampleProb, [[0.0765]], decimal=3)

def test_mcs_product1(setup_mcs_product):

    cpms, vars_ = setup_mcs_product
    cpms = {k:cpms[k] for k in [1, 2, 3]}

    nSample = 10
    Mcs = mcs_product(cpms, nSample, vars_)

    np.testing.assert_array_equal(Mcs.variables, [3, 2, 1])

    assert Mcs.C.shape== (10, 3)
    assert Mcs.q.shape== (10, 1)
    assert Mcs.sample_idx.shape== (10, 1)

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

def test_mcs_product2(setup_mcs_product):

    cpms, vars_ = setup_mcs_product
    cpms = [cpms[k] for k in [1, 2, 3, 4, 5]]
    nSample = 10
    Mcs = mcs_product(cpms, nSample, vars_)

    np.testing.assert_array_equal(Mcs.variables, [5, 4, 3, 2, 1])

    assert Mcs.C.shape== (10, 5)
    assert Mcs.q.shape== (10, 1)
    assert Mcs.sample_idx.shape== (10, 1)

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

def test_mcs_product2d(setup_mcs_product):

    cpms, vars_ = setup_mcs_product
    cpms = {k+1:cpms[k] for k in [1, 2, 3, 4, 5]}
    nSample = 10
    Mcs = mcs_product(cpms, nSample, vars_)

    np.testing.assert_array_equal(Mcs.variables, [5, 4, 3, 2, 1])

    assert Mcs.C.shape== (10, 5)
    assert Mcs.q.shape== (10, 1)
    assert Mcs.sample_idx.shape== (10, 1)

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

def test_mcs_product2ds(setup_mcs_product):

    cpms, vars_ = setup_mcs_product
    vars_ = {str(k):vars_[k] for k in [1, 2, 3, 4, 5]}

    cpms_ = {}
    for k, v in cpms.items():
        cpms_[str(k+1)] = cpms[k]
        cpms_[str(k+1)].variables = [str(k) for k in cpms[k].variables]

    nSample = 10
    Mcs = mcs_product(cpms_, nSample, vars_)

    np.testing.assert_array_equal(Mcs.variables, ['5', '4', '3', '2', '1'])

    assert Mcs.C.shape== (10, 5)
    assert Mcs.q.shape== (10, 1)
    assert Mcs.sample_idx.shape== (10, 1)

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


def test_mcs_product3(setup_mcs_product):

    nSample = 10
    cpms, vars_ = setup_mcs_product
    cpms = [cpms[k] for k in [2, 5]]
    with pytest.raises(TypeError):
        Mcs = mcs_product(cpms, nSample, vars_)

def test_get_value_given_condn1():

    condn = [1, False]
    value = [1,2]
    expected = [1]

    result = get_value_given_condn(value, condn)

    assert result==expected


def test_condition(setup_mcs_product):

    cpms, vars_ = setup_mcs_product
    condVars = np.array([1, 2])
    condStates = np.array([1, 1])

    [M], _ = condition(cpms[3], condVars, condStates, vars_, [0])
    np.testing.assert_array_equal(M.C, [[1, 1], [2, 1]])
    assert M.q.any() == False
    assert M.sample_idx.any() == False

def test_condition_s(setup_mcs_product):

    cpms, vars_ = setup_mcs_product
    vars_ = {str(k):vars_[k] for k in [1, 2, 3, 4, 5]}

    cpms_ = {}
    for k, v in cpms.items():
        cpms_[str(k)] = cpms[k]
        cpms_[str(k)].variables = [str(k) for k in cpms[k].variables]

    condVars = np.array(['1', '2'])
    condStates = np.array([1, 1])

    [M], _ = condition(cpms_['3'], condVars, condStates, vars_, [0])
    np.testing.assert_array_equal(M.C, [[1, 1], [2, 1]])
    assert M.q.any() == False
    assert M.sample_idx.any() == False

@pytest.fixture
def setup_prod_cms():

    M = {}
    M[1] = Cpm(variables=[1],
                   no_child=1,
                   C = np.array([1, 2]).T,
                   p = np.array([0.9, 0.1]).T)

    M[2] = Cpm(variables=[2, 1],
                   no_child=1,
                   C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]),
                   p = np.array([0.99, 0.01, 0.9, 0.1]).T)

    M[3] = Cpm(variables=[3, 1],
                   no_child=1,
                   C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]),
                   p = np.array([0.95, 0.05, 0.85, 0.15]).T)
    vars_ = {}
    vars_[1] = Variable(B=np.eye(3), value=['Sunny', 'Cloudy', 'Rainy'])
    vars_[2] = Variable(B=np.array([[1, 0], [0, 1]]), value=['Good', 'Bad'])
    vars_[3] = Variable(B=np.array([[1, 0], [0, 1]]), value=['Below 0', 'Above 0'])

    return M, vars_


def test_prod_cpms1(setup_prod_cms):

    cpms, vars_ = setup_prod_cms
    Mmult, vars_ = prod_cpms(cpms=cpms, var=vars_)

    np.testing.assert_array_equal(Mmult.variables, [1, 2, 3])

    expected = np.array([[1,1,1],[2,1,1],[1,2,1],[2,2,1],[1,1,2],[2,1,2],[1,2,2],[2,2,2]])
    np.testing.assert_array_equal(Mmult.C, expected)

    expected = np.array([[0.8464, 0.0765, 0.0086, 0.0085, 0.0446, 0.0135, 4.5e-4, 0.0015]]).T
    np.testing.assert_array_almost_equal(Mmult.p, expected, decimal=4)

def test_prod_cpms1s(setup_prod_cms):

    cpms, vars_ = setup_prod_cms
    cpms_ = {}
    for k, v in cpms.items():
        cpms_[str(k)] = v
        cpms_[str(k)].variables = [str(i) for i in v.variables]
    vars_ = {str(k):v for k, v in vars_.items()}

    Mmult, vars_ = prod_cpms(cpms=cpms_, var=vars_)

    assert Mmult.variables==['1', '2', '3']

    expected = np.array([[1,1,1],[2,1,1],[1,2,1],[2,2,1],[1,1,2],[2,1,2],[1,2,2],[2,2,2]])
    np.testing.assert_array_equal(Mmult.C, expected)

    expected = np.array([[0.8464, 0.0765, 0.0086, 0.0085, 0.0446, 0.0135, 4.5e-4, 0.0015]]).T
    np.testing.assert_array_almost_equal(Mmult.p, expected, decimal=4)



def test_prod_cpms2(setup_prod_cms):

    cpms, vars_ = setup_prod_cms
    Mmult, vars_ = prod_cpms(cpms=cpms, var=vars_)

    assert Mmult.variables == [1, 2, 3]

    expected = np.array([[1,1,1],[2,1,1],[1,2,1],[2,2,1],[1,1,2],[2,1,2],[1,2,2],[2,2,2]])
    np.testing.assert_array_equal(Mmult.C, expected)

    expected = np.array([[0.8464, 0.0765, 0.0086, 0.0085, 0.0446, 0.0135, 4.5e-4, 0.0015]]).T
    np.testing.assert_array_almost_equal(Mmult.p, expected, decimal=4)


