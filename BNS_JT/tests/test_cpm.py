import numpy as np
import sys, os
import pytest
import pdb
import copy

np.set_printoptions(precision=3)
#pd.set_option.display_precision = 3

from BNS_JT import cpm, variable


@pytest.fixture
def dict_cpm():
    ''' Use instance of Variables in the variables'''
    A1 = variable.Variable(**{'name': 'A1',
                              'B': [{0}, {1}, {0, 1}],
                              'values': ['s', 'f']})
    A2 = variable.Variable(**{'name': 'A2',
                              'B': [{0}, {1}, {0, 1}],
                              'values': ['s', 'f']})
    A3 = variable.Variable(**{'name': 'A3',
                              'B': [{0}, {1}, {0, 1}],
                              'values': ['s', 'f']})

    return {'variables': [A3, A2, A1],
            'no_child': 1,
            'C': np.array([[2, 2, 3], [2, 1, 2], [1, 1, 1]]) - 1,
            'p': [1, 1, 1]}


@pytest.fixture()
def var_A1_to_A5():

    A1 = variable.Variable(name='A1', B=[{0},{1}],values=['Mild', 'Severe'])
    A2 = variable.Variable(name='A2', B=[{0},{1},{0, 1}], values=['Survive', 'Fail'])
    A3 = variable.Variable(name='A3', B=[{0},{1},{0, 1}], values=['Survive', 'Fail'])
    A4 = variable.Variable(name='A4', B=[{0},{1},{0, 1}], values=['Survive', 'Fail'])
    A5 = variable.Variable(name='A5', B=[{0},{1}], values=['Survive', 'Fail'])

    return A1, A2, A3, A4, A5


def test_init(dict_cpm):

    a = cpm.Cpm(**dict_cpm)
    assert isinstance(a, cpm.Cpm)


def test_init1(dict_cpm):
    a = cpm.Cpm(**dict_cpm)

    assert isinstance(a, cpm.Cpm)
    assert a.variables==dict_cpm['variables']
    assert a.no_child == dict_cpm['no_child']
    np.testing.assert_array_equal(a.C, dict_cpm['C'])


def test_init2(dict_cpm):
    v = dict_cpm
    # using list for P
    a = cpm.Cpm(variables=[v['variables'][0]], no_child=1, C=np.array([1, 2]), p=[0.9, 0.1])
    assert isinstance(a, cpm.Cpm)


def test_init3(dict_cpm):
    v = dict_cpm
    # using list for P
    a = cpm.Cpm(variables=[v['variables'][0]], no_child=1, C=np.array([1, 2]))
    assert isinstance(a, cpm.Cpm)


def test_variables1(dict_cpm):

    f_variables = [1, 2]
    with pytest.raises(AssertionError):
        _ = cpm.Cpm(**{'variables': f_variables,
                   'no_child': dict_cpm['no_child'],
                   'C': dict_cpm['C'],
                   'p': dict_cpm['p']})

def test_variables2(dict_cpm):

    f_variables = [1, 2, 3, 4]
    with pytest.raises(AssertionError):
        _ = cpm.Cpm(**{'variables': f_variables,
                   'no_child': dict_cpm['no_child'],
                   'C': dict_cpm['C'],
                   'p': dict_cpm['p']})

def test_variables3(dict_cpm):

    f_variables = ['x', 2, 3]
    with pytest.raises(AssertionError):
        _ = cpm.Cpm(**{'variables': f_variables,
                   'no_child': dict_cpm['no_child'],
                   'C': dict_cpm['C'],
                   'p': dict_cpm['p']})

def test_no_child(dict_cpm):

    f_no_child = 4
    with pytest.raises(AssertionError):
        _ = cpm.Cpm(**{'variables': dict_cpm['variables'],
                   'no_child': f_no_child,
                   'C': dict_cpm['C'],
                   'p': dict_cpm['p']})

def test_sort1(dict_cpm):

    v = dict_cpm

    A3, A2, A1 = v['variables']

    p = np.array([[0.9405, 0.0495, 0.0095, 0.0005, 0.7650, 0.1350, 0.0850, 0.0150]]).T
    C = np.array([[1, 1, 1], [1, 2, 1], [2, 1, 1], [2, 2, 1], [1, 1, 2], [1, 2, 2], [2, 1, 2], [2, 2, 2]]) - 1

    M = cpm.Cpm(variables=[A2, A3, A1],
            no_child = 2,
            C = C,
            p = p)

    if any(M.sample_idx):
        rowIdx = cpm.argsort(M.sample_idx)
    else:
        rowIdx = cpm.argsort(list(map(tuple, C[:, ::-1])))

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

    Ms = cpm.Cpm(C=M.C[rowIdx, :],
             p=Ms_p,
             q=Ms_q,
             sample_idx=Ms_sample_idx,
             variables=M.variables,
             no_child=M.no_child)

    np.testing.assert_array_equal(Ms.C, np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1], [1, 1, 2], [2, 1, 2], [1, 2, 2], [2, 2, 2]]) - 1)
    np.testing.assert_array_almost_equal(Ms.p, np.array([[0.9405, 0.0095, 0.0495, 5.0e-4, 0.7650, 0.0850, 0.1350, 0.0150]]).T)


def test_ismember1s():

    checkVars = ['1']
    variables = ['2', '1']

    lia, idxInCheckVars = cpm.ismember(checkVars, variables)

    assert idxInCheckVars==[1]
    assert lia==[True]


def test_ismember1ss():

    A1 = variable.Variable(**{'name':'A1', 'B': [{0}, {1}, {0, 1}],
                   'values': ['s', 'f']})
    A2 = variable.Variable(**{'name': 'A2', 'B': [{0}, {1}, {0, 1}],
                   'values': ['s', 'f']})
    checkVars = [A1]
    variables = [A2, A1]

    lia, idxInCheckVars = cpm.ismember(checkVars, variables)

    assert idxInCheckVars==[1]
    assert lia==[True]


def test_ismember1ss1(var_A1_to_A5):

    A1, A2, A3, A4, A5 = var_A1_to_A5

    A = [A5, A2, A3, A4]
    B = [A2, A3]

    lia, res = cpm.ismember(A, B)

    assert res == [False, 0, 1, False]
    assert lia == [False, True, True, False]


def test_ismember1():

    checkVars = [1]
    variables = [2, 1]

    lia, idxInCheckVars = cpm.ismember(checkVars, variables)

    assert idxInCheckVars==[1]
    assert lia==[True]


def test_ismember2():

    A = [5, 3, 4, 2]
    B = [2, 4, 4, 4, 6, 8]

    # MATLAB: [0, 0, 2, 1] => [False, False, 1, 0]
    lia, result = cpm.ismember(A, B)
    assert result==[False, False, 1, 0]
    assert lia==[False, False, True, True]

    lia, result = cpm.ismember(B, A)
    assert result==[3, 2, 2, 2, False, False]
    assert lia==[True, True, True, True, False, False]

def test_ismember2s():

    A = ['5', '3', '4', '2']
    B = ['2', '4', '4', '4', '6', '8']

    # MATLAB: [0, 0, 2, 1] => [False, False, 1, 0]
    lia, result = cpm.ismember(A, B)
    assert result==[False, False, 1, 0]
    assert lia==[False, False, True, True]

    lia, result = cpm.ismember(B, A)
    assert result==[3, 2, 2, 2, False, False]
    assert lia==[True, True, True, True, False, False]

    lia, result = cpm.ismember(A, B)
    expected = [False, False, 1, 0]

    assert result==expected
    assert lia==[False, False, True, True]


def test_ismember2ss():

    A2 = variable.Variable(**{'name': 'A2', 'B': [{0},{1},{0, 1}], 'values': ['s', 'f']})
    A3 = variable.Variable(**{'name': 'A3', 'B': [{0},{1},{0, 1}], 'values': ['s', 'f']})
    A4 = variable.Variable(**{'name': 'A4', 'B': [{0},{1},{0, 1}], 'values': ['s', 'f']})
    A5 = variable.Variable(**{'name': 'A5', 'B': [{0},{1},{0, 1}], 'values': ['s', 'f']})
    A6 = variable.Variable(**{'name': 'A6', 'B': [{0},{1},{0, 1}], 'values': ['s', 'f']})
    A8 = variable.Variable(**{'name': 'A8', 'B': [{0},{1},{0, 1}], 'values': ['s', 'f']})

    A = [A5, A3, A4, A2]
    B = [A2, A4, A4, A4, A6, A8]

    # MATLAB: [0, 0, 2, 1] => [False, False, 1, 0]
    lia, result = cpm.ismember(A, B)
    assert result==[False, False, 1, 0]
    assert lia==[False, False, True, True]

    lia, result = cpm.ismember(B, A)
    assert result==[3, 2, 2, 2, False, False]
    assert lia==[True, True, True, True, False, False]

    lia, result = cpm.ismember(A, B)
    expected = [False, False, 1, 0]

    assert result==expected
    assert lia==[False, False, True, True]


def test_ismember3():

    A = np.array([5, 3, 4, 2])
    B = np.array([2, 4, 4, 4, 6, 8])
    # MATLAB: [0, 0, 2, 1] => [False, False, 1, 0]

    expected = [False, False, 1, 0]
    lia, result = cpm.ismember(A, B)

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
    lia, result = cpm.ismember(A, B)
    assert result==expected
    assert lia==[True, True, True, True]

def test_ismember4s():
    # row by row checking
    A = [{0}, {0}, {0}, {0}]
    B = [{0}, {1}, {0, 1}]

    expected = [0, 0, 0, 0]
    lia, result = cpm.ismember(A, B)
    assert result==expected
    assert lia==[True, True, True, True]

def test_ismember5():
    # row by row checking
    A = np.array([[0, 1], [1, 2], [1, 0], [1, 1]])
    B = np.array([[1, 0], [0, 1], [1, 1]])

    expected = [1, False, 0, 2]
    lia, result = cpm.ismember(A, B)
    assert result==expected
    assert lia==[True, False, True, True]

def test_ismember6():
    # row by row checking
    A = [1]
    B = np.array([[1, 0], [0, 1], [1, 1]])

    with pytest.raises(AssertionError):
        _ = cpm.ismember(A, B)

def test_ismember7():

    A = np.array([1])
    B = np.array([2])
    expected = [False]
    # MATLAB: [0, 0, 2, 1] => [False, False, 1, 0]
    lia, result = cpm.ismember(A, B)
    assert result==expected
    assert lia==[False]

    B = np.array([1])
    expected = [0]
    # MATLAB: [0, 0, 2, 1] => [False, False, 1, 0]
    lia, result = cpm.ismember(A, B)
    assert result==expected
    assert lia==[True]

def test_ismember8():

    A = [12, 8]
    B = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # MATLAB: [0, 1] => [False, 0]

    expected = [False, True]
    result, lib = cpm.ismember(A, B)

    assert expected==result
    assert lib==[False, 7]

def test_argsort():

    C = np.array([[1, 1, 1], [1, 2, 1], [2, 1, 1], [2, 2, 1], [1, 1, 2], [1, 2, 2], [2, 1, 2], [2, 2, 2]]) - 1
    x = list(map(tuple, C[:, ::-1]))
    res = cpm.argsort(x)
    assert res==[0, 2, 1, 3, 4, 6, 5, 7]  # matlab index -1

def test_get_prod():

    A = np.array([[0.95, 0.05]]).T
    B = np.array([0.99])

    result = cpm.get_prod(A, B)
    np.testing.assert_array_equal(result, np.array([[0.9405, 0.0495]]).T)
    np.testing.assert_array_equal(result, A*B)

def test_setdiff():

    A = [3, 6, 2, 1, 5, 1, 1]
    B = [2, 4, 6]
    C, ia = cpm.setdiff(A, B)

    assert C==[3, 1, 5]
    assert ia==[0, 3, 4]

def test_get_value_given_condn1():

    A = [1, 2, 3, 4]
    condn = [0, False, 1, 3]
    result = cpm.get_value_given_condn(A, condn)
    assert result==[1, 3, 4]

def test_get_value_given_condn2():

    A = [1, 2, 3]
    condn = [0, False, 1, 3]

    with pytest.raises(AssertionError):
        cpm.get_value_given_condn(A, condn)

def test_add_new_states():
    states = np.array([np.ones(8), np.zeros(8)]).T
    B = np.array([[1, 0], [0, 1], [1, 1]])

    _, newStateCheck = cpm.ismember(states, B)
    expected = [0, 0, 0, 0, 0, 0, 0, 0]
    assert newStateCheck==expected

    newStateCheck = cpm.flip(newStateCheck)
    np.testing.assert_array_equal(newStateCheck, np.zeros_like(newStateCheck, dtype=bool))
    newState = states[newStateCheck, :]
    np.testing.assert_array_equal(newState, np.empty(shape=(0, 2)))
    #B = np.append(B, newState, axis=1)

    result = cpm.add_new_states(states, B)
    np.testing.assert_array_equal(result, B)


def test_add_new_states1():
    states = [{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}]
    B = [{0}, {1}, {2}]

    _, newStateCheck = cpm.ismember(states, B)
    expected = [0, 0, 0, 0, 0, 0, 0, 0]
    assert newStateCheck==expected

    newStateCheck = cpm.flip(newStateCheck)
    np.testing.assert_array_equal(newStateCheck, np.zeros_like(newStateCheck, dtype=bool))
    newState = [states[i] for i in newStateCheck if i]
    #np.testing.assert_array_equal(newState, np.empty(shape=(0, 2)))
    assert newState == []
    #B = np.append(B, newState, axis=1)

    result = cpm.add_new_states(states, B)
    assert result == B



def test_isinscope1ss():

    cpms = []
    # Travel times (systems)
    c7 = np.array([
    [1,3,1,3,3,3,3],
    [2,1,2,1,3,3,3],
    [3,1,2,2,3,3,3],
    [3,2,2,3,3,3,3]]) - 1

    A1 = variable.Variable(**{'name': 'A1', 'B': [{0},{1},{0, 1}], 'values': ['s', 'f']})
    A2 = variable.Variable(**{'name': 'A2', 'B': [{0},{1},{0, 1}], 'values': ['s', 'f']})
    A3 = variable.Variable(**{'name': 'A3', 'B': [{0},{1},{0, 1}], 'values': ['s', 'f']})
    A4 = variable.Variable(**{'name': 'A4', 'B': [{0},{1},{0, 1}], 'values': ['s', 'f']})
    A5 = variable.Variable(**{'name': 'A5', 'B': [{0},{1},{0, 1}], 'values': ['s', 'f']})
    A6 = variable.Variable(**{'name': 'A6', 'B': [{0},{1},{0, 1}], 'values': ['s', 'f']})
    A7 = variable.Variable(**{'name': 'A7', 'B': [{0},{1},{0, 1}], 'values': ['s', 'f']})

    for i in range(1, 7):
        m = cpm.Cpm(variables= [eval(f'A{i}')],
                      no_child = 1,
                      C = np.array([[1, 0]]).T,
                      p = [1, 1])
        cpms.append(m)

    vars_ = [A7, A1, A2, A3, A4, A5, A6]
    for i in range(7, 11):
        m = cpm.Cpm(variables= vars_,
                      no_child = 1,
                      C = c7,
                      p = [1, 1, 1, 1])
        cpms.append(m)

    result = cpm.isinscope([A1], cpms)
    expected = np.array([[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]]).T
    np.testing.assert_array_equal(expected, result)

    result = cpm.isinscope([A1, A2], cpms)
    expected = np.array([[1, 1, 0, 0, 0, 0, 1, 1, 1, 1]]).T
    np.testing.assert_array_equal(expected, result)


@pytest.fixture
def setup_iscompatible():

    M = {}
    v = {}

    v[1] = variable.Variable(name='1', B=[{0}, {1}],values=['Mild', 'Severe'])
    v[2] = variable.Variable(name='2', B=[{0}, {1}, {0, 1}], values=['Survive', 'Fail'])
    v[3] = variable.Variable(name='3', B=[{0}, {1}, {0, 1}], values=['Survive', 'Fail'])
    v[4] = variable.Variable(name='4', B=[{0}, {1}, {0, 1}], values=['Survive', 'Fail'])
    v[5] = variable.Variable(name='5', B=[{0}, {1}], values=['Survive', 'Fail'])

    M[1] = cpm.Cpm(variables=[v[1]], no_child=1, C = np.array([[1, 2]]).T - 1, p = np.array([0.9, 0.1]).T)
    M[2] = cpm.Cpm(variables=[v[2], v[1]], no_child=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]) - 1, p = np.array([0.99, 0.01, 0.9, 0.1]).T)
    M[3] = cpm.Cpm(variables=[v[3], v[1]], no_child=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]) - 1, p = np.array([0.95, 0.05, 0.85, 0.15]).T)
    M[4] = cpm.Cpm(variables=[v[4], v[1]], no_child=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]) - 1, p = np.array([0.99, 0.01, 0.9, 0.1]).T)
    M[5] = cpm.Cpm(variables=[v[5], v[2], v[3], v[4]], no_child=1, C = np.array([[2, 3, 3, 2], [1, 1, 3, 1], [1, 2, 1, 1], [2, 2, 2, 1]]) - 1, p = np.array([1, 1, 1, 1]).T)

    return M, v


def test_iscompatible1(setup_iscompatible):

    M, v = setup_iscompatible

    # M[2]
    C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]) - 1 #M[2].C
    variables = [v[2], v[1]]
    checkVars = [v[1]]
    checkStates = [1-1]
    result = cpm.iscompatible(C, variables, checkVars, checkStates)
    expected = np.array([1, 1, 0, 0])
    np.testing.assert_array_equal(expected, result)


def test_iscompatible1s(setup_iscompatible):
    # using string for checkVars, checkStates
    M, v = setup_iscompatible

    # M[2]
    C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]) - 1 #M[2].C
    variables = [v[2], v[1]]
    checkVars = ['1']
    checkStates = ['Mild']
    result = cpm.iscompatible(C, variables, checkVars, checkStates)
    expected = np.array([1, 1, 0, 0])
    np.testing.assert_array_equal(expected, result)


def test_iscompatible2(setup_iscompatible):

    M, v = setup_iscompatible

    # M[5]
    C = np.array([[2, 3, 3, 2], [1, 1, 3, 1], [1, 2, 1, 1], [2, 2, 2, 1]]) - 1
    variables = [v[5], v[2], v[3], v[4]]
    checkVars = [v[3], v[4]]
    checkStates = [1-1, 1-1]
    result = cpm.iscompatible(C, variables, checkVars, checkStates)
    expected = np.array([0, 1, 1, 0])
    np.testing.assert_array_equal(expected, result)


def test_iscompatible3(setup_iscompatible):

    M, v = setup_iscompatible

    #M[1]
    C = np.array([[1, 2]]).T - 1
    variables = [v[1]]
    checkVars = [v[3], v[4]]
    checkStates = [1-1, 1-1]

    result = cpm.iscompatible(C, variables, checkVars, checkStates)
    expected = np.array([1, 1])
    np.testing.assert_array_equal(expected, result)


def test_iscompatible3ss():

    A1 = variable.Variable(name='A1', B=[{0},{1}],values=['Mild', 'Severe'])
    A2 = variable.Variable(name='A2', B=[{0},{1},{0, 1}], values=['Survive', 'Fail'])
    A3 = variable.Variable(name='A3', B=[{0},{1},{0, 1}], values=['Survive', 'Fail'])
    A4 = variable.Variable(name='A4', B=[{0},{1},{0, 1}], values=['Survive', 'Fail'])
    A5 = variable.Variable(name='A5', B=[{0},{1}], values=['Survive', 'Fail'])

    #M[1]
    C = np.array([[1, 2]]).T - 1
    variables = [A1]
    checkVars = [A3, A4]
    checkStates = [1, 1]
    # FIXME: redundant dict

    result = cpm.iscompatible(C, variables, checkVars, checkStates)
    expected = np.array([1, 1])
    np.testing.assert_array_equal(expected, result)


def test_iscompatible4(setup_iscompatible):

    M, v = setup_iscompatible

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
         [2,2,2,2,2]]) - 1
    variables = [v[2], v[3], v[5], v[1], v[4]]
    checkVars = [v[2], v[1]]
    checkStates = np.array([1-1, 1-1])

    result = cpm.iscompatible(C, variables, checkVars, checkStates)
    expected = np.array([1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0])
    np.testing.assert_array_equal(expected, result)

    _, idx = cpm.ismember(checkVars, variables)
    # should be one less than the Matlab result
    assert idx==[0, 3]

    checkVars = cpm.get_value_given_condn(checkVars, idx)
    assert checkVars==[v[2], v[1]]

    checkStates = cpm.get_value_given_condn(checkStates, idx)
    assert checkStates==[1-1, 1-1]

    C1_common = C1_common = C[:, idx].copy()
    compatFlag = np.ones(shape=C.shape[0], dtype=bool)
    B = checkVars[0].B
    C1 = C1_common[:, 0][np.newaxis, :]
    #x1_old = [B[k-1, :] for k in C1][0]
    x1 = [B[k] for k in C[compatFlag, 0]]
    x2 = B[checkStates[0]]
    #compatCheck = (np.sum(x1 * x2, axis=1) > 0)[:, np.newaxis]
    compatCheck = [bool(B[checkStates[0]].intersection(x)) for x in x1]

    expected = [{0},
                {1},
                {0},
                {1},
                {0},
                {1},
                {0},
                {1},
                {0},
                {1},
                {0},
                {1},
                {0},
                {1},
                {0},
                {1}]
    np.testing.assert_array_equal(x1, expected)
    assert x2 == {0}

    expected = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]).T
    np.testing.assert_array_equal(compatCheck, expected)


def test_get_subset1(setup_iscompatible):

    M = setup_iscompatible[0]

    # M[5]
    rowIndex = [0]  # 1 -> 0
    result = M[5].get_subset(rowIndex)

    np.testing.assert_array_equal(result.C, np.array([[2, 3, 3, 2]])-1)
    np.testing.assert_array_equal(result.p, [[1]])

def test_get_subset2(setup_iscompatible):

    M = setup_iscompatible[0]

    # M[5]
    rowIndex = [1, 2, 3]  # [2, 3, 4] -> 0
    result = M[5].get_subset(rowIndex, 0)

    np.testing.assert_array_equal(result.C, np.array([[2, 3, 3, 2]])-1)
    np.testing.assert_array_equal(result.p, [[1]])


def test_get_subset3(setup_iscompatible):

    _, v = setup_iscompatible
    M = cpm.Cpm(variables=[v[2], v[3], v[5], v[1], v[4]],
            no_child = 5,
            C=np.array([[2, 2, 2, 2, 2]]) - 1,
            p=np.array([[0.0150]]).T)

    result = M.get_subset([0], 0)

    assert result.C.any() == False
    assert result.p.any() == False

def test_iscompatibleCpm1(setup_iscompatible):

    # M[5]
    M, _ = setup_iscompatible
    rowIndex = [0]  # 1 -> 0
    M_sys_select = M[5].get_subset(rowIndex)
    result = M[3].iscompatible(M_sys_select, flag=True)
    expected = np.array([1, 1, 1, 1])
    np.testing.assert_array_equal(result, expected)


def test_iscompatibleCpm1s(var_A1_to_A5):

    A1, A2, A3, A4, A5 = var_A1_to_A5

    M = {}
    M[1] = cpm.Cpm(variables=[A1], no_child=1, C = np.array([[1, 2]]).T - 1, p = np.array([0.9, 0.1]).T)
    M[2] = cpm.Cpm(variables=[A2, A1], no_child=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]) - 1, p = np.array([0.99, 0.01, 0.9, 0.1]).T)
    M[3] = cpm.Cpm(variables=[A3, A1], no_child=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]) - 1, p = np.array([0.95, 0.05, 0.85, 0.15]).T)
    M[4] = cpm.Cpm(variables=[A4, A1], no_child=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]) - 1, p = np.array([0.99, 0.01, 0.9, 0.1]).T)
    M[5] = cpm.Cpm(variables=[A5, A2, A3, A4], no_child=1, C = np.array([[2, 3, 3, 2], [1, 1, 3, 1], [1, 2, 1, 1], [2, 2, 2, 1]]) - 1, p = np.array([1, 1, 1, 1]).T)

    # M[5]
    rowIndex = [0]  # 1 -> 0
    M_sys_select = M[5].get_subset(rowIndex)
    result = M[3].iscompatible(M_sys_select, flag=True)
    expected = np.array([1, 1, 1, 1])
    np.testing.assert_array_equal(result, expected)


def test_iscompatibleCpm2(setup_iscompatible):

    # M[5]
    M, _ = setup_iscompatible
    rowIndex = [0]  # 1 -> 0
    M_sys_select = M[5].get_subset(rowIndex)

    result = M[4].iscompatible(M_sys_select, flag=True)
    expected = np.array([0, 1, 0, 1])
    np.testing.assert_array_equal(result, expected)

def test_iscompatibleCpm3(setup_iscompatible):

    # M[5]
    M, _ = setup_iscompatible
    rowIndex = [0]  # 1 -> 0
    M_sys_select = M[5].get_subset(rowIndex)

    result = M[1].iscompatible(M_sys_select, flag=True)
    expected = np.array([1, 1])
    np.testing.assert_array_equal(result, expected)


def test_iscompatibleCpm4(setup_bridge):

    _, vars_arc = setup_bridge

    #M.iscompatible should be TFFF not TFFT
    M = cpm.Cpm(variables=[vars_arc[x] for x in ['od1', 'e2', 'e3', 'e4', 'e5', 'e6']],
            no_child=6,
            C=np.array([[2, 1, 1, 2, 2, 2],
               [0, 0, 2, 2, 2, 2],
               [0, 0, 2, 2, 2, 2],
               [2, 1, 2, 2, 2, 2]]),
            p=np.array([[0.839],
               [0.839],
               [0.161],
               [0.161]]))

    Mc = cpm.Cpm(variables=[vars_arc[x] for x in ['od1', 'e2', 'e3', 'e4', 'e5', 'e6']],
             no_child=6,
             C=np.array([[2, 1, 1, 2, 2, 2]]),
             p=np.array([[0.839]]))
    result = M.iscompatible(Mc, flag=False)

    expected = np.array([True, False, False, False])
    np.testing.assert_array_equal(result, expected)


@pytest.fixture
def setup_product():

    X1 = variable.Variable(name='X1', B=[{0}, {1}],values=['Mild', 'Severe'])
    X2 = variable.Variable(name='X2', B=[{0}, {1}, {0, 1}], values=['Survive', 'Fail'])
    X3 = variable.Variable(name='X3', B=[{0}, {1}, {0, 1}], values=['Survive', 'Fail'])
    X4 = variable.Variable(name='X4', B=[{0}, {1}, {0, 1}], values=['Survive', 'Fail'])
    X5 = variable.Variable(name='X5', B=[{0}, {1}], values=['Survive', 'Fail'])

    M = {}
    M[2] = cpm.Cpm(variables=[X2, X1], no_child=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]) - 1, p = np.array([0.99, 0.01, 0.9, 0.1]).T)
    M[3] = cpm.Cpm(variables=[X3, X1], no_child=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]) - 1, p = np.array([0.95, 0.05, 0.85, 0.15]).T)
    M[5] = cpm.Cpm(variables=[X5, X2, X3, X4], no_child=1, C = np.array([[2, 3, 3, 2], [1, 1, 3, 1], [1, 2, 1, 1], [2, 2, 2, 1]]) - 1, p = np.array([1, 1, 1, 1]).T)

    return M

def test_product1(setup_product):

    M = setup_product

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

    #assert list(commonVars) == [1]

    _, idxVarsM1 = cpm.ismember(M1.variables, M2.variables)
    commonVars = cpm.get_value_given_condn(M1.variables, idxVarsM1)

    np.testing.assert_array_equal(idxVarsM1, np.array([0, 1]))
    #assert commonVars==[1]

    for i in range(M1.C.shape[0]):
        c1_ = cpm.get_value_given_condn(M1.C[i, :], idxVarsM1)
        c1_notCommon = M1.C[i, cpm.flip(idxVarsM1)]

        if any(M1.sample_idx):
            sampleInd1 = M1.sample_idx[i]
        else:
            sampleInd1 = []

        #if isinstance(commonVars, list):
        #    commonVars = np.array(commonVars)

        #if isinstance(c1_, list):
        #    c1_ = np.array(c1_)
        [M2_] = cpm.condition([M2], commonVars, c1_, sampleInd1)

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
            _prod = cpm.get_prod(M2_.p, M1.p[i])
            #np.testing.assert_array_equal(pproductVal, np.array([[0.9405, 0.0495]]).T)
            #pproduct = np.array([])
            if i:
                #pprod = np.append(pprod, _prod, axis=0.reshape(_prod.shape[0], -1)

                pprod = np.append(pprod, _prod, axis=0)
            else:
                pprod = _prod

    np.testing.assert_array_almost_equal(pprod, np.array([[0.9405, 0.0495, 0.0095, 0.0005, 0.7650, 0.1350, 0.0850, 0.0150]]).T)
    np.testing.assert_array_almost_equal(Cprod, np.array([[1, 1, 1], [2, 1, 1], [1, 1, 2], [2, 1, 2], [1, 2, 1], [2, 2, 1], [1, 2, 2], [2, 2, 2]])-1)

    Cprod_vars = M2.variables + cpm.get_value_given_condn(M1.variables, cpm.flip(idxVarsM1))
    assert [x.name for x in Cprod_vars]==['X3', 'X1', 'X2']

    newVarsChild = M1.variables[:M1.no_child] + M2.variables[:M2.no_child]
    #newVarsChild = np.sort(newVarsChild)
    assert [x.name for x in newVarsChild] ==['X2', 'X3']

    newVarsParent = M1.variables[M1.no_child:] + M2.variables[M2.no_child:]
    newVarsParent, _ = cpm.setdiff(newVarsParent, newVarsChild)
    newVars = newVarsChild + newVarsParent
    assert [x.name for x in newVars]== ['X2', 'X3', 'X1']

    _, idxVars = cpm.ismember(newVars, Cprod_vars)

    assert idxVars==[2, 0, 1] # matlab 3, 1, 2

    Mprod = cpm.Cpm(variables=newVars,
                no_child = len(newVarsChild),
                C = Cprod[:, idxVars],
                p = pprod)

    Mprod.sort()

    assert [x.name for x in Mprod.variables]==['X2', 'X3', 'X1']
    assert Mprod.no_child==2
    np.testing.assert_array_equal(Mprod.C, np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1], [1, 1, 2], [2, 1, 2], [1, 2, 2], [2, 2, 2]])-1)
    np.testing.assert_array_almost_equal(Mprod.p, np.array([[0.9405, 0.0095, 0.0495, 5.0e-4, 0.7650, 0.0850, 0.1350, 0.0150]]).T)

def test_product2(setup_product):

    M = setup_product
    M1 = M[2]
    M2 = M[3]

    Mprod = M1.product(M2)

    names = [x.name for x in Mprod.variables]
    assert names == ['X2', 'X3', 'X1']

    assert Mprod.no_child==2
    np.testing.assert_array_equal(Mprod.C, np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1], [1, 1, 2], [2, 1, 2], [1, 2, 2], [2, 2, 2]]) - 1)
    np.testing.assert_array_almost_equal(Mprod.p, np.array([[0.9405, 0.0095, 0.0495, 5.0e-4, 0.7650, 0.0850, 0.1350, 0.0150]]).T)

def test_product3(setup_product):

    M = setup_product
    X2 = M[5].get_variables('X2')
    X3 = M[5].get_variables('X3')
    X1 = M[2].get_variables('X1')

    M2 = M[5]

    M1 = cpm.Cpm(variables=[X2, X3, X1], no_child=2, C = np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1], [1, 1, 2], [2, 1, 2], [1, 2, 2], [2, 2, 2]])-1, p = np.array([[0.9405, 0.0095, 0.0495, 5.0e-4, 0.7650, 0.0850, 0.1350, 0.0150]]).T)

    Mprod= M1.product(M2)

    names = [x.name for x in Mprod.variables]
    assert names == ['X2', 'X3', 'X5', 'X1', 'X4']
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
          [2,2,2,2,2]]) - 1

    expected_p = np.array([[0.9405,0.0095,0.0495,0.0005,0.7650,0.0850,0.1350,0.0150,0.9405,0.0095,0.0495,0.0005,0.7650,0.0850,0.1350,0.0150]]).T
    np.testing.assert_array_equal(Mprod.C, expected_C)
    np.testing.assert_array_almost_equal(Mprod.p, expected_p)


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
         [2,2,2,2,2]]) - 1
    p = np.array([[0.9405, 0.0095, 0.0495, 0.0005, 0.7650, 0.0850, 0.1350, 0.0150, 0.9405, 0.0095, 0.0495, 0.0005, 0.7650, 0.0850, 0.1350, 0.0150]]).T
    v1 = variable.Variable(name='v1', B=[{0}, {1}],values=['Mild', 'Severe'])
    v2 = variable.Variable(name='v2', B=[{0}, {1}, {0, 1}], values=['Survive', 'Fail'])
    v3 = variable.Variable(name='v3', B=[{0}, {1}, {0, 1}], values=['Survive', 'Fail'])
    v4 = variable.Variable(name='v4', B=[{0}, {1}, {0, 1}], values=['Survive', 'Fail'])
    v5 = variable.Variable(name='v5', B=[{0}, {1}], values=['Survive', 'Fail'])

    Mx = cpm.Cpm(variables=[v2, v3, v5, v1, v4], no_child=3, C = C, p = p)

    return Mx


def test_condition0(setup_condition):

    Mx = setup_condition
    v2, v1 = Mx.get_variables(['v2', 'v1'])

    condVars = [v2]
    condStates = np.array([1-1])

    Mx = cpm.Cpm(variables=[v2, v1],
             no_child = 1,
             C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]) - 1,
             p = np.array([[0.99, 0.01, 0.9, 0.1]]).T)

    compatFlag = cpm.iscompatible(Mx.C, Mx.variables, condVars, condStates)
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

    expected = np.array([[1, 1], [1, 2]]) - 1
    np.testing.assert_array_equal(expected, Ccompat)

    _, idxInC = cpm.ismember(condVars, Mx.variables)
    np.testing.assert_array_equal(idxInC, [0])  # matlab 1 though

    _, idxInCondVars = cpm.ismember(Mx.variables, condVars)
    np.testing.assert_array_equal(idxInCondVars, [0, False])  # matlab 1 though
    not_idxInCondVars = cpm.flip(idxInCondVars)
    assert not_idxInCondVars==[False, True]
    Ccond = np.zeros_like(Ccompat)
    Ccond[:, not_idxInCondVars] = cpm.get_value_given_condn(Ccompat, not_idxInCondVars)
    #np.testing.assert_array_equal(Ccond_, Ccompat[:, 1:])
    #Ccond[:, new_cond] = Ccond_
    expected = np.array([[0, 0], [0, 1]])
    np.testing.assert_array_equal(Ccond, expected)

    _condVars = cpm.get_value_given_condn(condVars, idxInC)
    _condStates = cpm.get_value_given_condn(condStates, idxInC)
    _idxInC = cpm.get_value_given_condn(idxInC, idxInC)
    assert _condVars == [v2]
    assert _condStates== np.array([0])
    assert _idxInC== np.array([0])

    B = _condVars[0].B
    np.testing.assert_array_equal(B, [{0}, {1}, {0, 1}])

    # FIXME: index or not
    _Ccompat = Ccompat[:, _idxInC[0]].copy()
    np.testing.assert_array_equal(_Ccompat, [0, 0])

    expected = [{0}, {0}]
    assert [B[x] for x in _Ccompat] == expected
    #np.testing.assert_array_equal(B[_Ccompat.flatten()], expected)
    # FIXME: index or not
    assert B[_condStates[0]] == {0}
    #np.testing.assert_array_equal(B[_condStates, :], np.array([1, 0]))
    #compatCheck_mv = B[_Ccompat.flatten(), :] * B[_condStates[0], :]
    compatCheck_mv = [B[x].intersection(B[_condStates[0]]) for x in _Ccompat]
    assert compatCheck_mv == expected
    #np.testing.assert_array_equal(compatCheck_mv, expected)

    B = cpm.add_new_states(compatCheck_mv, B)
    _condVars[0] = variable.Variable(name=_condVars[0].name,
                            B=B,
                            values=_condVars[0].values)
    #_condVars[0].B = B

    # FIXME: index or not
    Ccond[:, _idxInC[0]] = [x for x in cpm.ismember(compatCheck_mv, B)[1]]

    # Need to confirm whether 
    expected = np.array([[1, 1], [1, 2]]) - 1
    np.testing.assert_array_equal(Ccond, expected)

    # Mx.p
    expected = np.array([[0.9405,0.0495,0.7650,0.1350,0.9405,0.0495,0.7650,0.1350]]).T
    expected = np.array([[0.99, 0.9]]).T
    np.testing.assert_array_equal(Mx.p[compatFlag], expected)

def test_condition1(setup_condition):

    Mx = setup_condition
    v2 = Mx.get_variables('v2')
    condVars = [v2]
    condStates = np.array([1-1])

    [M_n] = cpm.condition([Mx], condVars, condStates)
    names = [x.name for x in M_n.variables]
    assert names==['v2', 'v3', 'v5', 'v1', 'v4']
    assert M_n.no_child == 3
    expected = np.array([[1,1,1,1,1],
                        [1,2,1,1,1],
                        [1,1,1,2,1],
                        [1,2,1,2,1],
                        [1,1,2,1,2],
                        [1,2,2,1,2],
                        [1,1,2,2,2],
                        [1,2,2,2,2]]) - 1
    np.testing.assert_array_equal(M_n.C, expected)

    expected = np.array([[0.9405,0.0495,0.7650,0.1350,0.9405,0.0495,0.7650,0.1350]]).T
    np.testing.assert_array_equal(M_n.p, expected)
    assert M_n.q.any() == False
    assert M_n.sample_idx.any() == False

def test_condition2(setup_condition):

    Mx = setup_condition
    v1 = Mx.get_variables('v1')
    condVars = [v1]
    condStates = np.array([1-1])

    [M_n] = cpm.condition([Mx], condVars, condStates)

    names = [x.name for x in M_n.variables]
    assert names==['v2', 'v3', 'v5', 'v1', 'v4']
    assert M_n.no_child==3
    expected = np.array([[1,1,1,1,1],
                        [2,1,1,1,1],
                        [1,2,1,1,1],
                        [2,2,2,1,1],
                        [1,1,2,1,2],
                        [2,1,2,1,2],
                        [1,2,2,1,2],
                        [2,2,2,1,2]]) - 1

    np.testing.assert_array_equal(M_n.C, expected)

    expected = np.array([[0.9405,0.0095,0.0495,0.0005,0.9405,0.0095,0.0495,0.0005]]).T
    np.testing.assert_array_equal(M_n.p, expected)
    assert M_n.q.any() == False
    assert M_n.sample_idx.any() == False

def test_condition3(setup_condition):
    # conditioning on multiple nodes
    Mx = setup_condition
    v2 = Mx.get_variables('v2')
    v1 = Mx.get_variables('v1')
    condVars = [v2, v1]
    condStates = np.array([1-1, 1-1])

    [M_n]= cpm.condition([Mx], condVars, condStates)

    names = [x.name for x in M_n.variables]
    assert names==['v2', 'v3', 'v5', 'v1', 'v4']
    assert M_n.no_child==3
    expected = np.array([[1,1,1,1,1],
                        [1,2,1,1,1],
                        [1,1,2,1,2],
                        [1,2,2,1,2]]) - 1

    np.testing.assert_array_equal(M_n.C, expected)

    expected = np.array([[0.9405,0.0495,0.9405,0.0495]]).T
    np.testing.assert_array_equal(M_n.p, expected)
    assert M_n.q.any() == False
    assert M_n.sample_idx.any() == False


def test_condition4(setup_condition):

    Mx_ = setup_condition

    v2, v3, v5, v4 = Mx_.get_variables(['v2', 'v3', 'v5', 'v4'])

    C = np.array([[2, 3, 3, 2],
                 [1, 1, 3, 1],
                 [1, 2, 1, 1],
                 [2, 2, 2, 1]]) - 1
    p = np.array([1, 1, 1, 1, ])
    Mx = cpm.Cpm(variables=[v5, v2, v3, v4], no_child=1, C = C, p = p.T)
    condVars = [v2, v3]
    condStates = np.array([1-1, 1-1])

    result = cpm.iscompatible(Mx.C, Mx.variables, condVars, condStates)
    expected = np.array([1,1,0,0])
    np.testing.assert_array_equal(expected, result)

    [M_n] = cpm.condition([Mx], condVars, condStates)

    assert M_n.variables == [v5, v2, v3, v4]
    assert M_n.no_child==1
    expected = np.array([[2,1,1,2],
                         [1,1,1,1]]) - 1
    np.testing.assert_array_equal(M_n.C, expected)

    expected = np.array([[1, 1]]).T
    np.testing.assert_array_equal(M_n.p, expected)
    assert M_n.q.any() == False
    assert M_n.sample_idx.any() == False

def test_condition4s(setup_condition):
    # using string for cond vars, condstates
    Mx_ = setup_condition

    v2, v3, v5, v4 = Mx_.get_variables(['v2', 'v3', 'v5', 'v4'])

    C = np.array([[2, 3, 3, 2],
                 [1, 1, 3, 1],
                 [1, 2, 1, 1],
                 [2, 2, 2, 1]]) - 1
    p = np.array([1, 1, 1, 1, ])
    Mx = cpm.Cpm(variables=[v5, v2, v3, v4], no_child=1, C = C, p = p.T)

    #condVars = [v2, v3]
    condVars = ['v2', 'v3']
    #condStates = np.array([1-1, 1-1])
    condStates = ['Survive', 'Survive']
    result = cpm.iscompatible(Mx.C, Mx.variables, condVars, condStates)
    expected = np.array([1,1,0,0])
    np.testing.assert_array_equal(expected, result)

    # using string instead of variables
    [M_n] = cpm.condition([Mx], ['v2', 'v3'], condStates)

    assert M_n.variables == [v5, v2, v3, v4]
    assert M_n.no_child==1
    expected = np.array([[2,1,1,2],
                         [1,1,1,1]]) - 1
    np.testing.assert_array_equal(M_n.C, expected)

    expected = np.array([[1, 1]]).T
    np.testing.assert_array_equal(M_n.p, expected)
    assert M_n.q.any() == False
    assert M_n.sample_idx.any() == False


def test_condition5(setup_condition):

    Mx = setup_condition
    v3, v1 = Mx.get_variables(['v3', 'v1'])
    C = np.array([[1, 1],
                 [2, 1],
                 [1, 2],
                 [2, 2]]) - 1
    p = np.array([0.95, 0.05, 0.85, 0.15])
    M2 = cpm.Cpm(variables=[v3, v1], no_child=1, C = C, p = p.T)
    condVars = [v1]
    states = np.array([2-1])

    [M_n]= cpm.condition([M2], condVars, states)

    assert M_n.variables== [v3, v1]
    assert M_n.no_child== 1
    expected = np.array([[1,2],
                         [2,2]]) - 1
    np.testing.assert_array_equal(M_n.C, expected)

    expected = np.array([[0.85, 0.15]]).T
    np.testing.assert_array_equal(M_n.p, expected)

def test_condition6(setup_condition):

    Mx = setup_condition
    v1 = Mx.get_variables('v1')
    C = np.array([[1, 2]]).T - 1
    p = np.array([0.9, 0.1])

    M2 = cpm.Cpm(variables=[v1], no_child=1, C = C, p = p.T)
    condVars = np.array([])
    states = np.array([])

    [M_n]= cpm.condition([M2], condVars, states)

    assert M_n.variables==[v1]
    assert M_n.no_child== 1
    expected = np.array([[1,2]]).T - 1
    np.testing.assert_array_equal(M_n.C, expected)

    expected = np.array([[0.9, 0.1]]).T
    np.testing.assert_array_equal(M_n.p, expected)


@pytest.fixture
def setup_sum(var_A1_to_A5):

    A1, A2, A3, A4, A5 = var_A1_to_A5

    variables = [A2, A3, A5, A1, A4]
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
          [2,2,2,2,2]]) - 1

    p = np.array([[0.9405,0.0095,0.0495,0.0005,0.7650,0.0850,0.1350,0.0150,0.9405,0.0095,0.0495,0.0005,0.7650,0.0850,0.1350,0.0150]]).T

    return {'variables': variables,
            'no_child': no_child,
            'C': C,
            'p': p}

def test_sum1(setup_sum):

    M = cpm.Cpm(**setup_sum)
    A1, A2, A3, A4, A5 = M.get_variables(['A1', 'A2', 'A3', 'A4', 'A5'])
    sumVars = [A1]
    varsRemainIdx = cpm.ismember(sumVars, M.variables[:M.no_child])

    sumFlag = 1
    if sumFlag:
        varsRemain, varsRemainIdx = cpm.setdiff(M.variables[:M.no_child], sumVars)
        assert [x.name for x in varsRemain]== ['A2', 'A3', 'A5']
        assert varsRemainIdx== [0, 1, 2]  # Matlab: [1, 2, 3]
    else:
        varsRemainIdx = cpm.get_value_given_condn(varsRemainIdx, varsRemainIdx)
        assert varsRemainIdx== []
        varsRemain = cpm.get_value_given_condn(M.variables, varsRemainIdx)

    no_child = len(varsRemain)

    if any(M.variables[M.no_child:]):
        varsRemain += M.variables[M.no_child:]
        varsRemainIdx += list(range(M.no_child, len(M.variables)))

    assert varsRemain ==[A2, A3, A5, A1, A4]
    np.testing.assert_array_equal(varsRemainIdx, [0, 1, 2, 3, 4])

    Mloop = cpm.Cpm(variables=cpm.get_value_given_condn(M.variables, varsRemainIdx),
                C=M.C[:, varsRemainIdx],
                p=M.p,
                q=M.q,
                sample_idx=M.sample_idx,
                no_child=len(varsRemainIdx))
    i = 0
    while Mloop.C.any():
        Mcompare = Mloop.get_subset([0]) # need to change to 0 
        if i==0:
            assert Mcompare.variables== [A2, A3, A5, A1, A4]
            np.testing.assert_array_equal(Mcompare.no_child, 5)
            np.testing.assert_array_equal(Mcompare.p, np.array([[0.9405]]).T)
            np.testing.assert_array_equal(Mcompare.C, np.array([[1, 1, 1, 1, 1]])-1)
            assert Mcompare.q.any()==False
            assert Mcompare.sample_idx.any() == False

        flag = Mloop.iscompatible(Mcompare, flag=True)
        expected = np.zeros(16)
        expected[0] = 1
        if i==0:
            np.testing.assert_array_equal(flag, expected)

        if i==0:
            Csum = Mloop.C[0, :][np.newaxis, :]
        else:
            Csum = np.append(Csum, Mloop.C[0, :][np.newaxis, :], axis=0)

        if i==0:
            np.testing.assert_array_equal(Csum, np.array([[1, 1, 1, 1, 1]])-1)

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
                              [2,2,2,2,2]]) - 1
        expected_p = np.array([[0.0095,0.0495,0.0005,0.7650,0.0850,0.1350,0.0150,0.9405,0.0095,0.0495,0.0005,0.7650,0.0850,0.1350,0.0150]]).T
        if i==0:
            assert Mloop.variables== [A2, A3, A5, A1, A4]
            np.testing.assert_array_equal(Mloop.no_child, 5)
            np.testing.assert_array_equal(Mloop.p, expected_p)
            np.testing.assert_array_equal(Mloop.C, expected_C)
            assert Mloop.q.any()==False
            assert Mloop.sample_idx.any()==False
        i += 1

    Msum = cpm.Cpm(variables=varsRemain,
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
                        [2,2,2,2,2]])  - 1

    expected_p = np.array([[0.9405,0.0095,0.0495,0.0005,0.7650,0.0850,0.1350,0.0150,0.9405,0.0095,0.0495,0.0005,0.7650,0.0850,0.1350,0.0150]]).T
    np.testing.assert_array_equal(Msum.C, expected_C)
    np.testing.assert_array_equal(Msum.p, expected_p)

def test_sum2(setup_sum):

    M = cpm.Cpm(**setup_sum)
    A1 = M.get_variables('A1')
    sumVars = [A1]
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
                        [2,2,2,2,2]]) - 1

    expected_p = np.array([[0.9405,0.0095,0.0495,0.0005,0.7650,0.0850,0.1350,0.0150,0.9405,0.0095,0.0495,0.0005,0.7650,0.0850,0.1350,0.0150]]).T
    np.testing.assert_array_equal(Ms.C, expected_C)
    np.testing.assert_array_equal(Ms.p, expected_p)

def test_sum3(setup_sum):

    M = cpm.Cpm(**setup_sum)
    A2, A3 = M.get_variables(['A2', 'A3'])

    sumVars = [A2, A3]
    Ms = M.sum(sumVars)
    expected_C = np.array([[1,1,1],
                          [2,1,1],
                          [1,2,1],
                          [2,2,1],
                          [2,1,2],
                          [2,2,2]]) - 1
    expected_p = np.array([[0.9995, 0.0005,0.985, 0.015, 1.00, 1.00]]).T

    np.testing.assert_array_equal(Ms.C, expected_C)
    np.testing.assert_array_almost_equal(Ms.p, expected_p)
    assert [x.name for x in Ms.variables]== ['A5', 'A1', 'A4']

def test_sum4(setup_sum):

    M = cpm.Cpm(**setup_sum)
    A5 = M.get_variables('A5')

    sumVars = [A5]
    Ms = M.sum(sumVars, flag=0)
    expected_C = np.array([[1,1,1],
                          [2,1,1],
                          [1,2,1],
                          [2,2,1],
                          [2,1,2],
                          [2,2,2]]) - 1
    expected_p = np.array([[0.9995, 0.0005,0.985, 0.015, 1.00, 1.00]]).T

    np.testing.assert_array_equal(Ms.C, expected_C)
    np.testing.assert_array_almost_equal(Ms.p, expected_p)
    assert [x.name for x in Ms.variables]== ['A5', 'A1', 'A4']
    assert Ms.no_child== 1

def test_sum5(setup_sum):

    M = cpm.Cpm(**setup_sum)
    #A5 = M.get_variables('A5')

    sumVars = ['A5']
    Ms = M.sum(sumVars, flag=0)
    expected_C = np.array([[1,1,1],
                          [2,1,1],
                          [1,2,1],
                          [2,2,1],
                          [2,1,2],
                          [2,2,2]]) - 1
    expected_p = np.array([[0.9995, 0.0005,0.985, 0.015, 1.00, 1.00]]).T

    np.testing.assert_array_equal(Ms.C, expected_C)
    np.testing.assert_array_almost_equal(Ms.p, expected_p)
    assert [x.name for x in Ms.variables]== ['A5', 'A1', 'A4']
    assert Ms.no_child== 1


def test_sum6(setup_bridge):

    cpms_arc, vars_arc = setup_bridge
    cpms_arc_cp = [cpms_arc[k] for k in ['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'od1']]

    is_inscope = cpm.isinscope([vars_arc['e1']], cpms_arc_cp)
    cpm_sel = [y for x, y in zip(is_inscope, cpms_arc_cp) if x]
    cpm_mult = cpm.prod_cpms(cpm_sel)

    assert [x.name for x in cpm_mult.variables] == ['e1', 'od1', 'e2', 'e3', 'e4', 'e5', 'e6']
    expected_C = np.array([[1, 2, 2, 1, 3, 3, 3],
                           [1, 3, 2, 2, 3, 3, 3],
                           [1, 1, 1, 3, 3, 3, 3],
                           [2, 1, 1, 3, 3, 3, 3],
                           [2, 3, 2, 3, 3, 3, 3]]) - 1
    np.testing.assert_array_equal(cpm_mult.C, expected_C)
    np.testing.assert_array_almost_equal(cpm_mult.p, np.array([[0.8390, 0.8390, 0.8390, 0.1610, 0.1610]]).T, decimal=4)

    a = cpm_mult.sum(['e1'])
    assert [x.name for x in a.variables] == ['od1', 'e2', 'e3', 'e4', 'e5', 'e6']
    expected_C = np.array([[2, 2, 1, 3, 3, 3],
                           [3, 2, 2, 3, 3, 3],
                           [1, 1, 3, 3, 3, 3],
                           [3, 2, 3, 3, 3, 3]]) - 1
    np.testing.assert_array_equal(a.C, expected_C)
    np.testing.assert_array_almost_equal(a.p, np.array([[0.8390, 0.8390, 1.0, 0.1610]]).T, decimal=4)


@pytest.fixture
def setup_mcs_product():

    v1 = variable.Variable(name='v1', B=[{0}, {1}],values=['Mild', 'Severe'])
    v2 = variable.Variable(name='v2', B=[{0}, {1}, {0, 1}], values=['Survive', 'Fail'])
    v3 = variable.Variable(name='v3', B=[{0}, {1}, {0, 1}], values=['Survive', 'Fail'])
    v4 = variable.Variable(name='v4', B=[{0}, {1}, {0, 1}], values=['Survive', 'Fail'])
    v5 = variable.Variable(name='v5', B=[{0}, {1}], values=['Survive', 'Fail'])

    M = {}
    M[1] = cpm.Cpm(variables=[v1],
                   no_child=1,
                   C = np.array([1, 2]).T - 1,
                   p = np.array([0.9, 0.1]).T)

    M[2]= cpm.Cpm(variables=[v2, v1],
                   no_child=1,
                   C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]) - 1,
                   p = np.array([0.99, 0.01, 0.9, 0.1]).T)

    M[3] = cpm.Cpm(variables=[v3, v1],
                   no_child=1,
                   C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]) - 1,
                   p = np.array([0.95, 0.05, 0.85, 0.15]).T)

    M[4] = cpm.Cpm(variables=[v4, v1],
                   no_child=1,
                   C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]) - 1,
                   p = np.array([0.99, 0.01, 0.9, 0.1]).T)

    M[5] = cpm.Cpm(variables=[v5, v2, v3, v4],
                   no_child=1,
                   C = np.array([[2, 3, 3, 2], [1, 1, 3, 1], [1, 2, 1, 1], [2, 2, 2, 1]]) -1,
                   p = np.array([1, 1, 1, 1]).T)

    return M


def test_get_sample_order(setup_mcs_product):

    cpms = setup_mcs_product
    cpms = [cpms[k] for k in [1, 2, 3]]

    sampleOrder, sampleVars, varAdditionOrder = cpm.get_sample_order(cpms)

    expected = [0, 1, 2]
    np.testing.assert_array_equal(sampleOrder, expected)
    np.testing.assert_array_equal(varAdditionOrder, expected)

    expected = ['v1', 'v2', 'v3']
    result = [x.name for x in sampleVars]
    np.testing.assert_array_equal(result, expected)

def test_get_prod_idx1(setup_mcs_product):

    cpms = setup_mcs_product
    cpms = [cpms[k] for k in [1, 2, 3]]

    result = cpm.get_prod_idx(cpms, [])

    #expected = [1, 0, 0]
    expected = 0

    np.testing.assert_array_equal(result, expected)

def test_get_prod_idx2(setup_mcs_product):

    cpms = setup_mcs_product
    cpms = {k:cpms[k] for k in [1, 2, 3]}

    result = cpm.get_prod_idx(cpms, [])

    #expected = [1, 0, 0]
    expected = 0

    np.testing.assert_array_equal(result, expected)

def test_single_sample1(setup_mcs_product):

    cpms = setup_mcs_product
    cpms = [cpms[k] for k in [1, 2, 3]]

    v2, v1 = cpms[1].get_variables(['v2', 'v1'])
    v3 = cpms[2].get_variables('v3')

    sampleOrder = [0, 1, 2]
    sampleVars = [v1, v2, v3]
    varAdditionOrder = [0, 1, 2]
    sampleInd = [v1]
    sample, sampleProb = cpm.single_sample(cpms, sampleOrder, sampleVars, varAdditionOrder, sampleInd)

    if (sample == [1, 1, 1]).all():
        np.testing.assert_array_almost_equal(sampleProb, [[0.846]], decimal=3)
    elif (sample == [2, 1, 1]).all():
        np.testing.assert_array_almost_equal(sampleProb, [[0.0765]], decimal=3)

def test_single_sample2(setup_mcs_product):

    cpms = setup_mcs_product
    cpms = {k:cpms[k] for k in [1, 2, 3]}
    v2, v1 = cpms[2].get_variables(['v2', 'v1'])
    v3 = cpms[3].get_variables('v3')

    sampleOrder = [0, 1, 2]
    sampleVars = [v1, v2, v3]
    varAdditionOrder = [0, 1, 2]
    sampleInd = [1]

    sample, sampleProb = cpm.single_sample(cpms, sampleOrder, sampleVars, varAdditionOrder, sampleInd)

    if (sample == [1, 1, 1]).all():
        np.testing.assert_array_almost_equal(sampleProb, [[0.846]], decimal=3)
    elif (sample == [2, 1, 1]).all():
        np.testing.assert_array_almost_equal(sampleProb, [[0.0765]], decimal=3)

def test_mcs_product1(setup_mcs_product):

    cpms = setup_mcs_product
    cpms = {k:cpms[k] for k in [1, 2, 3]}

    nSample = 10
    Mcs = cpm.mcs_product(cpms, nSample)

    assert [x.name for x in Mcs.variables]==['v3', 'v2', 'v1']

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

    cpms = setup_mcs_product
    cpms = [cpms[k] for k in [1, 2, 3, 4, 5]]
    nSample = 10
    Mcs = cpm.mcs_product(cpms, nSample)

    assert [x.name for x in Mcs.variables]==['v5', 'v4', 'v3', 'v2', 'v1']
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

    cpms = setup_mcs_product
    cpms = {k+1:cpms[k] for k in [1, 2, 3, 4, 5]}
    nSample = 10
    Mcs = cpm.mcs_product(cpms, nSample)

    assert [x.name for x in Mcs.variables]==['v5', 'v4', 'v3', 'v2', 'v1']

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

    cpms_ = setup_mcs_product

    nSample = 10
    Mcs = cpm.mcs_product(cpms_, nSample)

    assert [x.name for x in Mcs.variables]==['v5', 'v4', 'v3', 'v2', 'v1']

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
    cpms = setup_mcs_product
    cpms = [cpms[k] for k in [2, 5]]
    with pytest.raises(TypeError):
        Mcs = cpm.mcs_product(cpms, nSample)

def test_get_value_given_condn1():

    condn = [1, False]
    value = [1,2]
    expected = [1]

    result = cpm.get_value_given_condn(value, condn)

    assert result==expected


def test_condition(setup_mcs_product):

    cpms = setup_mcs_product
    v1, v2 = cpms[2].get_variables(['v1', 'v2'])
    condVars = [v1, v2]
    condStates = np.array([1-1, 1-1])

    [M]= cpm.condition(cpms[3], condVars, condStates, [0])
    np.testing.assert_array_equal(M.C, np.array([[1, 1], [2, 1]])-1)
    assert M.q.any() == False
    assert M.sample_idx.any() == False


@pytest.fixture
def setup_prod_cms():
    v1 = variable.Variable(name='v1', B=[{0}, {1}, {2}], values=['Sunny', 'Cloudy', 'Rainy'])
    v2 = variable.Variable(name='v2', B=[{0}, {1}], values=['Good', 'Bad'])
    v3 = variable.Variable(name='v3', B=[{0}, {1}], values=['Below 0', 'Above 0'])

    M = {}
    M[1] = cpm.Cpm(variables=[v1],
                   no_child=1,
                   C = np.array([1, 2]).T,
                   p = np.array([0.9, 0.1]).T)

    M[2] = cpm.Cpm(variables=[v2, v1],
                   no_child=1,
                   C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]),
                   p = np.array([0.99, 0.01, 0.9, 0.1]).T)

    M[3] = cpm.Cpm(variables=[v3, v1],
                   no_child=1,
                   C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]),
                   p = np.array([0.95, 0.05, 0.85, 0.15]).T)
    return M


def test_prod_cpms1(setup_prod_cms):

    cpms= setup_prod_cms
    Mmult = cpm.prod_cpms(cpms=cpms)

    assert [x.name for x in Mmult.variables] == ['v1', 'v2', 'v3']

    expected = np.array([[1,1,1],[2,1,1],[1,2,1],[2,2,1],[1,1,2],[2,1,2],[1,2,2],[2,2,2]])
    np.testing.assert_array_equal(Mmult.C, expected)

    expected = np.array([[0.8464, 0.0765, 0.0086, 0.0085, 0.0446, 0.0135, 4.5e-4, 0.0015]]).T
    np.testing.assert_array_almost_equal(Mmult.p, expected, decimal=4)


def test_prod_cpms2(setup_prod_cms):

    values = ['S', 'F']
    v1 = variable.Variable(name='v1', B=[{0}, {1}, {0, 1}], values=values)
    v2 = variable.Variable(name='v2', B=[{0}, {1}, {0, 1}], values=values)
    v3 = variable.Variable(name='v3', B=[{0}, {1}, {0, 1}], values=values)

    M = {}
    M[1] = cpm.Cpm(variables=[v1],
                   no_child=1,
                   C = np.array([1, 2]).T,
                   p = np.array([0.8390, 0.1610]).T)

    M[2] = cpm.Cpm(variables=[v2],
                   no_child=1,
                   C = np.array([1, 2]).T,
                   p = np.array([0.9417, 0.0583]).T)

    M[3] = cpm.Cpm(variables=[v3],
                   no_child=1,
                   C = np.array([1, 2]).T,
                   p = np.array([0.99948, 0.0052]).T)

    Mmult = cpm.prod_cpms([M[k] for k in [1, 2]])

    expected = np.array([[1, 1], [2, 1], [1, 2], [2, 2]])

    np.testing.assert_array_equal(Mmult.C, expected)

    np.testing.assert_array_equal(Mmult.C, expected)

    expected = np.array([[0.7901, 0.1517, 0.0489, 0.0094]]).T
    np.testing.assert_array_almost_equal(Mmult.p, expected, decimal=4)
    assert [x.name for x in Mmult.variables] == ['v1', 'v2']


def test_prod_cpms3(setup_prod_cms):

    values = ['S', 'F']
    v1 = variable.Variable(name='v1', B=[{0}, {1}, {0, 1}], values=values)
    v2 = variable.Variable(name='v2', B=[{0}, {1}, {0, 1}], values=values)
    v3 = variable.Variable(name='v3', B=[{0}, {1}, {0, 1}], values=values)

    M = {}
    M['e1'] = cpm.Cpm(variables=[v1],
                   no_child=1,
                   C = np.array([1, 2]).T - 1,
                   p = np.array([0.83896, 0.16103]).T)

    M['e2'] = cpm.Cpm(variables=[v2],
                   no_child=1,
                   C = np.array([1, 2]).T - 1,
                   p = np.array([0.94173, 0.05827]).T)

    M['e3'] = cpm.Cpm(variables=[v3],
                   no_child=1,
                   C = np.array([1, 2]).T - 1,
                   p = np.array([0.99476, 0.00524]).T)

    Mmult = cpm.prod_cpms(M)

    expected = np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1], [1, 1, 2], [2, 1, 2], [1, 2, 2], [2, 2, 2]]) - 1
    np.testing.assert_array_equal(Mmult.C, expected)

    expected = np.array([[0.7859, 0.1509, 0.0486, 0.0093, 0.0041, 0.0008, 0.0003, 0.000]]).T
    np.testing.assert_array_almost_equal(Mmult.p, expected, decimal=4)

    assert [x.name for x in Mmult.variables] == ['v1', 'v2', 'v3']


def test_get_variables_from_cpms():

    values = ['S', 'F']
    v1 = variable.Variable(name='v1', B=[{0}, {1}, {0, 1}], values=values)
    v2 = variable.Variable(name='v2', B=[{0}, {1}, {0, 1}], values=values)
    v3 = variable.Variable(name='v3', B=[{0}, {1}, {0, 1}], values=values)

    m1 = cpm.Cpm(variables=[v1, v2],
             C = np.array([[1, 1], [2, 2]]) - 1,
             p = np.array([[1.0, 1.0]]).T,
             no_child=1,
             )

    m2 = cpm.Cpm(variables=[v2, v3],
             C = np.array([[1, 1], [2, 2]]) - 1,
             p = np.array([[1.0, 1.0]]).T,
             no_child=1,
             )

    m3 = cpm.Cpm(variables=[v3],
                   no_child=1,
                   C = np.array([1, 2]).T - 1,
                   p = np.array([0.99476, 0.00524]).T)

    M = [m1, m2, m3]

    [res] = cpm.get_variables_from_cpms(M, ['v1'])
    assert res.name == 'v1'

    res = cpm.get_variables_from_cpms(M, ['v1', 'v3', 'v2'])
    assert [x.name for x in res] == ['v1', 'v3', 'v2']

    with pytest.raises(AssertionError):
        cpm.get_variables_from_cpms(M, ['v4', 'v1', 'v2'])


def test_get_variables_from_cpms2(setup_condition):
    Mx_ = setup_condition

    v2, v3, v5, v4 = Mx_.get_variables(['v2', 'v3', 'v5', 'v4'])

    C = np.array([[2, 3, 3, 2],
                 [1, 1, 3, 1],
                 [1, 2, 1, 1],
                 [2, 2, 2, 1]]) - 1
    p = np.array([1, 1, 1, 1, ])
    Mx = cpm.Cpm(variables=[v5, v2, v3, v4], no_child=1, C = C, p = p.T)
    condVars = ['v2', 'v3']

    condVars = cpm.get_variables_from_cpms([Mx], condVars)
    assert [x.name for x in condVars] == ['v2', 'v3']


