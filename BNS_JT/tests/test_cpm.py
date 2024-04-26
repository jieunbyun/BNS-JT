import numpy as np
import sys, os
import pytest
import pdb
import copy
import random
from pathlib import Path

np.set_printoptions(precision=3)
#pd.set_option.display_precision = 3

from BNS_JT import cpm, variable, config, trans, gen_bnb, betasumrat

HOME = Path(__file__).parent


@pytest.fixture
def dict_cpm():
    ''' Use instance of Variables in the variables'''
    A1 = variable.Variable(**{'name': 'A1',
                              'values': ['s', 'f']})
    A2 = variable.Variable(**{'name': 'A2',
                              'values': ['s', 'f']})
    A3 = variable.Variable(**{'name': 'A3',
                              'values': ['s', 'f']})

    return {'variables': [A3, A2, A1],
            'no_child': 1,
            'C': np.array([[2, 2, 3], [2, 1, 2], [1, 1, 1]]) - 1,
            'p': [1, 1, 1]}


@pytest.fixture()
def var_A1_to_A5():

    A1 = variable.Variable(name='A1', values=['Mild', 'Severe'])
    A2 = variable.Variable(name='A2', values=['Survive', 'Fail'])
    A3 = variable.Variable(name='A3', values=['Survive', 'Fail'])
    A4 = variable.Variable(name='A4', values=['Survive', 'Fail'])
    A5 = variable.Variable(name='A5', values=['Survive', 'Fail'])

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

def test_init4():
    with pytest.raises(AssertionError):
        a = cpm.Cpm(variables=['1'], no_child=1)


def test_init5():

    a = cpm.Cpm([], 0)

    with pytest.raises(AssertionError):
        a.variables = ['1']


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

    rowIdx = cpm.argsort(list(map(tuple, C[:, ::-1])))

    try:
        Ms_p = M.p[rowIdx]
    except IndexError:
        Ms_p = M.p

    try:
        Ms_Cs = M.Cs[rowIdx,:]
    except IndexError:
        Ms_Cs = M.Cs

    try:
        Ms_q = M.q[rowIdx]
    except IndexError:
        Ms_q = M.q

    try:
        Ms_ps = M.ps[rowIdx]
    except IndexError:
        Ms_ps = M.ps

    Ms = cpm.Cpm(C=M.C[rowIdx, :],
             p=Ms_p,
             Cs = Ms_Cs,
             q=Ms_q,
             ps=Ms_ps,
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

    A1 = variable.Variable(**{'name':'A1',
                   'values': ['s', 'f']})
    A2 = variable.Variable(**{'name': 'A2',
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

    A2 = variable.Variable(**{'name': 'A2', 'values': ['s', 'f']})
    A3 = variable.Variable(**{'name': 'A3', 'values': ['s', 'f']})
    A4 = variable.Variable(**{'name': 'A4', 'values': ['s', 'f']})
    A5 = variable.Variable(**{'name': 'A5', 'values': ['s', 'f']})
    A6 = variable.Variable(**{'name': 'A6', 'values': ['s', 'f']})
    A8 = variable.Variable(**{'name': 'A8', 'values': ['s', 'f']})

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

    A1 = variable.Variable(**{'name': 'A1', 'values': ['s', 'f']})
    A2 = variable.Variable(**{'name': 'A2', 'values': ['s', 'f']})
    A3 = variable.Variable(**{'name': 'A3', 'values': ['s', 'f']})
    A4 = variable.Variable(**{'name': 'A4', 'values': ['s', 'f']})
    A5 = variable.Variable(**{'name': 'A5', 'values': ['s', 'f']})
    A6 = variable.Variable(**{'name': 'A6', 'values': ['s', 'f']})
    A7 = variable.Variable(**{'name': 'A7', 'values': ['s', 'f']})

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

    v[1] = variable.Variable(name='1', values=['Mild', 'Severe'])
    v[2] = variable.Variable(name='2', values=['Survive', 'Fail'])
    v[3] = variable.Variable(name='3', values=['Survive', 'Fail'])
    v[4] = variable.Variable(name='4', values=['Survive', 'Fail'])
    v[5] = variable.Variable(name='5', values=['Survive', 'Fail'])

    M[1] = cpm.Cpm(variables=[v[1]], no_child=1, C = np.array([[1, 2]]).T - 1, p = np.array([0.9, 0.1]).T)
    M[2] = cpm.Cpm(variables=[v[2], v[1]], no_child=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]) - 1, p = np.array([0.99, 0.01, 0.9, 0.1]).T)
    M[3] = cpm.Cpm(variables=[v[3], v[1]], no_child=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]) - 1, p = np.array([0.95, 0.05, 0.85, 0.15]).T)
    M[4] = cpm.Cpm(variables=[v[4], v[1]], no_child=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]) - 1, p = np.array([0.99, 0.01, 0.9, 0.1]).T)
    M[5] = cpm.Cpm(variables=[v[5], v[2], v[3], v[4]], no_child=1, C = np.array([[2, 3, 3, 2], [1, 1, 3, 1], [1, 2, 1, 1], [2, 2, 2, 1]]) - 1, p = np.array([1, 1, 1, 1]).T)

    return M, v

@pytest.fixture
def setup_iscompatible_Bfly():

    M = {}
    v = {}

    v[1] = variable.Variable(name='1', values=['Mild', 'Severe'], B_flag = 'fly')
    v[2] = variable.Variable(name='2', values=['Survive', 'Fail'], B_flag = 'fly')
    v[3] = variable.Variable(name='3', values=['Survive', 'Fail'], B_flag = 'fly')
    v[4] = variable.Variable(name='4', values=['Survive', 'Fail'], B_flag = 'fly')
    v[5] = variable.Variable(name='5', values=['Survive', 'Fail'], B_flag = 'fly')

    M[1] = cpm.Cpm(variables=[v[1]], no_child=1, C = np.array([[1, 2]]).T - 1, p = np.array([0.9, 0.1]).T)
    M[2] = cpm.Cpm(variables=[v[2], v[1]], no_child=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]) - 1, p = np.array([0.99, 0.01, 0.9, 0.1]).T)
    M[3] = cpm.Cpm(variables=[v[3], v[1]], no_child=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]) - 1, p = np.array([0.95, 0.05, 0.85, 0.15]).T)
    M[4] = cpm.Cpm(variables=[v[4], v[1]], no_child=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]) - 1, p = np.array([0.99, 0.01, 0.9, 0.1]).T)
    M[5] = cpm.Cpm(variables=[v[5], v[2], v[3], v[4]], no_child=1, C = np.array([[2, 3, 3, 2], [1, 1, 3, 1], [1, 2, 1, 1], [2, 2, 2, 1]]) - 1, p = np.array([1, 1, 1, 1]).T)

    return M, v

@pytest.fixture
def setup_iscompatible_Bfly2():

    M = {}
    v = {}

    v[1] = variable.Variable(name='1', values=['Mild', 'mid', 'Severe'], B_flag = 'fly')
    v[2] = variable.Variable(name='2', values=['Survive', 'Fail'], B_flag = 'fly')
    v[3] = variable.Variable(name='3', values=['Survive', 'Fail'], B_flag = 'fly')
    v[4] = variable.Variable(name='4', values=['Survive', 'Fail'], B_flag = 'fly')
    v[5] = variable.Variable(name='5', values=['Survive', 'Fail'], B_flag = 'fly')

    M[1] = cpm.Cpm(variables=[v[1]], no_child=1, C = np.array([[1, 2, 3]]).T - 1, p = np.array([0.7, 0.2, 0.1]).T)
    M[2] = cpm.Cpm(variables=[v[2], v[1]], no_child=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2], [1,3], [2,3]]) - 1, p = np.array([0.99, 0.01, 0.95, 0.05, 0.9, 0.1]).T)
    M[3] = cpm.Cpm(variables=[v[3], v[1]], no_child=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2], [1,3], [2,3]]) - 1, p = np.array([0.95, 0.05, 0.90, 0.10, 0.85, 0.15]).T)
    M[4] = cpm.Cpm(variables=[v[4], v[1]], no_child=1, C = np.array([[1, 1], [2, 1], [1, 2], [2, 2], [1,3], [2,3]]) - 1, p = np.array([0.99, 0.01, 0.95, 0.05, 0.9, 0.1]).T)
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

def test_iscompatible1f(setup_iscompatible_Bfly):

    M, v = setup_iscompatible_Bfly

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

def test_iscompatible1sf(setup_iscompatible_Bfly):
    # using string for checkVars, checkStates
    M, v = setup_iscompatible_Bfly

    # M[2]
    C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]) - 1 #M[2].C
    variables = [v[2], v[1]]
    checkVars = ['1']
    checkStates = ['Mild']
    result = cpm.iscompatible(C, variables, checkVars, checkStates)
    expected = np.array([1, 1, 0, 0])
    np.testing.assert_array_equal(expected, result)

def test_iscompatible1f2(setup_iscompatible_Bfly2):

    M, v = setup_iscompatible_Bfly2

    # M[2]
    C = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]]) #M[2].C
    variables = [v[2], v[1]]
    checkVars = [v[1]]
    checkStates = [3]
    result = cpm.iscompatible(C, variables, checkVars, checkStates)
    expected = np.array([1, 1, 1, 1, 0, 0])
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

    A1 = variable.Variable(name='A1', values=['Mild', 'Severe'])
    A2 = variable.Variable(name='A2', values=['Survive', 'Fail'])
    A3 = variable.Variable(name='A3', values=['Survive', 'Fail'])
    A4 = variable.Variable(name='A4', values=['Survive', 'Fail'])
    A5 = variable.Variable(name='A5', values=['Survive', 'Fail'])

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
    np.testing.assert_array_equal(compatCheck, expected)#

def test_iscompatible5(setup_iscompatible_Bfly2):

    M, v = setup_iscompatible_Bfly2

    st0 = v[1].B_fly({0})
    st1 = v[1].B_fly({1})
    st2 = v[1].B_fly({2})
    st3 = v[1].B_fly({0,1})
    st4 = v[1].B_fly({0,2})
    st5 = v[1].B_fly({1,2})
    st6 = v[1].B_fly({0,1,2})

    assert [st0, st1, st2, st3, st4, st5, st6] == [0, 1, 2, 3, 4, 5, 6]


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

def test_iscompatibleCpm1sf(var_A1_to_A5):

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

    _, d_vars_arc, _, _ = setup_bridge
    vars_arc = copy.deepcopy(d_vars_arc)

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

def test_iscompatibleCpm5(setup_iscompatible_Bfly):

    # M[5]
    M, _ = setup_iscompatible_Bfly
    rowIndex = [0]  # 1 -> 0
    M_sys_select = M[5].get_subset(rowIndex)

    result = M[4].iscompatible(M_sys_select, flag=True)
    expected = np.array([0, 1, 0, 1])
    np.testing.assert_array_equal(result, expected)

def test_iscompatibleCpm6(setup_iscompatible_Bfly):

    # M[5]
    M, _ = setup_iscompatible_Bfly
    rowIndex = [0]  # 1 -> 0
    M_sys_select = M[5].get_subset(rowIndex)

    result = M[4].iscompatible(M_sys_select, flag=True)
    expected = np.array([0, 1, 0, 1])
    np.testing.assert_array_equal(result, expected)

def test_iscompatible():
    C = np.array([[0,1], [0,1], [2,1], [1,1]])
    varis = {'e1': variable.Variable('e1', [0,1,2]), 'sys': variable.Variable('sys', [0,1])}
    variables = [varis['e1'], varis['sys']]
    check_vars = ['sys', 'e1']
    check_states = [0, 3]
    iscmp = cpm.iscompatible( C, variables, check_vars, check_states )

    np.testing.assert_array_equal(iscmp, np.array([False, False, False, False]))


@pytest.fixture
def setup_product():

    X1 = variable.Variable(name='X1', values=['Mild', 'Severe'])
    X2 = variable.Variable(name='X2', values=['Survive', 'Fail'])
    X3 = variable.Variable(name='X3', values=['Survive', 'Fail'])
    X4 = variable.Variable(name='X4', values=['Survive', 'Fail'])
    X5 = variable.Variable(name='X5', values=['Survive', 'Fail'])

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

        #FIXME
        """if any(M1.sample_idx):
            sampleInd1 = M1.sample_idx[i]
        else:
            sampleInd1 = []"""
        sampleInd1 = []

        #if isinstance(commonVars, list):
        #    commonVars = np.array(commonVars)

        #if isinstance(c1_, list):
        #    c1_ = np.array(c1_)
        [M2_] = cpm.condition([M2], commonVars, c1_)

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
    v1 = variable.Variable(name='v1', values=['Mild', 'Severe'])
    v2 = variable.Variable(name='v2', values=['Survive', 'Fail'])
    v3 = variable.Variable(name='v3', values=['Survive', 'Fail'])
    v4 = variable.Variable(name='v4', values=['Survive', 'Fail'])
    v5 = variable.Variable(name='v5', values=['Survive', 'Fail'])

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
                            #B=B,
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


@pytest.mark.skip('FIXME')
def test_sum6(setup_bridge):

    d_cpms_arc, d_vars_arc, _, _ = setup_bridge
    cpms_arc = copy.deepcopy(d_cpms_arc)
    vars_arc = copy.deepcopy(d_vars_arc)

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
    #np.testing.assert_array_equal(cpm_mult.C, expected_C)
    np.testing.assert_array_almost_equal(cpm_mult.p, np.array([[0.8390, 0.8390, 0.8390, 0.1610, 0.1610]]).T, decimal=4)

    a = cpm_mult.sum(['e1'])
    assert [x.name for x in a.variables] == ['od1', 'e2', 'e3', 'e4', 'e5', 'e6']
    expected_C = np.array([[2, 2, 1, 3, 3, 3],
                           [3, 2, 2, 3, 3, 3],
                           [1, 1, 3, 3, 3, 3],
                           [3, 2, 3, 3, 3, 3]]) - 1
    #np.testing.assert_array_equal(a.C, expected_C)
    np.testing.assert_array_almost_equal(a.p, np.array([[0.8390, 0.8390, 1.0, 0.1610]]).T, decimal=4)


@pytest.fixture
def setup_mcs_product():

    v1 = variable.Variable(name='v1', values=['Mild', 'Severe'])
    v2 = variable.Variable(name='v2', values=['Survive', 'Fail'])
    v3 = variable.Variable(name='v3', values=['Survive', 'Fail'])
    v4 = variable.Variable(name='v4', values=['Survive', 'Fail'])
    v5 = variable.Variable(name='v5', values=['Survive', 'Fail'])

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

def test_single_sample3(setup_mcs_product):
    cpms = setup_mcs_product
    cpms = {k:cpms[k] for k in [1, 2, 3]}
    v2, v1 = cpms[2].get_variables(['v2', 'v1'])
    v3 = cpms[3].get_variables('v3')

    sampleOrder = [0, 1, 2]
    sampleVars = [v1, v2, v3]
    varAdditionOrder = [0, 1, 2]
    sampleInd = [1]

    sample, sampleProb = cpm.single_sample(cpms, sampleOrder, sampleVars, varAdditionOrder, sampleInd, is_scalar=False)

    if (sample == [1, 1, 1]).all():
        np.testing.assert_array_almost_equal(sampleProb, [[0.9],[0.99],[0.95]], decimal=3)
    elif (sample == [2, 1, 1]).all():
        np.testing.assert_array_almost_equal(sampleProb, [[0.1],[0.9],[0.85]], decimal=3)

def test_mcs_product1(setup_mcs_product):

    cpms = setup_mcs_product
    cpms = {k:cpms[k] for k in [1, 2, 3]}

    nSample = 10
    Mcs = cpm.mcs_product(cpms, nSample)

    assert [x.name for x in Mcs.variables]==['v3', 'v2', 'v1']

    assert Mcs.Cs.shape== (10, 3)
    assert Mcs.q.shape== (10, 1)
    assert Mcs.sample_idx.shape== (10, 1)

    irow = np.where((Mcs.Cs == (0, 0, 0)).all(axis=1))[0]
    try:
        np.testing.assert_array_almost_equal(Mcs.q[irow], 0.8464*np.ones((len(irow), 1)), decimal=4)
    except AssertionError:
        print(f'{Mcs.q[irow]} vs 0.8464')

    irow = np.where((Mcs.Cs == (0, 0, 1)).all(axis=1))[0]
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
    assert Mcs.Cs.shape== (10, 5)
    assert Mcs.q.shape== (10, 1)
    assert Mcs.sample_idx.shape== (10, 1)

    irow = np.where((Mcs.Cs == (0, 0, 0, 0, 0)).all(axis=1))[0]
    try:
        np.testing.assert_array_almost_equal(Mcs.q[irow], 0.8380*np.ones((len(irow), 1)), decimal=4)
    except AssertionError:
        print(f'{Mcs.q[irow]} vs 0.8380')

    irow = np.where((Mcs.Cs == (0, 0, 0, 0, 1)).all(axis=1))[0]
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

    assert Mcs.Cs.shape== (10, 5)
    assert Mcs.q.shape== (10, 1)
    assert Mcs.sample_idx.shape== (10, 1)

    irow = np.where((Mcs.Cs == (0, 0, 0, 0, 0)).all(axis=1))[0]
    try:
        np.testing.assert_array_almost_equal(Mcs.q[irow], 0.8380*np.ones((len(irow), 1)), decimal=4)
    except AssertionError:
        print(f'{Mcs.q[irow]} vs 0.8380')

    irow = np.where((Mcs.Cs == (0, 0, 0, 0, 1)).all(axis=1))[0]
    try:
        np.testing.assert_array_almost_equal(Mcs.q[irow], 0.0688*np.ones((len(irow), 1)), decimal=3)
    except AssertionError:
        print(f'{Mcs.q[irow]} vs 0.0688')

def test_mcs_product2ds(setup_mcs_product):

    cpms_ = setup_mcs_product

    nSample = 10
    Mcs = cpm.mcs_product(cpms_, nSample)

    assert [x.name for x in Mcs.variables]==['v5', 'v4', 'v3', 'v2', 'v1']

    assert Mcs.Cs.shape== (10, 5)
    assert Mcs.q.shape== (10, 1)
    assert Mcs.sample_idx.shape== (10, 1)

    irow = np.where((Mcs.Cs == (0,0,0,0,0)).all(axis=1))[0]
    try:
        np.testing.assert_array_almost_equal(Mcs.q[irow], 0.8380*np.ones((len(irow), 1)), decimal=4)
    except AssertionError:
        print(f'{Mcs.q[irow]} vs 0.8380')

    irow = np.where((Mcs.Cs == (0, 0, 0, 0, 1)).all(axis=1))[0]
    try:
        np.testing.assert_array_almost_equal(Mcs.q[irow], 0.0688*np.ones((len(irow), 1)), decimal=3)
    except AssertionError:
        print(f'{Mcs.q[irow]} vs 0.0688')

def test_mcs_product2ds2(setup_mcs_product):

    cpms_ = setup_mcs_product

    nSample = 10
    Mcs = cpm.mcs_product(cpms_, nSample, is_scalar=False)

    assert [x.name for x in Mcs.variables]==['v5', 'v4', 'v3', 'v2', 'v1']

    assert Mcs.Cs.shape== (10, 5)
    assert Mcs.q.shape== (10, 5)
    assert Mcs.sample_idx.shape== (10, 1)

    irow = np.where((Mcs.Cs == (0,0,0,0,0)).all(axis=1))[0]
    try:
        np.testing.assert_array_almost_equal(Mcs.q[irow], np.array([[1, 0.99, 0.95, 0.99, 0.9]]), decimal=2)
    except AssertionError:
        print(f'{Mcs.q[irow]} vs [1, 0.99, 0.95, 0.99, 0.9]')

    irow = np.where((Mcs.Cs == (0, 0, 0, 0, 1)).all(axis=1))[0]
    try:
        np.testing.assert_array_almost_equal(Mcs.q[irow], np.array([[1, 0.9, 0.85, 0.9, 0.1]]), decimal=2)
    except AssertionError:
        print(f'{Mcs.q[irow]} vs [1, 0.9, 0.85, 0.9, 0.1]')


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

    [M]= cpm.condition(cpms[3], condVars, condStates)
    np.testing.assert_array_equal(M.C, np.array([[1, 1], [2, 1]])-1)
    assert M.q.any() == False
    assert M.sample_idx.any() == False


@pytest.fixture
def setup_prod_cms():
    v1 = variable.Variable(name='v1', values=['Sunny', 'Cloudy', 'Rainy'])
    v2 = variable.Variable(name='v2', values=['Good', 'Bad'])
    v3 = variable.Variable(name='v3', values=['Below 0', 'Above 0'])

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
    v1 = variable.Variable(name='v1', values=values)
    v2 = variable.Variable(name='v2', values=values)
    v3 = variable.Variable(name='v3', values=values)

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
    v1 = variable.Variable(name='v1', values=values)
    v2 = variable.Variable(name='v2', values=values)
    v3 = variable.Variable(name='v3', values=values)

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
    v1 = variable.Variable(name='v1', values=values)
    v2 = variable.Variable(name='v2', values=values)
    v3 = variable.Variable(name='v3', values=values)

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


def test_get_prob(setup_inference):

    d_cpms, d_varis, var_elim_order, arcs = setup_inference

    cpms = copy.deepcopy(d_cpms)
    varis = copy.deepcopy(d_varis)

    ## Repeat inferences again using new functions -- the results must be the same.
    # Probability of delay and disconnection
    M = [cpms[k] for k in varis.keys()]
    M_VE2 = cpm.variable_elim(M, var_elim_order)

    # Prob. of failure
    prob_f1 = cpm.get_prob(M_VE2, [varis['sys']], [0])
    prob_f2 = cpm.get_prob(M_VE2, [varis['sys']], ['f'])
    assert prob_f1 == prob_f2

    with pytest.raises(AssertionError):
        prob_f2 = cpm.get_prob(M_VE2, [varis['sys']], ['d'])


def test_variable_elim(setup_bridge):

    d_cpms_arc, d_vars_arc, arcs, _ = setup_bridge
    cpms_arc = copy.deepcopy(d_cpms_arc)
    vars_arc = copy.deepcopy(d_vars_arc)

    cpms = [cpms_arc[k] for k in ['od1'] + list(arcs.keys())]
    var_elim_order = [vars_arc[i] for i in arcs.keys()]
    result = cpm.variable_elim(cpms, var_elim_order)

    np.testing.assert_array_almost_equal(result.C, np.array([[0, 1, 2]]).T)
    np.testing.assert_array_almost_equal(result.p, np.array([[0.009, 0.048, 0.942]]).T, decimal=3)


def test_prod_cpm_sys_and_comps():

    cfg = config.Config(HOME.joinpath('../demos/routine/config.json'))

    st_br_to_cs = {'f':0, 's':1, 'u': 2}

    od_pair = cfg.infra['ODs']['od1']

    probs = {'e1': {0: 0.01, 1:0.99}, 'e2': {0:0.02, 1:0.98}, 'e3': {0:0.03, 1:0.97}, 'e4': {0:0.04, 1:0.96}, 'e5': {0:0.05, 1:0.95}}

    varis = {}
    cpms = {}
    for k in cfg.infra['edges'].keys():
        varis[k] = variable.Variable(name=k, values=['f', 's'])

        cpms[k] = cpm.Cpm(variables = [varis[k]], no_child=1,
                          C = np.array([0, 1]).T, p = [probs[k][0], probs[k][1]])

    #sys_fun = lambda comps_st : conn(comps_st, od_pair, arcs)
    sys_fun = trans.sys_fun_wrap(od_pair, cfg.infra['edges'], varis)

    brs, _, _, _ = gen_bnb.proposed_branch_and_bound_using_probs(
            sys_fun, varis, probs, max_sf=1000, output_path=HOME, key='routine')

    csys_by_od, varis_by_od = gen_bnb.get_csys_from_brs(brs, varis, st_br_to_cs)

    varis['sys'] = variable.Variable(name='sys', values=['f', 's'])
    cpms['sys'] = cpm.Cpm(variables = [varis[k] for k in ['sys'] + list(cfg.infra['edges'].keys())],
                          no_child = 1,
                          C = csys_by_od.copy(),
                          p = np.ones(csys_by_od.shape[0]))
    cpms_comps = {k: cpms[k] for k in cfg.infra['edges'].keys()}

    cpms_new = cpm.prod_Msys_and_Mcomps(cpms['sys'], list(cpms_comps.values()))

    expected_C = np.array([[1, 1, 2, 2, 1, 2],
                           [1, 1, 1, 2, 0, 1],
                           [1, 0, 1, 2, 2, 1],
                           [0, 1, 2, 2, 0, 0],
                           [0, 1, 0, 0, 0, 1],
                           [1, 1, 0, 1, 0, 1],
                           [0, 0, 1, 1, 0, 0],
                           [1, 0, 1, 1, 1, 0],
                           [0, 0, 0, 2, 2, 2],
                           [0, 0, 1, 0, 2, 0]])

    expected_p = np.array([[9.50400e-01], [3.68676e-02], [9.31000e-03],
                           [1.98000e-03], [2.25720e-05], [7.29828e-04],
                           [1.90120e-05], [4.56288e-04], [2.00000e-04],
                           [1.47000e-05]])

    np.testing.assert_array_equal(cpms_new.C, expected_C)
    np.testing.assert_array_almost_equal(cpms_new.p, expected_p, decimal=5)
    assert cpms_new.no_child == 6
    assert len(cpms_new.variables) == 6

    p_f = cpm.get_prob(cpms_new, ['sys'], [0])
    p_s = cpm.get_prob(cpms_new, ['sys'], [1])

    assert p_f == pytest.approx(0.002236, rel=1.0e-3)
    assert p_s == pytest.approx(0.997763, rel=1.0e-3)

    # FIXME: expected value required
    p_x = cpm.get_prob(cpms_new, ['sys', 'e1'], [0, 1])
    assert p_x == pytest.approx(0.002002, rel=1.0e-3)

    p_f = cpm.get_prob(cpms_new, [varis['sys']], [0])
    assert p_f == pytest.approx(0.002236, rel=1.0e-3)


@pytest.fixture()
def setup_hybrid(): 
    varis, cpms = {}, {}

    varis['haz'] = variable.Variable(name='haz', values=['mild', 'severe'])
    cpms['haz'] = cpm.Cpm(variables=[varis['haz']], no_child=1, C=np.array([0,1]), p=[0.7, 0.3])

    varis['x0'] = variable.Variable(name='x0', values=['fail', 'surv'])
    cpms['x0'] = cpm.Cpm(variables=[varis['x0'], varis['haz']], no_child=1, C=np.array([[0,0],[1,0],[0,1],[1,1]]), p=[0.1,0.9,0.2,0.8])
    varis['x1'] = variable.Variable(name='x1', values=['fail', 'surv'])
    cpms['x1'] = cpm.Cpm(variables=[varis['x1'], varis['haz']], no_child=1, C=np.array([[0,0],[1,0],[0,1],[1,1]]), p=[0.3,0.7,0.4,0.6])

    varis['sys'] = variable.Variable(name='sys', values=['fail', 'surv'])
    cpms['sys'] = cpm.Cpm(variables=[varis['sys'], varis['x0'], varis['x1']], no_child=1, C=np.array([[0,0,0],[1,1,1]]), p=np.array([1.0,1.0])) # incomplete C (i.e. C does not include all samples)

    # samples
    cpms['haz'].Cs, cpms['haz'].q, cpms['haz'].sample_idx = np.array([0,0,0,1,0]), [0.7,0.7,0.7,0.3,0.7], [0,1,2,3,4]
    cpms['x0'].Cs, cpms['x0'].q, cpms['x0'].sample_idx = np.array([[0,0],[1,0],[1,0],[0,1],[1,0]]), [0.1,0.9,0.9,0.2,0.9], [0,1,2,3,4]
    cpms['x1'].Cs, cpms['x1'].q, cpms['x1'].sample_idx = np.array([[1,0],[0,0],[0,0],[1,1],[0,0]]), [0.7,0.3,0.3,0.6,0.3], [0,1,2,3,4]
    cpms['sys'].Cs, cpms['sys'].q, cpms['sys'].sample_idx = np.array([[0,0,1],[1,1,0],[1,1,0],[0,0,1],[1,1,0]]), [1,1,1,1,1], [0,1,2,3,4]

    return varis, cpms

def test_get_subset4(setup_hybrid):

    _,cpms = setup_hybrid

    rowIndex = [0]  
    result = cpms['haz'].get_subset(rowIndex, isC=False)

    np.testing.assert_array_equal(result.Cs, np.array([[0]]))
    np.testing.assert_array_equal(result.q, [[0.7]])

def test_get_subset5(setup_hybrid):

    _,cpms = setup_hybrid

    rowIndex = [0,1,2]  
    result = cpms['haz'].get_subset(rowIndex, flag=False, isC=False)

    np.testing.assert_array_equal(result.Cs, np.array([[1],[0]]))
    np.testing.assert_array_equal(result.q, [[0.3],[0.7]])

def test_get_variables_from_cpms(setup_hybrid):

    varis, cpms = setup_hybrid
    vars_ = cpm.get_variables_from_cpms(cpms,['x0','x1'])

    assert all(isinstance(v, variable.Variable) for v in vars_)
    assert vars_[0].name == 'x0'
    assert vars_[1].name == 'x1'

def test_condition6( setup_hybrid ):

    varis, cpms = setup_hybrid
    Mc = cpm.condition(cpms, ['haz'], [0])

    np.testing.assert_array_almost_equal(Mc['haz'].C, np.array([[0]]))
    np.testing.assert_array_almost_equal(Mc['haz'].ps, cpms['haz'].q * 0.7)
    np.testing.assert_array_almost_equal(Mc['x0'].ps, np.array([[0.1],[0.9],[0.9],[0.1],[0.9]]))
    np.testing.assert_array_almost_equal(Mc['x1'].ps, np.array([[0.7],[0.3],[0.3],[0.7],[0.3]]))
    np.testing.assert_array_almost_equal(Mc['sys'].ps, np.array([[1.0],[1.0],[1.0],[1.0],[1.0]]))

def test_product4( setup_hybrid ):

    varis, cpms = setup_hybrid
    Mp = cpms['haz'].product( cpms['x0'] )

    assert Mp.variables[0].name == 'haz' and Mp.variables[1].name == 'x0'
    np.testing.assert_array_almost_equal(Mp.C, np.array([[0,0], [1,0], [0,1], [1,1]]))
    np.testing.assert_array_almost_equal(Mp.p, np.array([[0.07], [0.06], [0.63], [0.24]]))
    np.testing.assert_array_almost_equal(Mp.Cs, np.array([[0,0],[0,1],[0,1],[1,0],[0,1]]))
    np.testing.assert_array_almost_equal(Mp.q, np.array([[0.07], [0.63], [0.63], [0.06], [0.63]]))
    np.testing.assert_array_almost_equal(Mp.q, Mp.ps)
    np.testing.assert_array_almost_equal(Mp.sample_idx, np.array([[0],[1],[2],[3],[4]]))

def test_product5( setup_hybrid ):
    varis, cpms = setup_hybrid
    Mc = cpm.condition(cpms, ['haz'], [0])
    Mp0 = Mc['haz'].product( Mc['x0'] )
    Mp1 = Mp0.product(Mc['x1'])

    assert Mp0.variables[0].name == 'haz' and Mp0.variables[1].name == 'x0'
    np.testing.assert_array_almost_equal(Mp0.C, np.array([[0,0], [0,1]]))
    np.testing.assert_array_almost_equal(Mp0.p, np.array([[0.07], [0.63]]))
    np.testing.assert_array_almost_equal(Mp0.Cs, np.array([[0,0],[0,1],[0,1],[1,0],[0,1]]))
    np.testing.assert_array_almost_equal(Mp0.q, np.array([[0.07], [0.63], [0.63], [0.06], [0.63]]))
    np.testing.assert_array_almost_equal(Mp0.ps, np.array([[0.049], [0.441], [0.441], [0.021], [0.441]]))
    np.testing.assert_array_almost_equal(Mp0.sample_idx, np.array([[0],[1],[2],[3],[4]]))

    assert Mp1.variables[0].name == 'haz' and Mp1.variables[1].name == 'x0' and Mp1.variables[2].name == 'x1'
    np.testing.assert_array_almost_equal(Mp1.C, np.array([[0,0,0], [0,1,0], [0,0,1], [0,1,1]]))
    np.testing.assert_array_almost_equal(Mp1.p, np.array([[0.021], [0.189], [0.049], [0.441]]))
    np.testing.assert_array_almost_equal(Mp1.Cs, np.array([[0,0,1],[0,1,0],[0,1,0],[1,0,1],[0,1,0]]))
    np.testing.assert_array_almost_equal(Mp1.q, np.array([[0.049], [0.189], [0.189], [0.036], [0.189]]))
    np.testing.assert_array_almost_equal(Mp1.ps, np.array([[0.0343], [0.1323], [0.1323], [0.015], [0.132]]), decimal=3)

def test_sum7(setup_hybrid):
    varis, cpms = setup_hybrid
    Mc = cpm.condition(cpms, ['haz'], [0])
    Mp0 = Mc['haz'].product( Mc['x0'] )
    Mp1 = Mp0.product(Mc['x1'])

    Mp0_s = Mp0.sum([varis['x0']])
    Mp1_s = Mp1.sum([varis['haz']])

    assert Mp0_s.variables[0].name == 'haz'
    np.testing.assert_array_almost_equal(Mp0_s.C, np.array([[0]]))
    np.testing.assert_array_almost_equal(Mp0_s.p, np.array([[0.7]]))
    np.testing.assert_array_almost_equal(Mp0_s.Cs, np.array([[0],[0],[0],[1],[0]]))
    np.testing.assert_array_almost_equal(Mp0_s.q, Mp0.q)
    np.testing.assert_array_almost_equal(Mp0_s.ps, Mp0.ps)
    np.testing.assert_array_almost_equal(Mp0_s.sample_idx, Mp0.sample_idx)

    assert Mp1_s.variables[0].name == 'x0' and Mp1_s.variables[1].name == 'x1'
    np.testing.assert_array_almost_equal(Mp1_s.C, np.array([[0,0], [1,0], [0,1], [1,1]]))
    np.testing.assert_array_almost_equal(Mp1_s.p, np.array([[0.021], [0.189], [0.049], [0.441]]))
    np.testing.assert_array_almost_equal(Mp1_s.Cs, np.array([[0,1],[1,0],[1,0],[0,1],[1,0]]))
    np.testing.assert_array_almost_equal(Mp1_s.q, Mp1.q)
    np.testing.assert_array_almost_equal(Mp1_s.ps, Mp1.ps)

def test_variable_elim1(setup_hybrid):

    varis, cpms = setup_hybrid

    var_elim_order = [varis['haz'], varis['x0'], varis['x1']]
    result = cpm.variable_elim(cpms, var_elim_order)

    np.testing.assert_array_almost_equal(result.C, np.array([[0], [1]]))
    np.testing.assert_array_almost_equal(result.p, np.array([[0.045, 0.585]]).T, decimal=3)
    np.testing.assert_array_almost_equal(result.Cs, np.array([[0,1,1,0,1]]).T, decimal=3)
    np.testing.assert_array_almost_equal(result.q, np.array([[0.049, 0.189, 0.189, 0.036, 0.189]]).T, decimal=3)
    np.testing.assert_array_almost_equal(result.ps, result.q)
    np.testing.assert_array_almost_equal(result.sample_idx, np.array([[0,1,2,3,4]]).T)

    prob, cov, cint = cpm.get_prob_and_cov( result, ['sys'], [0] )

    assert prob == pytest.approx(0.193, rel=1.0e-3)
    assert cov == pytest.approx(0.4200, rel=1.0e-3)

@pytest.fixture()
def setup_Msys_Mcomps():
    varis, cpms = {}, {}

    varis['x0'] = variable.Variable(name='x0', values=['fail', 'surv'])
    cpms['x0'] = cpm.Cpm( [varis['x0']], 1, np.array([0, 1]), p = [0.1, 0.9] )

    varis['x1'] = variable.Variable(name='x1', values=['fail', 'surv'])
    cpms['x1'] = cpm.Cpm( [varis['x1']], 1, np.array([0, 1]), p = [0.2, 0.8] )

    varis['x2'] = variable.Variable(name='x2', values=['fail', 'surv'])
    cpms['x2'] = cpm.Cpm( [varis['x2']], 1, np.array([0, 1]), p = [0.3, 0.7] )

    varis['sys'] = variable.Variable(name='sys', values=['fail', 'surv'])
    cpms['sys'] = cpm.Cpm(variables=[varis['sys'], varis['x0'], varis['x1'], varis['x2']], no_child=1, C=np.array([[0,0,2,2],[1,1,1,2],[1,1,0,1],[0,1,0,0]]), p=np.array([1.0,1.0,1.0,1.0], dtype=float))

    # Conditional model
    varis['haz'] = variable.Variable(name='haz', values=['mild', 'severe'])
    cpms['x0_c'] = cpm.Cpm( [varis['x0'], varis['haz']], 1, np.array([[0,0], [1,0]]), p = [0.05, 0.95] )
    cpms['x1_c'] = cpm.Cpm( [varis['x1'], varis['haz']], 1, np.array([[0,0], [1,0]]), p = [0.1, 0.9] )
    cpms['x2_c'] = cpm.Cpm( [varis['x2'], varis['haz']], 1, np.array([[0,0], [1,0]]), p = [0.2, 0.8] )

    return varis, cpms

def test_prod_cpm_sys_and_comps1(setup_Msys_Mcomps):

    varis, cpms = setup_Msys_Mcomps

    Msys = cpm.prod_Msys_and_Mcomps(cpms['sys'], [cpms['x0'], cpms['x1'], cpms['x2']])

    assert Msys.variables == [varis['sys'], varis['x0'], varis['x1'], varis['x2']]
    assert Msys.no_child == 4
    np.testing.assert_array_almost_equal(Msys.C, np.array([[0,0,2,2],[1,1,1,2],[1,1,0,1],[0,1,0,0]]))
    np.testing.assert_array_almost_equal(Msys.p, np.array([[0.1, 0.72, 0.126, 0.054]]).T, decimal=3)

def test_prod_cpm_sys_and_comps2(setup_Msys_Mcomps):

    varis, cpms = setup_Msys_Mcomps

    Msys = cpm.prod_Msys_and_Mcomps(cpms['sys'], [cpms['x0_c'], cpms['x1_c'], cpms['x2_c']])

    assert Msys.variables == [varis['sys'], varis['x0'], varis['x1'], varis['x2'], varis['haz']]
    assert Msys.no_child == 4
    np.testing.assert_array_almost_equal(Msys.C, np.array([[0,0,2,2,0],[1,1,1,2,0],[1,1,0,1,0],[0,1,0,0,0]]))
    np.testing.assert_array_almost_equal(Msys.p, np.array([[0.05, 0.855, 0.076, 0.019]]).T, decimal=3)
    
def test_prod_cpm_sys_and_comps3(setup_Msys_Mcomps):

    varis, cpms = setup_Msys_Mcomps

    Msys = cpm.prod_Msys_and_Mcomps(cpms['sys'], [cpms['x0'], cpms['x2']])

    assert Msys.variables == [varis['sys'], varis['x0'], varis['x2'], varis['x1']]
    assert Msys.no_child == 3
    np.testing.assert_array_almost_equal(Msys.C, np.array([[0,0,2,2],[1,1,2,1],[1,1,1,0],[0,1,0,0]]))
    np.testing.assert_array_almost_equal(Msys.p, np.array([[0.1, 0.9, 0.63, 0.27]]).T, decimal=3)

def test_prod_cpm_sys_and_comps4(setup_Msys_Mcomps):

    varis, cpms = setup_Msys_Mcomps

    Msys = cpm.prod_Msys_and_Mcomps(cpms['sys'], [cpms['x1_c'], cpms['x2_c']])

    assert Msys.variables == [varis['sys'], varis['x1'], varis['x2'], varis['x0'], varis['haz']]
    assert Msys.no_child == 3
    np.testing.assert_array_almost_equal(Msys.C, np.array([[0,2,2,0,0],[1,1,2,1,0],[1,0,1,1,0],[0,0,0,1,0]]))
    np.testing.assert_array_almost_equal(Msys.p, np.array([[1.0, 0.9, 0.08, 0.02]]).T, decimal=3)

def test_prod_cpm_sys_and_comps5(setup_Msys_Mcomps):

    varis, cpms = setup_Msys_Mcomps

    cpms['x1_c'].C = np.array([[0,1], [1,1]])
    with pytest.raises(AssertionError):
        Msys = cpm.prod_Msys_and_Mcomps(cpms['sys'], [cpms['x0_c'], cpms['x1_c']])

    varis['haz2'] = variable.Variable(name='haz2', values=['mild', 'severe'])
    cpms['x2_c'].variables = [varis['x2'], varis['haz2']]
    with pytest.raises(AssertionError):
        Msys = cpm.prod_Msys_and_Mcomps(cpms['sys'], [cpms['x0_c'], cpms['x2_c']])

def test_prod_cpm_sys_and_comps6(setup_Msys_Mcomps):

    varis, cpms = setup_Msys_Mcomps

    varis['x3'] = variable.Variable(name='x3', values=['fail', 'surv'])
    cpms['x3'] = cpm.Cpm( [varis['x3']], 1, np.array([0, 1]), p = [0.3, 0.7] )
    Msys = cpm.prod_Msys_and_Mcomps(cpms['sys'], [cpms['x0'], cpms['x2'], cpms['x3']])

    assert Msys.variables == [varis['sys'], varis['x0'], varis['x2'], varis['x1']]
    assert Msys.no_child == 3
    np.testing.assert_array_almost_equal(Msys.C, np.array([[0,0,2,2],[1,1,2,1],[1,1,1,0],[0,1,0,0]]))
    np.testing.assert_array_almost_equal(Msys.p, np.array([[0.1, 0.9, 0.63, 0.27]]).T, decimal=3)


def test_cal_Msys_by_cond_VE1(setup_hybrid):

    varis, cpms = setup_hybrid

    var_elim_order = ['haz', 'x0', 'x1']
    result = cpm.cal_Msys_by_cond_VE(cpms, varis, ['haz'], var_elim_order, 'sys')

    np.testing.assert_array_almost_equal(result.C, np.array([[0], [1]]))
    np.testing.assert_array_almost_equal(result.p, np.array([[0.045, 0.585]]).T, decimal=3)
    np.testing.assert_array_almost_equal(result.Cs, np.array([[0,1,1,0,1,0,1,1,0,1]]).T, decimal=3)
    np.testing.assert_array_almost_equal(result.q, np.array([[0.049, 0.189, 0.189, 0.036, 0.189, 0.049, 0.189, 0.189, 0.036, 0.189]]).T, decimal=3)
    np.testing.assert_array_almost_equal(result.ps, np.array([[0.0343, 0.1323, 0.1323, 0.0147, 0.1323, 0.0252, 0.0672, 0.0672, 0.0108, 0.0672]]).T, decimal=3)
    np.testing.assert_array_almost_equal(result.sample_idx, np.array([[0,1,2,3,4,0,1,2,3,4]]).T)

    prob, cov, cint = cpm.get_prob_and_cov( result, ['sys'], [0], flag = True, nsample_repeat = 5 )

    assert prob == pytest.approx(0.1873, rel=1.0e-3)
    assert cov == pytest.approx(0.3143, rel=1.0e-3) # In this case, applying conditioning to the same samples reduces c.o.v.; not sure if this is universal

def test_cal_Msys_by_cond_VE2(setup_hybrid):
    #(sys, x*) is computed for e.g. component importance.

    varis, cpms = setup_hybrid

    var_elim_order = ['haz', 'x1']
    result = cpm.cal_Msys_by_cond_VE(cpms, varis, ['haz'], var_elim_order, 'sys')

    np.testing.assert_array_almost_equal(result.C, np.array([[0, 0], [1, 1]]))
    np.testing.assert_array_almost_equal(result.p, np.array([[0.045, 0.585]]).T, decimal=3)
    np.testing.assert_array_almost_equal(result.Cs, np.array([[0,0],[1,1],[1,1],[0,0],[1,1],[0,0],[1,1],[1,1],[0,0],[1,1]]), decimal=3)
    np.testing.assert_array_almost_equal(result.q, np.array([[0.049, 0.189, 0.189, 0.036, 0.189, 0.049, 0.189, 0.189, 0.036, 0.189]]).T, decimal=3)
    np.testing.assert_array_almost_equal(result.ps, np.array([[0.0343, 0.1323, 0.1323, 0.0147, 0.1323, 0.0252, 0.0672, 0.0672, 0.0108, 0.0672]]).T, decimal=3)
    np.testing.assert_array_almost_equal(result.sample_idx, np.array([[0,1,2,3,4,0,1,2,3,4]]).T)

    prob, cov, cint = cpm.get_prob_and_cov( result, ['sys', 'x0'], [0,0], flag = True, nsample_repeat = 5 )

    assert prob == pytest.approx(0.1873, rel=1.0e-3)
    assert cov == pytest.approx(0.3143, rel=1.0e-3) # In this case, applying conditioning to the same samples reduces c.o.v.; not sure if this is universal


def test_get_prob_and_cov_cond1(setup_hybrid):

    varis, cpms = setup_hybrid

    var_elim_order = ['haz', 'x1']
    result = cpm.cal_Msys_by_cond_VE(cpms, varis, ['haz'], var_elim_order, 'sys')

    prob_m, cov_m, cint_m = cpm.get_prob_and_cov( result, ['sys'], [0], method='MLE', nsample_repeat=5 )
    prob_b, cov_b, cint_b = cpm.get_prob_and_cov( result, ['sys'], [0], method='Bayesian', nsample_repeat=5 )

    prob_c, cov_c, cint_c = cpm.get_prob_and_cov_cond( result, ['x0', 'sys'], [0,0], ['sys'], [0], nsample_repeat=5, conf_p=0.95 )

    assert prob_c >= cint_c[0] and prob_c <= cint_c[1]
    assert cint_c[0] <= 1 and cint_c[1] >= 1 # Truth: P(X0=0 | S = 0) = 1

def test_get_prob_and_cov_cond2(setup_hybrid):

    varis, cpms = setup_hybrid

    var_elim_order = ['haz', 'x0']
    result = cpm.cal_Msys_by_cond_VE(cpms, varis, ['haz'], var_elim_order, 'sys')

    prob_c, cov_c, cint_c = cpm.get_prob_and_cov_cond( result, ['x1', 'sys'], [0,0], ['sys'], [0], nsample_repeat=5, conf_p=0.95 )

    assert prob_c >= cint_c[0] and prob_c <= cint_c[1]
    pr_x0_s0 = 0.3462 # True value of P(X1=0 | S = 0) 
    assert cint_c[0] <= pr_x0_s0 and cint_c[1] >= pr_x0_s0 


@pytest.fixture()
def setup_hybrid_no_samp(): 
    varis, cpms = {}, {}

    varis['haz'] = variable.Variable(name='haz', values=['mild', 'severe'])
    cpms['haz'] = cpm.Cpm(variables=[varis['haz']], no_child=1, C=np.array([0,1]), p=[0.7, 0.3])

    varis['x0'] = variable.Variable(name='x0', values=['fail', 'surv'])
    cpms['x0'] = cpm.Cpm(variables=[varis['x0'], varis['haz']], no_child=1, C=np.array([[0,0],[1,0],[0,1],[1,1]]), p=[0.1,0.9,0.2,0.8])
    varis['x1'] = variable.Variable(name='x1', values=['fail', 'surv'])
    cpms['x1'] = cpm.Cpm(variables=[varis['x1'], varis['haz']], no_child=1, C=np.array([[0,0],[1,0],[0,1],[1,1]]), p=[0.3,0.7,0.4,0.6])

    varis['sys'] = variable.Variable(name='sys', values=['fail', 'surv'])
    cpms['sys'] = cpm.Cpm(variables=[varis['sys'], varis['x0'], varis['x1']], no_child=1, C=np.array([[0,0,0],[1,1,1]]), p=np.array([1.0,1.0])) # incomplete C (i.e. C does not include all samples)

    def sys_fun(comp_st): # ground truth (similar format for gen_bnb.py but no "min_comps_st") # TODO: can we put this as a property of a CPM..?
        if [comp_st['x0'], comp_st['x1']] == [0,1] or [comp_st['x0'], comp_st['x1']] == [0,0]:
            sys_val, sys_st = 0, 0
        elif [comp_st['x0'], comp_st['x1']] == [1,0] or [comp_st['x0'], comp_st['x1']] == [1,1]:
            sys_val, sys_st = 1, 1
        return sys_val, sys_st

    return varis, cpms, sys_fun

def test_cal_Msys_by_cond_VE3(setup_hybrid_no_samp):

    varis, cpms, _ = setup_hybrid_no_samp

    var_elim_order = ['haz', 'x0', 'x1']
    result = cpm.cal_Msys_by_cond_VE(cpms, varis, ['haz'], var_elim_order, 'sys')

    np.testing.assert_array_almost_equal(result.C, np.array([[0], [1]]))
    np.testing.assert_array_almost_equal(result.p, np.array([[0.045, 0.585]]).T, decimal=3)

    prob_bnd_s0 = cpm.get_prob_bnd( result, ['sys'], [0] )

    assert prob_bnd_s0 == pytest.approx([0.045, 0.415], rel=1.0e-3)

def test_cal_Msys_by_cond_VE4(setup_hybrid_no_samp):

    varis, cpms, _ = setup_hybrid_no_samp

    var_elim_order = ['haz', 'x1']
    result = cpm.cal_Msys_by_cond_VE(cpms, varis, ['haz'], var_elim_order, 'sys')

    np.testing.assert_array_almost_equal(result.C, np.array([[0, 0], [1, 1]]))
    np.testing.assert_array_almost_equal(result.p, np.array([[0.045, 0.585]]).T, decimal=3)

    prob_bnd_x0_s0 = cpm.get_prob_bnd( result, ['x0','sys'], [0,0], cvar_inds = ['sys'], cvar_states = [0] )

    assert prob_bnd_x0_s0 == pytest.approx( [0.045/0.415, 1], rel=1.0e-3 )


def test_rejection_sampling_sys(setup_hybrid_no_samp):

    var_elim_order = ['haz', 'x0', 'x1', 'sys']
    varis, cpms, sys_fun = setup_hybrid_no_samp
    Msys = cpm.cal_Msys_by_cond_VE(cpms, varis, ['haz'], var_elim_order, 'sys')

    cpms2, result = cpm.rejection_sampling_sys(cpms, 'sys', sys_fun, 0.05, sys_st_monitor = 0, known_prob = Msys.p.sum(), sys_st_prob = Msys.p[0], rand_seed = 1)

    var_elim_order = [varis['haz'], varis['x0'], varis['x1']]
    cpm_sys = cpm.variable_elim(cpms2, var_elim_order)

    prob_m, cov_m, cint_m = cpm.get_prob_and_cov(cpm_sys, ['sys'], [0], method='MLE')
    prob_b, cov_b, cint_b = cpm.get_prob_and_cov(cpm_sys, ['sys'], [0], method='Bayesian') # confidence interval (cint) is more conservative than with MLE

    assert prob_m == pytest.approx(prob_b, abs=1.0e-3), f'{result["pf"]}'
    assert cov_b == pytest.approx(result['cov'][0], abs=1.0e-3), f'{result["cov"]}'
    assert prob_m > cint_m[0] and prob_m < cint_m[1]
    assert prob_b > cint_b[0] and prob_b < cint_b[1]

# FIXME: NYI
def test_get_means():

    pass
