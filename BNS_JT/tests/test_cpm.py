import numpy as np
import sys, os
import pytest
import pdb
import copy
import random
from pathlib import Path

np.set_printoptions(precision=3)
#pd.set_option.display_precision = 3

from BNS_JT import cpm, variable, config, trans, gen_bnb, operation

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
    # using list for P
    v = dict_cpm
    a = cpm.Cpm(variables=[v['variables'][0]], no_child=1, C=np.array([1, 2]), p=[0.9, 0.1])
    assert isinstance(a, cpm.Cpm)


def test_init3(dict_cpm):
    # no p
    v = dict_cpm
    a = cpm.Cpm(variables=[v['variables'][0]], no_child=1, C=np.array([1, 2]))
    assert isinstance(a, cpm.Cpm)


def test_init4():
    # variables must be a list of Variable
    with pytest.raises(AssertionError):
        a = cpm.Cpm(variables=['1'], no_child=1)


def test_init5():
    # empty variables
    a = cpm.Cpm([], 0)
    with pytest.raises(AssertionError):
        a.variables = ['1']


def test_variables1(dict_cpm):
    # variable must be a list of Variables
    f_variables = [1, 2]
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


def test_get_variables1(dict_cpm):

    a = cpm.Cpm(**dict_cpm)
    A1x = a.get_variables('A1')

    assert A1x.name=='A1'
    assert A1x.values == ['s', 'f']


def test_get_variables2(dict_cpm):

    a = cpm.Cpm(**dict_cpm)

    A1x = a.get_variables('A1')

    assert A1x is dict_cpm['variables'][-1]

    # modify the A1x
    A1x.values = ['sy', 'fy']
    assert A1x.values == dict_cpm['variables'][-1].values


def test_get_subset1(setup_iscompatible):

    M = setup_iscompatible[0]

    # M[5]
    rowIndex = [0]
    result = M[5].get_subset(rowIndex)

    np.testing.assert_array_equal(result.C, np.array([[2, 3, 3, 2]])-1)
    np.testing.assert_array_equal(result.p, [[1]])


def test_get_subset2(setup_iscompatible):
    # flag=False

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


def test_get_subset4(setup_hybrid):

    _, cpms = setup_hybrid

    rowIndex = [0]
    result = cpms['haz'].get_subset(rowIndex, isC=False)

    np.testing.assert_array_equal(result.Cs, np.array([[0]]))
    np.testing.assert_array_equal(result.q, [[0.7]])


def test_get_subset5(setup_hybrid):

    _, cpms = setup_hybrid

    rowIndex = [0, 1, 2]
    result = cpms['haz'].get_subset(rowIndex, flag=False, isC=False)

    np.testing.assert_array_equal(result.Cs, np.array([[1],[0]]))
    np.testing.assert_array_equal(result.q, [[0.3],[0.7]])


def test_get_means1(setup_hybrid):

    _, cpms = setup_hybrid

    means = cpms['x0'].get_means(['haz'])

    assert means[0] == pytest.approx(1.0)


def test_get_means2(setup_mcs_product):

    cpms = setup_mcs_product
    means = cpms[5].get_means(['v4', 'v3'])

    assert means[0] == pytest.approx(1.0)
    assert means[1] == pytest.approx(5.0)


def test_iscompatibleCpm1(setup_iscompatible):

    # M[5]
    M, _ = setup_iscompatible
    rowIndex = [0]
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


def test_get_col_ind(setup_product):

    M = setup_product

    idx = M[2].get_col_ind(['X2'])
    assert idx == [0]

    idx = M[2].get_col_ind(['X1'])
    assert idx == [1]


def test_sort1(dict_cpm):

    v = dict_cpm

    A3, A2, A1 = v['variables']

    p = np.array([[0.9405, 0.0495, 0.0095, 0.0005, 0.7650, 0.1350, 0.0850, 0.0150]]).T
    C = np.array([[1, 1, 1], [1, 2, 1], [2, 1, 1], [2, 2, 1], [1, 1, 2], [1, 2, 2], [2, 1, 2], [2, 2, 2]]) - 1

    M = cpm.Cpm(variables=[A2, A3, A1],
            no_child = 2,
            C = C,
            p = p)

    M.sort()

    np.testing.assert_array_equal(M.C, np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1], [1, 1, 2], [2, 1, 2], [1, 2, 2], [2, 2, 2]]) - 1)
    np.testing.assert_array_almost_equal(M.p, np.array([[0.9405, 0.0095, 0.0495, 5.0e-4, 0.7650, 0.0850, 0.1350, 0.0150]]).T)


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


def test_iscompatible():
    C = np.array([[0,1], [0,1], [2,1], [1,1]])
    varis = {'e1': variable.Variable('e1', [0,1,2]), 'sys': variable.Variable('sys', [0,1])}
    variables = [varis['e1'], varis['sys']]
    check_vars = ['sys', 'e1']
    check_states = [0, 3]
    iscmp = cpm.iscompatible( C, variables, check_vars, check_states )

    np.testing.assert_array_equal(iscmp, np.array([False, False, False, False]))


def test_product1(setup_product):

    M = setup_product
    M1 = M[2]
    M2 = M[3]

    Mprod = M1.product(M2)

    names = [x.name for x in Mprod.variables]
    assert names == ['X2', 'X3', 'X1']

    assert Mprod.no_child==2
    np.testing.assert_array_equal(Mprod.C, np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1], [1, 1, 2], [2, 1, 2], [1, 2, 2], [2, 2, 2]]) - 1)
    np.testing.assert_array_almost_equal(Mprod.p, np.array([[0.9405, 0.0095, 0.0495, 5.0e-4, 0.7650, 0.0850, 0.1350, 0.0150]]).T)

def test_product2(setup_product):

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


def test_sum1(setup_sum):

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

def test_sum2(setup_sum):

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

def test_sum3(setup_sum):

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

def test_sum4(setup_sum):

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
def test_sum5(setup_bridge):

    d_cpms_arc, d_vars_arc, _, _ = setup_bridge
    cpms_arc = copy.deepcopy(d_cpms_arc)
    vars_arc = copy.deepcopy(d_vars_arc)

    cpms_arc_cp = [cpms_arc[k] for k in ['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'od1']]

    is_inscope = cpm.isinscope([vars_arc['e1']], cpms_arc_cp)
    cpm_sel = [y for x, y in zip(is_inscope, cpms_arc_cp) if x]
    cpm_mult = cpm.product(cpm_sel)

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

    M = cpms[3].condition(condVars, condStates)
    np.testing.assert_array_equal(M.C, np.array([[0, 0], [1, 0]]))
    assert M.q.any() == False
    assert M.sample_idx.any() == False


def test_get_prob(setup_inference):

    d_cpms, d_varis, var_elim_order, arcs = setup_inference

    cpms = copy.deepcopy(d_cpms)
    varis = copy.deepcopy(d_varis)

    ## Repeat inferences again using new functions -- the results must be the same.
    # Probability of delay and disconnection
    M = [cpms[k] for k in varis.keys()]
    M_VE2 = operation.variable_elim(M, var_elim_order)

    # Prob. of failure
    prob_f1 = M_VE2.get_prob([varis['sys']], [0])
    prob_f2 = M_VE2.get_prob([varis['sys']], ['f'])
    assert prob_f1 == prob_f2

    with pytest.raises(AssertionError):
        prob_f2 = M_VE2.get_prob([varis['sys']], ['d'])


def test_get_prob_bnd1(setup_hybrid_no_samp):

    varis, cpms, _ = setup_hybrid_no_samp

    var_elim_order = ['haz', 'x0', 'x1']
    result = operation.cal_Msys_by_cond_VE(cpms, varis, ['haz'], var_elim_order, 'sys')

    np.testing.assert_array_almost_equal(result.C, np.array([[0], [1]]))
    np.testing.assert_array_almost_equal(result.p, np.array([[0.045, 0.585]]).T, decimal=3)

    prob_bnd_s0 = result.get_prob_bnd(['sys'], [0])

    assert prob_bnd_s0 == pytest.approx([0.045, 0.415], rel=1.0e-3)


def test_get_prob_and_cov1(setup_hybrid):

    varis, cpms = setup_hybrid

    var_elim_order = [varis['haz'], varis['x0'], varis['x1']]
    result = operation.variable_elim(cpms, var_elim_order)

    prob, cov, cint = result.get_prob_and_cov(['sys'], [0])

    assert prob == pytest.approx(0.193, rel=1.0e-3)
    assert cov == pytest.approx(0.4200, rel=1.0e-3)


def test_get_prob_and_cov_cond1(setup_hybrid):

    varis, cpms = setup_hybrid

    var_elim_order = ['haz', 'x1']
    M = operation.cal_Msys_by_cond_VE(cpms, varis, ['haz'], var_elim_order, 'sys')

    prob_m, cov_m, cint_m = M.get_prob_and_cov(['sys'], [0], method='MLE', nsample_repeat=5 )
    prob_b, cov_b, cint_b = M.get_prob_and_cov(['sys'], [0], method='Bayesian', nsample_repeat=5 )

    prob_c, cov_c, cint_c = M.get_prob_and_cov_cond(['x0', 'sys'], [0,0], ['sys'], [0], nsample_repeat=5, conf_p=0.95 )

    assert prob_c >= cint_c[0] and prob_c <= cint_c[1]
    assert cint_c[0] <= 1 and cint_c[1] >= 1 # Truth: P(X0=0 | S = 0) = 1


def test_get_prob_and_cov_cond2(setup_hybrid):

    varis, cpms = setup_hybrid

    var_elim_order = ['haz', 'x0']
    M = operation.cal_Msys_by_cond_VE(cpms, varis, ['haz'], var_elim_order, 'sys')

    prob_c, cov_c, cint_c = M.get_prob_and_cov_cond(['x1', 'sys'], [0,0], ['sys'], [0], nsample_repeat=5, conf_p=0.95 )

    assert prob_c >= cint_c[0] and prob_c <= cint_c[1]
    pr_x0_s0 = 0.3462 # True value of P(X1=0 | S = 0) 
    assert cint_c[0] <= pr_x0_s0 and cint_c[1] >= pr_x0_s0 


def test_prod_cpms1(setup_prod_cms):

    cpms= setup_prod_cms
    Mmult = cpm.product(cpms=cpms)

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

    Mmult = cpm.product([M[k] for k in [1, 2]])

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

    Mmult = cpm.product(M)

    expected = np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1], [1, 1, 2], [2, 1, 2], [1, 2, 2], [2, 2, 2]]) - 1
    np.testing.assert_array_equal(Mmult.C, expected)

    expected = np.array([[0.7859, 0.1509, 0.0486, 0.0093, 0.0041, 0.0008, 0.0003, 0.000]]).T
    np.testing.assert_array_almost_equal(Mmult.p, expected, decimal=4)

    assert [x.name for x in Mmult.variables] == ['v1', 'v2', 'v3']


def test_product3( setup_hybrid ):

    varis, cpms = setup_hybrid
    Mp = cpms['haz'].product( cpms['x0'] )

    assert Mp.variables[0].name == 'haz' and Mp.variables[1].name == 'x0'
    np.testing.assert_array_almost_equal(Mp.C, np.array([[0,0], [1,0], [0,1], [1,1]]))
    np.testing.assert_array_almost_equal(Mp.p, np.array([[0.07], [0.06], [0.63], [0.24]]))
    np.testing.assert_array_almost_equal(Mp.Cs, np.array([[0,0],[0,1],[0,1],[1,0],[0,1]]))
    np.testing.assert_array_almost_equal(Mp.q, np.array([[0.07], [0.63], [0.63], [0.06], [0.63]]))
    np.testing.assert_array_almost_equal(Mp.q, Mp.ps)
    np.testing.assert_array_almost_equal(Mp.sample_idx, np.array([[0],[1],[2],[3],[4]]))


def test_product4( setup_hybrid ):
    varis, cpms = setup_hybrid
    Mc = operation.condition(cpms, ['haz'], [0])
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


def test_product5( setup_hybrid ):
    varis, cpms = setup_hybrid
    Mc = operation.condition(cpms, ['haz'], [0])
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



def test_sum6(setup_hybrid):
    varis, cpms = setup_hybrid
    Mc = operation.condition(cpms, ['haz'], [0])
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


def test_sum7(setup_hybrid):
    varis, cpms = setup_hybrid
    Mc = operation.condition(cpms, ['haz'], [0])
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


def test_merge1(setup_product):

    M = setup_product


    with pytest.raises(AssertionError):
        M_new = M[2].merge(M[3])


def test_merge2(setup_iscompatible):

    _, v = setup_iscompatible

    M1 = cpm.Cpm(variables=[v[2], v[1]], no_child=1, C = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]), p = np.array([0.99, 0.01, 0.9, 0.1]).T)

    M2 = cpm.Cpm(variables=[v[2], v[1]], no_child=1, C = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]), p = np.array([0.89, 0.11, 0.7, 0.3]).T)

    M_new = M1.merge(M2)

    np.testing.assert_array_equal(M_new.C, np.array([[0, 0], [1, 0], [0, 1], [1, 1]]))
    np.testing.assert_array_equal(M_new.p, np.array([[1.88, 0.12, 1.6, 0.4]]).T)
    np.testing.assert_array_equal(M_new.Cs, np.empty(shape=(2,0)))


def test_flip():

    idx = [1, 0, 2, True]

    assert cpm.flip(idx) == [False, False, False, False]
