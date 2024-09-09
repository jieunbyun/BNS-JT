import numpy as np
import sys, os
import pytest
import pdb
import copy
import random
from pathlib import Path

np.set_printoptions(precision=3)
#pd.set_option.display_precision = 3

from BNS_JT import cpm, variable, config, trans, brc, operation

HOME = Path(__file__).parent


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

    result = operation.isinscope([A1], cpms)
    expected = np.array([[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]]).T
    np.testing.assert_array_equal(expected, result)

    result = operation.isinscope([A1, A2], cpms)
    expected = np.array([[1, 1, 0, 0, 0, 0, 1, 1, 1, 1]]).T
    np.testing.assert_array_equal(expected, result)


def test_get_sample_order(setup_mcs_product):

    cpms = setup_mcs_product
    cpms = [cpms[k] for k in [1, 2, 3]]

    sampleOrder, sampleVars, varAdditionOrder = operation.get_sample_order(cpms)
    expected = [0, 1, 2]
    np.testing.assert_array_equal(sampleOrder, expected)
    np.testing.assert_array_equal(varAdditionOrder, expected)

    expected = ['v1', 'v2', 'v3']
    result = [x.name for x in sampleVars]
    np.testing.assert_array_equal(result, expected)


def test_get_prod_idx1(setup_mcs_product):

    cpms = setup_mcs_product
    cpms = [cpms[k] for k in [1, 2, 3]]
    result = operation.get_prod_idx(cpms, [])

    expected = 0
    np.testing.assert_array_equal(result, expected)


def test_get_prod_idx2(setup_mcs_product):

    cpms = setup_mcs_product
    cpms = {k:cpms[k] for k in [1, 2, 3]}

    varis = cpms[3].get_variables(['v3'])
    result = operation.get_prod_idx(cpms, varis)

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
    sample, sampleProb = operation.single_sample(cpms, sampleOrder, sampleVars, varAdditionOrder, sampleInd)

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

    sample, sampleProb = operation.single_sample(cpms, sampleOrder, sampleVars, varAdditionOrder, sampleInd)

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

    sample, sampleProb = operation.single_sample(cpms, sampleOrder, sampleVars, varAdditionOrder, sampleInd, is_scalar=False)

    if (sample == [1, 1, 1]).all():
        np.testing.assert_array_almost_equal(sampleProb, [[0.9],[0.99],[0.95]], decimal=3)
    elif (sample == [2, 1, 1]).all():
        np.testing.assert_array_almost_equal(sampleProb, [[0.1],[0.9],[0.85]], decimal=3)

def test_mcs_product1(setup_mcs_product):

    cpms = setup_mcs_product
    cpms = {k:cpms[k] for k in [1, 2, 3]}

    nSample = 10
    Mcs = operation.mcs_product(cpms, nSample)

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
    Mcs = operation.mcs_product(cpms, nSample)

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
    Mcs = operation.mcs_product(cpms, nSample)

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
    Mcs = operation.mcs_product(cpms_, nSample)

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
    Mcs = operation.mcs_product(cpms_, nSample, is_scalar=False)

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

    #with pytest.raises(TypeError):
    Mcs = operation.mcs_product(cpms, nSample)
    assert Mcs.variables == []
    assert Mcs.no_child == 0
    assert Mcs.C.shape[0] == 0
    assert Mcs.p.shape[0] == 0


def test_condition(setup_mcs_product):

    cpms = setup_mcs_product
    v1, v2 = cpms[2].get_variables(['v1', 'v2'])
    condVars = [v1, v2]
    condStates = np.array([0, 0])
    """
    M[3] = cpm.Cpm(variables=[v3, v1],
                   no_child=1,
                   C = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]),
                   p = np.array([0.95, 0.05, 0.85, 0.15]).T)
    """
    [M]= operation.condition([cpms[3]], condVars, condStates)
    np.testing.assert_array_equal(M.C, np.array([[0, 0], [1, 0]]))
    np.testing.assert_array_equal(M.p, np.array([[0.95], [0.05]]))
    np.testing.assert_array_equal([x.name for x in M.variables], ['v3', 'v1'])
    assert M.q.any() == False
    assert M.sample_idx.any() == False

    # condvars = v1
    condVars = [v1]
    condStates = [0]
    [M]= operation.condition([cpms[3]], condVars, condStates)
    np.testing.assert_array_equal(M.C, np.array([[0, 0], [1, 0]]))
    np.testing.assert_array_equal(M.p, np.array([[0.95], [0.05]]))
    np.testing.assert_array_equal([x.name for x in M.variables], ['v3', 'v1'])
    assert M.q.any() == False
    assert M.sample_idx.any() == False

    # using string for cond_vars -> not working as variable['v2'] does not exist in cpms
    #condVars = ['v1', 'v2']

    # using dict for cpms
    M= operation.condition({0: cpms[3]}, condVars, condStates)
    np.testing.assert_array_equal(M[0].C, np.array([[0, 0], [1, 0]]))
    np.testing.assert_array_equal(M[0].p, np.array([[0.95], [0.05]]))
    np.testing.assert_array_equal([x.name for x in M[0].variables], ['v3', 'v1'])
    assert M[0].q.any() == False
    assert M[0].sample_idx.any() == False

    # using cpm for cpm
    [M]= operation.condition(cpms[3], condVars, condStates)
    np.testing.assert_array_equal(M.C, np.array([[0, 0], [1, 0]]))
    np.testing.assert_array_equal(M.p, np.array([[0.95], [0.05]]))
    np.testing.assert_array_equal([x.name for x in M.variables], ['v3', 'v1'])
    assert M.q.any() == False
    assert M.sample_idx.any() == False

    # test cpm.method
    M = cpms[3].condition(condVars, condStates)
    np.testing.assert_array_equal(M.C, np.array([[0, 0], [1, 0]]))
    np.testing.assert_array_equal(M.p, np.array([[0.95], [0.05]]))
    np.testing.assert_array_equal([x.name for x in M.variables], ['v3', 'v1'])
    assert M.q.any() == False
    assert M.sample_idx.any() == False


def test_get_var_idx(setup_Msys_Mcomps):

    varis, _ = setup_Msys_Mcomps

    _varis = [varis['x0'], varis['x2']]

    assert operation.get_var_idx(_varis, ['x0', 'x2']) == [0, 1]
    assert operation.get_var_idx(_varis, ['x2', 'x0']) == [1, 0]


def test_get_variables_from_cpms1():

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

    [res] = operation.get_variables(M, ['v1'])
    assert res.name == 'v1'

    res = operation.get_variables(M, ['v1', 'v3', 'v2'])
    assert [x.name for x in res] == ['v1', 'v3', 'v2']

    with pytest.raises(AssertionError):
        operation.get_variables(M, ['v4', 'v1', 'v2'])


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

    condVars = operation.get_variables([Mx], condVars)
    assert [x.name for x in condVars] == ['v2', 'v3']


def test_variable_elim(setup_bridge):

    d_cpms_arc, d_vars_arc, arcs, _ = setup_bridge
    cpms_arc = copy.deepcopy(d_cpms_arc)
    vars_arc = copy.deepcopy(d_vars_arc)

    cpms = [cpms_arc[k] for k in ['od1'] + list(arcs.keys())]
    var_elim_order = [vars_arc[i] for i in arcs.keys()]
    result = operation.variable_elim(cpms, var_elim_order)

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
    sys_fun = trans.sys_fun_wrap(cfg.infra['G'], od_pair, varis)

    brs, _, _, _ = brc.run(varis, probs, sys_fun, max_sf=1000, max_nb=1000)

    csys_by_od, varis_by_od = brc.get_csys(brs, varis, st_br_to_cs)

    varis['sys'] = variable.Variable(name='sys', values=['f', 's'])
    cpms['sys'] = cpm.Cpm(variables = [varis[k] for k in ['sys'] + list(cfg.infra['edges'].keys())],
                          no_child = 1,
                          C = csys_by_od.copy(),
                          p = np.ones(csys_by_od.shape[0]))
    cpms_comps = {k: cpms[k] for k in cfg.infra['edges'].keys()}

    cpms_new = operation.prod_Msys_and_Mcomps(cpms['sys'], list(cpms_comps.values()))

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

    p_f = cpms_new.get_prob(['sys'], [0])
    p_s = cpms_new.get_prob(['sys'], [1])

    assert p_f == pytest.approx(0.002236, rel=1.0e-3)
    assert p_s == pytest.approx(0.997763, rel=1.0e-3)

    # FIXME: expected value required
    p_x = cpms_new.get_prob(['sys', 'e1'], [0, 1])
    assert p_x == pytest.approx(0.002002, rel=1.0e-3)

    p_f = cpms_new.get_prob([varis['sys']], [0])
    assert p_f == pytest.approx(0.002236, rel=1.0e-3)


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


def test_get_variables_from_cpms3(setup_hybrid):

    varis, cpms = setup_hybrid
    vars_ = operation.get_variables(cpms,['x0','x1'])

    assert all(isinstance(v, variable.Variable) for v in vars_)
    assert vars_[0].name == 'x0'
    assert vars_[1].name == 'x1'


def test_get_variables_from_cpms4(setup_hybrid_no_samp):

    varis, cpms, _ = setup_hybrid_no_samp
    with pytest.raises(AssertionError):
        _ = operation.get_variables(cpms, varis.values())


def test_condition6(setup_hybrid):

    varis, cpms = setup_hybrid
    """
    cpms['haz'] = cpm.Cpm(variables=[varis['haz']], no_child=1, C=np.array([0,1]), p=[0.7, 0.3])
    cpms['x0'] = cpm.Cpm(variables=[varis['x0'], varis['haz']], no_child=1, C=np.array([[0,0],[1,0],[0,1],[1,1]]), p=[0.1,0.9,0.2,0.8])
    cpms['x1'] = cpm.Cpm(variables=[varis['x1'], varis['haz']], no_child=1, C=np.array([[0,0],[1,0],[0,1],[1,1]]), p=[0.3,0.7,0.4,0.6])
    cpms['sys'] = cpm.Cpm(variables=[varis['sys'], varis['x0'], varis['x1']], no_child=1, C=np.array([[0,0,0],[1,1,1]]), p=np.array([1.0,1.0])) # incomplete C (i.e. C does not include all samples)
    """
    Mc = operation.condition(cpms, ['haz'], [0])

    np.testing.assert_array_almost_equal(Mc['haz'].C, np.array([[0]]))
    np.testing.assert_array_almost_equal(Mc['haz'].ps, cpms['haz'].q * 0.7)

    np.testing.assert_array_almost_equal(Mc['x0'].C, np.array([[0, 0], [1, 0]]))
    np.testing.assert_array_almost_equal(Mc['x0'].ps, np.array([[0.1],[0.9],[0.9],[0.1],[0.9]]))

    np.testing.assert_array_almost_equal(Mc['x1'].C, np.array([[0, 0], [1, 0]]))
    np.testing.assert_array_almost_equal(Mc['x1'].ps, np.array([[0.7],[0.3],[0.3],[0.7],[0.3]]))

    np.testing.assert_array_almost_equal(Mc['sys'].C, np.array([[0, 0, 0], [1, 1, 1]]))
    np.testing.assert_array_almost_equal(Mc['sys'].ps, np.array([[1.0],[1.0],[1.0],[1.0],[1.0]]))


def test_variable_elim1(setup_hybrid):

    varis, cpms = setup_hybrid

    var_elim_order = [varis['haz'], varis['x0'], varis['x1']]
    result = operation.variable_elim(cpms, var_elim_order)

    np.testing.assert_array_almost_equal(result.C, np.array([[0], [1]]))
    np.testing.assert_array_almost_equal(result.p, np.array([[0.045, 0.585]]).T, decimal=3)
    np.testing.assert_array_almost_equal(result.Cs, np.array([[0,1,1,0,1]]).T, decimal=3)
    np.testing.assert_array_almost_equal(result.q, np.array([[0.049, 0.189, 0.189, 0.036, 0.189]]).T, decimal=3)
    np.testing.assert_array_almost_equal(result.ps, result.q)
    np.testing.assert_array_almost_equal(result.sample_idx, np.array([[0,1,2,3,4]]).T)

    prob, cov, cint = result.get_prob_and_cov(['sys'], [0] )

    assert prob == pytest.approx(0.193, rel=1.0e-3)
    assert cov == pytest.approx(0.4200, rel=1.0e-3)


def test_prod_cpm_sys_and_comps1(setup_Msys_Mcomps):

    varis, cpms = setup_Msys_Mcomps

    Msys = operation.prod_Msys_and_Mcomps(cpms['sys'], [cpms['x0'], cpms['x1'], cpms['x2']])

    assert Msys.variables == [varis['sys'], varis['x0'], varis['x1'], varis['x2']]
    assert Msys.no_child == 4
    np.testing.assert_array_almost_equal(Msys.C, np.array([[0,0,2,2],[1,1,1,2],[1,1,0,1],[0,1,0,0]]))
    np.testing.assert_array_almost_equal(Msys.p, np.array([[0.1, 0.72, 0.126, 0.054]]).T, decimal=3)


def test_prod_cpm_sys_and_comps2(setup_Msys_Mcomps):

    varis, cpms = setup_Msys_Mcomps

    Msys = operation.prod_Msys_and_Mcomps(cpms['sys'], [cpms['x0_c'], cpms['x1_c'], cpms['x2_c']])

    assert Msys.variables == [varis['sys'], varis['x0'], varis['x1'], varis['x2'], varis['haz']]
    assert Msys.no_child == 4
    np.testing.assert_array_almost_equal(Msys.C, np.array([[0,0,2,2,0],[1,1,1,2,0],[1,1,0,1,0],[0,1,0,0,0]]))
    np.testing.assert_array_almost_equal(Msys.p, np.array([[0.05, 0.855, 0.076, 0.019]]).T, decimal=3)


def test_prod_cpm_sys_and_comps3(setup_Msys_Mcomps):

    varis, cpms = setup_Msys_Mcomps

    Msys = operation.prod_Msys_and_Mcomps(cpms['sys'], [cpms['x0'], cpms['x2']])

    assert Msys.variables == [varis['sys'], varis['x0'], varis['x2'], varis['x1']]
    assert Msys.no_child == 3
    np.testing.assert_array_almost_equal(Msys.C, np.array([[0,0,2,2],[1,1,2,1],[1,1,1,0],[0,1,0,0]]))
    np.testing.assert_array_almost_equal(Msys.p, np.array([[0.1, 0.9, 0.63, 0.27]]).T, decimal=3)


def test_prod_cpm_sys_and_comps4(setup_Msys_Mcomps):

    varis, cpms = setup_Msys_Mcomps

    Msys = operation.prod_Msys_and_Mcomps(cpms['sys'], [cpms['x1_c'], cpms['x2_c']])

    assert Msys.variables == [varis['sys'], varis['x1'], varis['x2'], varis['x0'], varis['haz']]
    assert Msys.no_child == 3
    np.testing.assert_array_almost_equal(Msys.C, np.array([[0,2,2,0,0],[1,1,2,1,0],[1,0,1,1,0],[0,0,0,1,0]]))
    np.testing.assert_array_almost_equal(Msys.p, np.array([[1.0, 0.9, 0.08, 0.02]]).T, decimal=3)


def test_prod_cpm_sys_and_comps5(setup_Msys_Mcomps):

    varis, cpms = setup_Msys_Mcomps

    cpms['x1_c'].C = np.array([[0,1], [1,1]])
    with pytest.raises(AssertionError):
        Msys = operation.prod_Msys_and_Mcomps(cpms['sys'], [cpms['x0_c'], cpms['x1_c']])

    varis['haz2'] = variable.Variable(name='haz2', values=['mild', 'severe'])
    cpms['x2_c'].variables = [varis['x2'], varis['haz2']]
    with pytest.raises(AssertionError):
        Msys = operation.prod_Msys_and_Mcomps(cpms['sys'], [cpms['x0_c'], cpms['x2_c']])


def test_prod_cpm_sys_and_comps6(setup_Msys_Mcomps):

    varis, cpms = setup_Msys_Mcomps

    varis['x3'] = variable.Variable(name='x3', values=['fail', 'surv'])
    cpms['x3'] = cpm.Cpm( [varis['x3']], 1, np.array([0, 1]), p = [0.3, 0.7] )
    Msys = operation.prod_Msys_and_Mcomps(cpms['sys'], [cpms['x0'], cpms['x2'], cpms['x3']])

    assert Msys.variables == [varis['sys'], varis['x0'], varis['x2'], varis['x1']]
    assert Msys.no_child == 3
    np.testing.assert_array_almost_equal(Msys.C, np.array([[0,0,2,2],[1,1,2,1],[1,1,1,0],[0,1,0,0]]))
    np.testing.assert_array_almost_equal(Msys.p, np.array([[0.1, 0.9, 0.63, 0.27]]).T, decimal=3)


def test_cal_Msys_by_cond_VE1(setup_hybrid):

    varis, cpms = setup_hybrid

    var_elim_order = ['haz', 'x0', 'x1']
    result = operation.cal_Msys_by_cond_VE(cpms, varis, ['haz'], var_elim_order, 'sys')

    np.testing.assert_array_almost_equal(result.C, np.array([[0], [1]]))
    np.testing.assert_array_almost_equal(result.p, np.array([[0.045, 0.585]]).T, decimal=3)
    np.testing.assert_array_almost_equal(result.Cs, np.array([[0,1,1,0,1,0,1,1,0,1]]).T, decimal=3)
    np.testing.assert_array_almost_equal(result.q, np.array([[0.049, 0.189, 0.189, 0.036, 0.189, 0.049, 0.189, 0.189, 0.036, 0.189]]).T, decimal=3)
    np.testing.assert_array_almost_equal(result.ps, np.array([[0.0343, 0.1323, 0.1323, 0.0147, 0.1323, 0.0252, 0.0672, 0.0672, 0.0108, 0.0672]]).T, decimal=3)
    np.testing.assert_array_almost_equal(result.sample_idx, np.array([[0,1,2,3,4,0,1,2,3,4]]).T)

    prob, cov, cint = result.get_prob_and_cov(['sys'], [0], flag = True, nsample_repeat = 5 )

    assert prob == pytest.approx(0.1873, rel=1.0e-3)
    assert cov == pytest.approx(0.3143, rel=1.0e-3) # In this case, applying conditioning to the same samples reduces c.o.v.; not sure if this is universal


def test_cal_Msys_by_cond_VE2(setup_hybrid):
    #(sys, x*) is computed for e.g. component importance.

    varis, cpms = setup_hybrid

    var_elim_order = ['haz', 'x1']
    result = operation.cal_Msys_by_cond_VE(cpms, varis, ['haz'], var_elim_order, 'sys')

    np.testing.assert_array_almost_equal(result.C, np.array([[0, 0], [1, 1]]))
    np.testing.assert_array_almost_equal(result.p, np.array([[0.045, 0.585]]).T, decimal=3)
    np.testing.assert_array_almost_equal(result.Cs, np.array([[0,0],[1,1],[1,1],[0,0],[1,1],[0,0],[1,1],[1,1],[0,0],[1,1]]), decimal=3)
    np.testing.assert_array_almost_equal(result.q, np.array([[0.049, 0.189, 0.189, 0.036, 0.189, 0.049, 0.189, 0.189, 0.036, 0.189]]).T, decimal=3)
    np.testing.assert_array_almost_equal(result.ps, np.array([[0.0343, 0.1323, 0.1323, 0.0147, 0.1323, 0.0252, 0.0672, 0.0672, 0.0108, 0.0672]]).T, decimal=3)
    np.testing.assert_array_almost_equal(result.sample_idx, np.array([[0,1,2,3,4,0,1,2,3,4]]).T)

    prob, cov, cint = result.get_prob_and_cov(['sys', 'x0'], [0,0], flag = True, nsample_repeat = 5 )

    assert prob == pytest.approx(0.1873, rel=1.0e-3)
    assert cov == pytest.approx(0.3143, rel=1.0e-3) # In this case, applying conditioning to the same samples reduces c.o.v.; not sure if this is universal



def test_cal_Msys_by_cond_VE3(setup_hybrid_no_samp):

    varis, cpms, _ = setup_hybrid_no_samp

    var_elim_order = ['haz', 'x0', 'x1']
    result = operation.cal_Msys_by_cond_VE(cpms, varis, ['haz'], var_elim_order, 'sys')

    np.testing.assert_array_almost_equal(result.C, np.array([[0], [1]]))
    np.testing.assert_array_almost_equal(result.p, np.array([[0.045, 0.585]]).T, decimal=3)


def test_cal_Msys_by_cond_VE4(setup_hybrid_no_samp):

    varis, cpms, _ = setup_hybrid_no_samp

    var_elim_order = ['haz', 'x1']
    result = operation.cal_Msys_by_cond_VE(cpms, varis, ['haz'], var_elim_order, 'sys')

    np.testing.assert_array_almost_equal(result.C, np.array([[0, 0], [1, 1]]))
    np.testing.assert_array_almost_equal(result.p, np.array([[0.045, 0.585]]).T, decimal=3)

    prob_bnd_x0_s0 = result.get_prob_bnd(['x0','sys'], [0,0], cvar_inds = ['sys'], cvar_states = [0] )

    assert prob_bnd_x0_s0 == pytest.approx([0.045/0.415, 1], rel=1.0e-3 )


def test_get_prod_idx3(setup_hybrid_no_samp):
    """
    created to replicate the problem in test_rejection_sampling_sys

    """
    _, cpms, _ = setup_hybrid_no_samp

    sample_vars = []
    cpms_ = [cpms[k] for k in ['x0', 'x1']]
    # x0, haz
    # x1, haz

    out = operation.get_prod_idx(cpms_, sample_vars)

    assert out == None


@pytest.mark.skip("FIXME: Does not work anymore. get_prod_idx return None")
def test_rejection_sampling_sys(setup_hybrid_no_samp):

    var_elim_order = ['haz', 'x0', 'x1', 'sys']
    varis, cpms, sys_fun = setup_hybrid_no_samp
    Msys = operation.cal_Msys_by_cond_VE(cpms, varis, ['haz'], var_elim_order, 'sys')

    cpms2, result = operation.rejection_sampling_sys(cpms, 'sys', sys_fun, 0.05, sys_st_monitor = 0, known_prob = Msys.p.sum(), sys_st_prob = Msys.p[0], rand_seed = 1)

    var_elim_order = [varis['haz'], varis['x0'], varis['x1']]
    cpm_sys = operation.variable_elim(cpms2, var_elim_order)

    prob_m, cov_m, cint_m = cpm_sys.get_prob_and_cov(['sys'], [0], method='MLE')
    prob_b, cov_b, cint_b = cpm_sys.get_prob_and_cov(['sys'], [0], method='Bayesian') # confidence interval (cint) is more conservative than with MLE

    assert prob_m == pytest.approx(prob_b, abs=1.0e-3), f'{result["pf"]}'
    assert cov_b == pytest.approx(result['cov'][0], abs=1.0e-3), f'{result["cov"]}'
    assert prob_m > cint_m[0] and prob_m < cint_m[1]
    assert prob_b > cint_b[0] and prob_b < cint_b[1]

# FIXME: NYI
def test_get_means():

    pass



@pytest.fixture(scope='package')
def sys_2comps():

    vars_p = {}
    vars_p['x0'] = variable.Variable(name='x0', values=[0,1,2])
    vars_p['x1'] = variable.Variable(name='x1', values=[0,1,2,3])

    return vars_p

@pytest.fixture(scope='package')
def sys_3comps():

    vars_p = {}
    vars_p['x0'] = variable.Variable(name='x0', values=[0,1,2])
    vars_p['x1'] = variable.Variable(name='x1', values=[0,1,2,3])
    vars_p['x2'] = variable.Variable(name='x2', values=[0,1])

    return vars_p


def test_sys_max_val1(sys_2comps):

    vars_p = sys_2comps
    M, v = operation.sys_max_val('s', [vars_p['x0'], vars_p['x1']])

    np.testing.assert_array_equal(M.C, [[0, 0, 0], [1, 1, 4], [2, 2, 10], [1, 0, 1], [2, 3, 2], [3, 6, 3]])
    assert v.values==[0,1,2,3]


def test_sys_max_val2(sys_3comps):

    vars_p = sys_3comps
    M, v = operation.sys_max_val('s', [vars_p['x0'], vars_p['x1'], vars_p['x2']])

    np.testing.assert_array_equal(M.C, [[0, 0, 0, 0], [1, 1, 4, 2], [2, 2, 10, 2], [1, 0, 1, 2], [2, 3, 2, 2], [3, 6, 3, 2], [1, 0, 0, 1]])
    assert v.values==[0,1,2,3]


def test_sys_min_val1(sys_2comps):

    vars_p = sys_2comps
    M, v = operation.sys_min_val('s', [vars_p['x0'], vars_p['x1']])

    np.testing.assert_array_equal(M.C, [[0, 0, 14], [1, 1, 13], [2, 2, 9], [0, 5, 0], [1, 2, 1]])
    assert v.values==[0,1,2]

def test_sys_min_val2(sys_3comps):

    vars_p = sys_3comps
    M, v = operation.sys_min_val('s', [vars_p['x0'], vars_p['x1'], vars_p['x2']])

    np.testing.assert_array_equal(M.C, [[0, 6, 14, 0], [1, 5, 13, 1], [0, 0, 14, 1], [0, 5, 0, 1]])
    assert v.values==[0,1]


def test_get_inf_vars(setup_hybrid):

    varis, cpms = setup_hybrid

    var_elim_order = ['haz', 'x0', 'x1']
    result = operation.get_inf_vars(cpms, 'sys', ve_ord=var_elim_order)
    assert result == ['haz', 'x0', 'x1', 'sys']

    result = operation.get_inf_vars(cpms, ['sys'], ve_ord=var_elim_order)
    assert result == ['haz', 'x0', 'x1', 'sys']


@pytest.fixture()
def sys_2comps_2haz():

    varis, cpms = {}, {}

    # Two hazard events
    varis['h0'] = variable.Variable(name='h0', values=['low', 'high'])
    cpms['h0'] = cpm.Cpm( [varis['h0']], 1, np.array([0, 1]), p = [0.9, 0.1] )

    varis['h1'] = variable.Variable(name='h1', values=['low', 'high'])
    cpms['h1'] = cpm.Cpm( [varis['h1']], 1, np.array([0, 1]), p = [0.8, 0.2] )


    # Component events. x1 is a highly fragile component.
    Cx = np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0],
                    [0,0,1], [1,0,1], [0,1,1], [1,1,1]])

    varis['x0'] = variable.Variable(name='x0', values=['fail', 'surv'])
    cpms['x0'] = cpm.Cpm( [varis['x0'], varis['h0'], varis['h1']], 1, Cx,
                         p = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6] )

    varis['x1'] = variable.Variable(name='x1', values=['fail', 'surv'])
    cpms['x1'] = cpm.Cpm( [varis['x1'], varis['h0'], varis['h1']], 1, Cx,
                         p = [0.5, 0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.2] )


    # This is parallel system
    varis['sys'] = variable.Variable(name='sys', values=['fail', 'surv'])
    cpms['sys'] = cpm.Cpm(variables=[varis['sys'], varis['x0'], varis['x1']], no_child=1,
                          C=np.array([[1,1,2],[1,0,1],[0,0,0]]), p=np.array([1,1,1], dtype=float))

    return varis, cpms


def test_variable_elim_cond0(sys_2comps_2haz):

    varis, cpms = sys_2comps_2haz
    ve_order = ['x0', 'x1']
    cpms_ve = [cpms[k] for k in ['sys', 'x0', 'x1']]
    M = operation.variable_elim_cond(cpms_ve, ve_order, [cpms['h0'], cpms['h1']])

    np.testing.assert_array_equal(M.C, [[0], [1]])
    np.testing.assert_array_almost_equal(M.p, [[0.0898], [0.9102]], decimal=3)


@pytest.fixture()
def max_flow_net_5_edge():
    nodes = {'n1': (0, 0),
            'n2': (1, 1),
            'n3': (1, -1),
            'n4': (2, 0)}

    edges = {'e1': ['n1', 'n2'],
            'e2': ['n1', 'n3'],
            'e3': ['n2', 'n3'],
            'e4': ['n2', 'n4'],
            'e5': ['n3', 'n4']}

    od_pair=('n1','n4')

    varis = {}
    for k, v in edges.items():
        varis[k] = variable.Variable( name=k, values = [0, 1, 2]) # values: edge flow capacity

    return od_pair, edges, varis

def test_max_flow1(max_flow_net_5_edge):

    od_pair, edges, varis = max_flow_net_5_edge

    comps_st = {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 2}
    f_val, sys_st, min_comps_st = operation.max_flow(comps_st, 1, od_pair, edges, varis)

    assert f_val == 1
    assert sys_st == 's'
    assert min_comps_st == {'e1': 1, 'e4': 1}
    
def test_max_flow2(max_flow_net_5_edge):

    od_pair, edges, varis = max_flow_net_5_edge

    comps_st = {'e1': 0, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 2}
    f_val, sys_st, min_comps_st = operation.max_flow(comps_st, 1, od_pair, edges, varis)

    assert f_val == 1
    assert sys_st == 's'
    assert min_comps_st == {'e2': 1, 'e5': 1}

def test_max_flow3(max_flow_net_5_edge):

    od_pair, edges, varis = max_flow_net_5_edge

    comps_st = {'e1': 0, 'e2': 0, 'e3': 2, 'e4': 2, 'e5': 2}
    f_val, sys_st, min_comps_st = operation.max_flow(comps_st, 1, od_pair, edges, varis)

    assert f_val == 0
    assert sys_st == 'f'
    assert min_comps_st == None