import pytest
import numpy as np
from pathlib import Path

from BNS_JT import trans, branch, variable, cpm, quant

HOME = Path(__file__).parent

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
    M, v = quant.sys_max_val('s', [vars_p['x0'], vars_p['x1']])

    np.testing.assert_array_equal(M.C, [[0, 0, 0], [1, 1, 4], [2, 2, 10], [1, 0, 1], [2, 3, 2], [3, 6, 3]])
    assert v.values==[0,1,2,3]


def test_sys_max_val2(sys_3comps):

    vars_p = sys_3comps
    M, v = quant.sys_max_val('s', [vars_p['x0'], vars_p['x1'], vars_p['x2']])

    np.testing.assert_array_equal(M.C, [[0, 0, 0, 0], [1, 1, 4, 2], [2, 2, 10, 2], [1, 0, 1, 2], [2, 3, 2, 2], [3, 6, 3, 2], [1, 0, 0, 1]])
    assert v.values==[0,1,2,3]


def test_sys_min_val1(sys_2comps):

    vars_p = sys_2comps
    M, v = quant.sys_min_val('s', [vars_p['x0'], vars_p['x1']])

    np.testing.assert_array_equal(M.C, [[0, 0, 14], [1, 1, 13], [2, 2, 9], [0, 5, 0], [1, 2, 1]])
    assert v.values==[0,1,2]

def test_sys_min_val2(sys_3comps):

    vars_p = sys_3comps
    M, v = quant.sys_min_val('s', [vars_p['x0'], vars_p['x1'], vars_p['x2']])

    np.testing.assert_array_equal(M.C, [[0, 6, 14, 0], [1, 5, 13, 1], [0, 0, 14, 1], [0, 5, 0, 1]])
    assert v.values==[0,1]
