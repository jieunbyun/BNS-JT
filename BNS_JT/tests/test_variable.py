import numpy as np
import pytest
from collections import namedtuple

from BNS_JT import variable

np.set_printoptions(precision=3)
#pd.set_option.display_precision = 3

def test_init1():
    name = 'A'
    B = [{0}, {1},  {0, 1}]
    value = ['survival', 'fail']

    var = {'name': name, 'values': value}
    a = variable.Variable(**var)

    assert isinstance(a, variable.Variable)
    np.testing.assert_array_equal(a.name, name)
    np.testing.assert_array_equal(a.B, B)
    np.testing.assert_array_equal(a.values, value)


def test_init2():
    name = 'A'
    a = variable.Variable(name)
    value = ['survival', 'fail']
    a.values = value

    assert isinstance(a, variable.Variable)
    np.testing.assert_array_equal(a.name, name)
    np.testing.assert_array_equal(a.B, [{0}, {1}, {0, 1}])
    np.testing.assert_array_equal(a.values, value)


def test_init3():

    name = 'A'
    a = variable.Variable(name)
    value = ['survival', 'fail']
    B = [{0}, {1},  {0, 1}]

    # should define value first
    with pytest.raises(AttributeError):
        a.B = B


def test_init4():

    name = 'A'
    a = variable.Variable(name)
    value = ['survival', 'fail']
    B = [{0}, {1}, {0, 2}] # should be {0, 1}
    a.values = value

    with pytest.raises(AttributeError):
        a.B = B


def test_init4():

    name = 'A'
    a = variable.Variable(name)
    value = ['survival', 'fail']
    B = [{0}, {1}, {0, 1}, {0, 1}] #  max. len of B == 3
    a.values = value

    with pytest.raises(AttributeError):
        a.B = B


def test_B1():

    f_B = [{0}, {1},  {2, 1}]
    with pytest.raises(TypeError):
        _ = variable.Variable(**{'name': 'A',
                        'B': f_B,
                        'values': ['T', 'F']})


def test_B2():

    f_B = [{2}]
    with pytest.raises(TypeError):
        _ = variable.Variable(**{'name': 'x',
                        'B': f_B,
                        'values': ['T', 'F']})


def test_eq1():
    name = 'A'
    B = [{0}, {1},  {0, 1}]
    value = ['survival', 'fail']

    var = {'name': name, 'values': value}
    a = variable.Variable(**var)
    b = variable.Variable(**var)

    assert a == b


def test_eq2():
    name = 'A'
    B = [{0}, {1},  {0, 1}]
    value = ['survival', 'fail']

    var = {'name': name, 'values': value}
    a = variable.Variable(**var)

    b = variable.Variable(**var)
    c = variable.Variable(**var)
    _list = [b, c]

    assert a in _list


def test_eq3():
    name = 'A'
    B = [{0}, {1},  {0, 1}]
    value = ['survival', 'fail']
    var = {'name': name, 'values': value}
    a = variable.Variable(**var)

    A = [a, a, a]

    var = {'name': 'A', 'values': value}
    b = variable.Variable(**var)

    var = {'name': 'C', 'values': value}
    c = variable.Variable(**var)

    B = [b, c]

    result = [B.index(x) if x in B else False for x in A]

    assert result == [0, 0, 0]
