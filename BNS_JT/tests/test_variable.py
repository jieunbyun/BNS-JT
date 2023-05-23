import numpy as np
import pytest
from collections import namedtuple

from BNS_JT.variable import Variable

np.set_printoptions(precision=3)
#pd.set_option.display_precision = 3

def test_init():
    name = 'A'
    B = np.array([[1, 0], [0, 1], [1, 1]])
    value = ['survival', 'fail']

    var = {'name': name, 'B': B, 'values': value}
    a = Variable(**var)

    assert isinstance(a, Variable)
    np.testing.assert_array_equal(a.name, var['name'])
    np.testing.assert_array_equal(a.B, var['B'])
    np.testing.assert_array_equal(a.values, var['values'])

def test_B1():

    f_B = np.array([[1, 2], [0, 1], [1, 1]])
    with pytest.raises(AssertionError):
        _ = Variable(**{'name': 'A',
                        'B': f_B,
                        'values': ['T', 'F']})

def test_B2():

    f_B = np.array([[1, 2]])
    with pytest.raises(AssertionError):
        _ = Variable(**{'name': 'x',
                        'B': f_B,
                        'values': ['T', 'F']})

def test_eq1():
    name = 'A'
    B = np.array([[1, 0], [0, 1], [1, 1]])
    value = ['survival', 'fail']

    var = {'name': name, 'B': B, 'values': value}
    a = Variable(**var)
    b = Variable(**var)

    assert a == b

def test_eq2():
    name = 'A'
    B = np.array([[1, 0], [0, 1], [1, 1]])
    value = ['survival', 'fail']

    var = {'name': name, 'B': B, 'values': value}
    a = Variable(**var)

    b = Variable(**var)
    c = Variable(**var)
    _list = [b, c]

    assert a in _list

def test_eq3():
    name = 'A'
    B = np.array([[1, 0], [0, 1], [1, 1]])
    value = ['survival', 'fail']
    var = {'name': name, 'B': B, 'values': value}
    a = Variable(**var)

    A = [a, a, a]

    var = {'name': 'A', 'B': B, 'values': value}
    b = Variable(**var)

    var = {'name': 'C', 'B': B, 'values': value}
    c = Variable(**var)

    B = [b, c]

    result = [B.index(x) if x in B else False for x in A]

    assert result == [0, 0, 0]
