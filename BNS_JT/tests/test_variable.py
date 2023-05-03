import numpy as np
import pytest

from BNS_JT.variable import Variable

np.set_printoptions(precision=3)
#pd.set_option.display_precision = 3

def test_init():
    B = np.array([[1, 0], [0, 1], [1, 1]])
    value = ['survival', 'fail']

    var = {'B': B, 'value': value}
    a = Variable(**var)

    assert isinstance(a, Variable)
    np.testing.assert_array_equal(a.B, var['B'])
    np.testing.assert_array_equal(a.value, var['value'])

def test_B1():

    f_B = [[1, 2], [0, 1], [1, 1]]
    with pytest.raises(AssertionError):
        _ = Variable(**{'B': f_B,
                        'value': ['T', 'F']})

def test_B2():

    f_B = [[1, 2]]
    with pytest.raises(AssertionError):
        _ = Variable(**{'B': f_B,
                     'value': ['T', 'F']})


