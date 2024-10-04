import numpy as np
import pytest
from collections import namedtuple

from BNS_JT import variable


np.set_printoptions(precision=3)
#pd.set_option.display_precision = 3

def compare_list_of_sets(a, b):

    return set([tuple(x) for x in a]) == set([tuple(x) for x in b])


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
    a.update_B()

    assert isinstance(a, variable.Variable)
    np.testing.assert_array_equal(a.name, name)
    np.testing.assert_array_equal(a.B, [{0}, {1}, {0, 1}])
    np.testing.assert_array_equal(a.values, value)

@pytest.mark.skip('FIXME: what this is meant')
def test_init3():

    name = 'A'
    a = variable.Variable(name)
    value = ['survival', 'fail']
    B = [{0}, {1},  {0, 1}]

    # should define value first
    with pytest.raises(AttributeError):
        a.B = B

@pytest.mark.skip('FIXME: what this is meant')
def test_init4():

    name = 'A'
    a = variable.Variable(name)
    value = ['survival', 'fail']
    B = [{0}, {1}, {0, 2}] # should be {0, 1}
    a.values = value

    with pytest.raises(AttributeError):
        a.B = B


@pytest.mark.skip('FIXME: what this is meant')
def test_init5():

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


def test_get_composite_state1():

    varis = {}
    varis['e1'] = variable.Variable(name='e1', values=[1.5, 0.3, 0.15])

    states = [1, 2]
    result = variable.get_composite_state(varis['e1'], states)
    expected = [{0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 2}]
    assert compare_list_of_sets(result[0].B, expected)
    assert result[1] == 5


def test_get_composite_state2():

    #od_pair, arcs, varis = main_sys
    varis = {}
    varis['e1'] = variable.Variable(name='e1', values=[1.5, 0.3, 0.15])

    states = [1, 2]
    result = variable.get_composite_state(varis['e1'], states)

    expected = [{0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 2}]
    #expected = [{0}, {1}, {2}, {1, 2}]
    assert compare_list_of_sets(result[0].B, expected)
    assert result[1] == 5


def test_get_composite_state3():

    #od_pair, arcs, varis = main_sys
    varis = {}
    varis['e1'] = variable.Variable(name='e1', values=[1.5, 0.3, 0.15])
    states = [0, 2]
    result = variable.get_composite_state(varis['e1'], states)

    expected = [{0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 2}]
    #expected = [{0}, {1}, {2}, {0, 2}]
    assert compare_list_of_sets(result[0].B, expected)
    assert result[1] == 4


def test_get_state1():

    varis = {'x1': variable.Variable( 'x1', values=[0,1]),
             'x2': variable.Variable( 'x2', [0,1,2,3]),
             'x3': variable.Variable( 'x3', np.arange(20).tolist())}
    
    assert varis['x1'].get_state( {0,1} ) == 2
    assert varis['x2'].get_state( {0,1,2} ) == 10
    assert varis['x3'].get_state( {3,4,5} ) == 670

def test_get_set1():

    varis = {'x1': variable.Variable( 'x1', values=[0,1]),
             'x2': variable.Variable( 'x2', [0,1,2,3]),
             'x3': variable.Variable( 'x3', np.arange(20).tolist())}
    
    assert varis['x1'].get_set( 2 ) == {0,1}
    assert varis['x2'].get_set( 10 ) == {0,1,2}
    assert varis['x3'].get_set( 670 ) == {3,4,5}