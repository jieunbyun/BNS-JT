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

def test_find_state_from_vector1():

    varis = {'x1': variable.Variable( 'x1', [0,1]),
             'x2': variable.Variable( 'x2', [0,1,2] )}
    

    assert varis['x1'].find_state_from_vector([1,0]) == 0
    assert varis['x1'].find_state_from_vector([0,1]) == 1
    assert varis['x1'].find_state_from_vector([1,1]) == 2

    assert varis['x2'].find_state_from_vector([1,1,0]) == 3
    assert varis['x2'].find_state_from_vector([1,1,1]) == 6
    assert varis['x2'].find_state_from_vector([0,0,0]) == -1

def test_get_Bst_from_Bvec1():

    varis = {'x1': variable.Variable( 'x1', [0,1,2])}

    Bvec = np.array([[[1, 0, 0],
                      [1, 0, 0],
                      [1, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]],
                        
                     [[1, 0, 0],
                      [1, 0, 0],
                      [1, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]],
                        
                     [[0, 1, 1],
                      [0, 0, 0],
                      [0, 0, 0],
                      [0, 1, 0],
                      [0, 1, 0],
                      [0, 0, 1]],
                      
                     [[0, 1, 1],
                      [0, 0, 0],
                      [0, 0, 0],
                      [0, 1, 0],
                      [0, 1, 0],
                      [0, 0, 1]],

                     [[0, 1, 1],
                      [0, 0, 0],
                      [0, 0, 0],
                      [0, 1, 0],
                      [0, 1, 0],
                      [0, 0, 1]]])
    
    Bst = varis['x1'].get_Bst_from_Bvec( Bvec )
    np.testing.assert_array_equal(Bst, np.array([[0, 0, 0, -1, -1, -1],
                                                 [0, 0, 0, -1, -1, -1],
                                                 [5, -1, -1, 1, 1, 2],
                                                 [5, -1, -1, 1, 1, 2],
                                                 [5, -1, -1, 1, 1, 2]]))