from io import StringIO
import importlib
import numpy as np
import pytest

from BNS_JT import variable, cpm, utils

np.set_printoptions(precision=3)
#pd.set_option.display_precision = 3


@pytest.fixture
def setup_sys():
    '''
         -x1-
       -      -
    O-         --x3--D
       -      -
         -x2-

    '''
    values = ['survival', 'fail']
    x1= variable.Variable(**{'name': 'x1', 'B': [{0}, {1}, {0, 1}], 'values': values})
    x2= variable.Variable(**{'name': 'x2', 'B': [{0}, {1}, {0, 1}], 'values': values})
    x3= variable.Variable(**{'name': 'x3', 'B': [{0}, {1}, {0, 1}], 'values': values})
    x4= variable.Variable(**{'name': 'x4', 'B': [{0}, {1}, {0, 1}], 'values': values})

    variables = [x4, x1, x2, x3]
    no_child = 1
    C = np.array([[2, 3, 3, 2], [1, 1, 3, 1], [1, 2, 1, 1], [2, 2, 2, 1]]) - 1
    p = [1, 1, 1, 1]

    cpms = cpm.Cpm(**{'variables': variables,
                  'no_child': no_child,
                  'C': C,
                  'p': p})
    vars_ = {}
    return cpms, vars_

def test_init(setup_sys):

    cpms, vars_ = setup_sys
    assert isinstance(cpms, cpm.Cpm)
    for x in cpms.variables:
        assert isinstance(x, variable.Variable)


def test_read_nodes():
    file_node = StringIO("""
id,x,y
1,-2,3
2,-2,-3
3,2,-2
4,1,1
5,0,0
        """)
    node_coords = utils.read_nodes(file_node)

    expected = {1: (-2, 3),
                2: (-2, -3),
                3: (2, -2),
                4: (1, 1),
                5: (0, 0)}

    assert node_coords == expected


