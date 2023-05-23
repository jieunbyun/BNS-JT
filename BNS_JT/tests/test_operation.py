from io import StringIO
import importlib
import numpy as np
import pytest

from BNS_JT.variable import Variable
from BNS_JT.cpm import Cpm
from BNS_JT.utils import read_nodes

np.set_printoptions(precision=3)
#pd.set_option.display_precision = 3


@pytest.fixture
def setup_bridge():
    '''
         -x1-
       -      -
    O-         --x3--D
       -      -
         -x2-

    '''
    B = [[1, 0], [0, 1], [1, 1]]
    values = ['survival', 'fail']
    x1= Variable(**{'name': 'x1', 'B': B, 'values': values})
    x2= Variable(**{'name': 'x2', 'B': B, 'values': values})
    x3= Variable(**{'name': 'x3', 'B': B, 'values': values})
    x4= Variable(**{'name': 'x4', 'B': B, 'values': values})

    variables = [x4, x1, x2, x3]
    no_child = 1
    C = np.array([[2, 3, 3, 2], [1, 1, 3, 1], [1, 2, 1, 1], [2, 2, 2, 1]]) - 1
    p = [1, 1, 1, 1]

    cpms = Cpm(**{'variables': variables,
                  'no_child': no_child,
                  'C': C,
                  'p': p})
    vars_ = {}
    return cpms, vars_

def test_init(setup_bridge):

    cpms, vars_ = setup_bridge
    assert isinstance(cpms, Cpm)
    for x in cpms.variables:
        assert isinstance(x, Variable)


def test_read_nodes():
    file_node = StringIO("""
id,x,y
1,-2,3
2,-2,-3
3,2,-2
4,1,1
5,0,0
        """)
    node_coords = read_nodes(file_node)

    expected = {1: (-2, 3),
                2: (-2, -3),
                3: (2, -2),
                4: (1, 1),
                5: (0, 0)}

    assert node_coords == expected


