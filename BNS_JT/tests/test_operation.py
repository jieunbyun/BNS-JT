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
    variables = [4, 1, 2, 3]
    no_child = 1
    C = np.array([[2, 3, 3, 2], [1, 1, 3, 1], [1, 2, 1, 1], [2, 2, 2, 1]])
    p = [1, 1, 1, 1]
    B = [[1, 0], [0, 1], [1, 1]]
    value = ['survival', 'fail']

    cpms = Cpm(**{'variables': variables,
                      'no_child': no_child,
                      'C': C,
                      'p': p})
    vars_ = {}
    vars_[1] = Variable(**{'B': B, 'value': value})
    vars_[2] = Variable(**{'B': B, 'value': value})
    vars_[3] = Variable(**{'B': B, 'value': value})
    vars_[4] = Variable(**{'B': B, 'value': value})

    return cpms, vars_

def test_init(setup_bridge):

    cpms, vars_ = setup_bridge
    assert isinstance(cpms, Cpm)
    assert isinstance(vars_[1], Variable)
    assert isinstance(vars_[2], Variable)
    assert isinstance(vars_[3], Variable)
    assert isinstance(vars_[4], Variable)


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


