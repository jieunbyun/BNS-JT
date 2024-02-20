from io import StringIO
import importlib
import numpy as np
import pytest

from BNS_JT import variable, cpm, utils

np.set_printoptions(precision=3)
#pd.set_option.display_precision = 3


@pytest.fixture
def setup_sys_three_edges():
    '''
         -x1-
       -      -
    O-         --x3--D
       -      -
         -x2-

    '''
    varis = {}
    values = ['fail', 'survival']
    varis['x1']= variable.Variable(**{'name': 'x1', 'B': [{0}, {1}, {0, 1}], 'values': values})
    varis['x2']= variable.Variable(**{'name': 'x2', 'B': [{0}, {1}, {0, 1}], 'values': values})
    varis['x3']= variable.Variable(**{'name': 'x3', 'B': [{0}, {1}, {0, 1}], 'values': values})
    varis['sys']= variable.Variable(**{'name': 'sys', 'B': [{0}, {1}, {0, 1}], 'values': values})

    no_child = 1
    C = np.array([[0, 2, 2, 0],
                  [1, 1, 2, 1],
                  [1, 0, 1, 1],
                  [0, 0, 0, 1]])
    p = [1, 1, 1, 1]

    cpms = cpm.Cpm(**{'variables': list(varis.values()),
                  'no_child': no_child,
                  'C': C,
                  'p': p})

    return cpms, varis


@pytest.fixture
def setup_sys_rbd():
    '''
    see Figure 2 from https://doi.org/10.1016/j.ress.2019.01.007
    '''
    varis = {}
    values = ['fail', 'survival']
    varis['x1']= variable.Variable(**{'name': 'x1', 'B': [{0}, {1}, {0, 1}], 'values': values})
    varis['x2']= variable.Variable(**{'name': 'x2', 'B': [{0}, {1}, {0, 1}], 'values': values})
    varis['x3']= variable.Variable(**{'name': 'x3', 'B': [{0}, {1}, {0, 1}], 'values': values})
    varis['x4']= variable.Variable(**{'name': 'x4', 'B': [{0}, {1}, {0, 1}], 'values': values})
    varis['x5']= variable.Variable(**{'name': 'x5', 'B': [{0}, {1}, {0, 1}], 'values': values})
    varis['x6']= variable.Variable(**{'name': 'x6', 'B': [{0}, {1}, {0, 1}], 'values': values})
    varis['x7']= variable.Variable(**{'name': 'x7', 'B': [{0}, {1}, {0, 1}], 'values': values})
    varis['x8']= variable.Variable(**{'name': 'x8', 'B': [{0}, {1}, {0, 1}], 'values': values})
    varis['source']= variable.Variable(**{'name': 'source', 'B': [{0}, {1}, {0, 1}], 'values': values})
    varis['sink']= variable.Variable(**{'name': 'sink', 'B': [{0}, {1}, {0, 1}], 'values': values})
    varis['sys']= variable.Variable(**{'name': 'sys', 'B': [{0}, {1}, {0, 1}], 'values': values})

    return varis


def test_init(setup_sys_three_edges):

    cpms, vars_ = setup_sys_three_edges
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


