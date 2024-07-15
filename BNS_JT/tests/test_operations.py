from io import StringIO
import importlib
import numpy as np
import pytest
import networkx as nx

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

    varis['x1']= variable.Variable(**{'name': 'x1', 'values': values})
    varis['x2']= variable.Variable(**{'name': 'x2', 'values': values})
    varis['x3']= variable.Variable(**{'name': 'x3', 'values': values})
    varis['sys']= variable.Variable(**{'name': 'sys', 'values': values})

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

    arcs = {'e1': ['source', 'x1'],
            'e2': ['source', 'x2'],
            'e3': ['source', 'x3'],
            'e4': ['source', 'x4'],
            'e5': ['x4', 'x5'],
            'e6': ['x5', 'x6'],
            'e7': ['x6', 'x7'],
            'e8': ['x7', 'x8'],
            'e9': ['x8', 'sink'],
            'e10': ['x1', 'x7'],
            'e11': ['x2', 'x7'],
            'e12': ['x3', 'x7'],
            }

    # nodes
    for k in range(1, 9):
        varis[f'x{k}'] = variable.Variable(**{'name': f'x{k}', 'values': values})

    # edges
    for k in range(1, 13):
        varis[f'e{k}'] = variable.Variable(**{'name': f'e{k}', 'values': values})

    varis['source']= variable.Variable(**{'name': 'source', 'values': values})
    varis['sink']= variable.Variable(**{'name': 'sink', 'values': values})
    varis['sys']= variable.Variable(**{'name': 'sys', 'values': values})

    G = nx.DiGraph()

    # edges
    for k, v in arcs.items():
        G.add_edge(v[0], v[1], label=k, key=k, weight=1)

    # nodes
    [G.add_node(f'x{i}', key=f'x{i}', label=f'x{i}') for i in range(1, 9)]
    G.add_node('source', key='source', label='source')
    G.add_node('sink', key='sink', label='sink')

    return varis, arcs, G


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


