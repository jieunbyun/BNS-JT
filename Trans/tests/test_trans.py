import numpy as np
import pandas as pd
import networkx as nx
import pdb

from Trans.trans import get_arcs_length, do_branch, get_all_paths_and_times

np.set_printoptions(precision=3)
#pd.set_option.display_precision = 3

def test_get_arcs_length():

    node_coords = {1: [-2, 3],
                   2: [-2, -3],
                   3: [2, -2],
                   4: [1, 1],
                   5: [0, 0]}

    arcs = {1: [1, 2],
            2: [1,5],
            3: [2,5],
            4: [3,4],
            5: [3,5],
            6: [4,5]}

    result = get_arcs_length(arcs, node_coords)

    expected = {1: 6.0,
                2: 3.6056,
                3: 3.6056,
                4: 3.1623,
                5: 2.8284,
                6: 1.4142}

    pd.testing.assert_series_equal(pd.Series(result), pd.Series(expected), rtol=1.0e-3)

def test_get_all_paths_and_times():

    arcs = {1: [1, 2],
            2: [1, 5],
            3: [2, 5],
            4: [3, 4],
            5: [3, 5],
            6: [4, 5]}

    arc_times_h = {1: 0.15, 2: 0.0901, 3: 0.0901, 4: 0.1054,
                   5: 0.0943, 6: 0.0707}

    G = nx.Graph()
    for k, x in arcs.items():
        G.add_edge(x[0], x[1], time=arc_times_h[k], label=k)

    ODs = [(5, 1), (5, 2), (5, 3), (5, 4)]

    path_time = get_all_paths_and_times(ODs, G)

    expected = {(5, 1): [([2], 0.0901),
                         ([3, 1], 0.2401)],
                (5, 2): [([2, 1], 0.2401),
                         ([3], 0.0901)],
                (5, 3): [([5], 0.0943),
                         ([6, 4], 0.1761)],
                (5, 4): [([5, 4], 0.1997),
                         ([6], 0.0707)],
                }

def test_do_branch1():
    # parallel system 
    #    (1)   2 (3)
    # 1              4
    #    (2)  3  (4)   
    # edge: 1: 1-2 (0.1)
    #       2: 1-3 (0.2)
    #       3: 2-4 (0.1)
    #       4: 3-4 (0.2)
    # 1: Ok, 2: Failure 3: Either     

    # 0.2
    group = [[1, 1, 1, 1],
             [1, 1, 2, 1],
             [1, 2, 1, 1],
             [1, 2, 2, 1]]

    complete = {x: (1, 2) for x in range(4)}

    result = do_branch(group, complete, id_any=3)

    assert result==[[1, 3, 3, 1]]

    group = [[1, 1, 1, 1],
             [1, 2, 1, 1],
             [1, 2, 2, 1],
             [1, 1, 2, 1]]

    result = do_branch(group, complete, id_any=3)

    assert result == [[1, 3, 3, 1]]

def test_do_branch2():
    # parallel system 
    #    (1)   2 (3)
    # 1              4
    #    (2)  3  (4)   
    # edge: 1: 1-2 (0.1)
    #       2: 1-3 (0.2)
    #       3: 2-4 (0.1)
    #       4: 3-4 (0.2)
    # 1: Ok, 2: Failure 3: Either     
    # result varies by order
    group = [[1, 1, 1, 2],
             [2, 1, 1, 1],
             [2, 1, 1, 2]]

    complete = {x: (1, 2) for x in range(4)}

    result = do_branch(group, complete, id_any=3)
    expected = set(map(tuple,[[3, 1, 1, 2], [2, 1, 1, 1]]))
    assert expected == set(map(tuple, result))


    group = [[2, 1, 1, 1],
             [2, 1, 1, 2],
             [1, 1, 1, 2]]

    result = do_branch(group, complete, id_any=3)
    expected = set(map(tuple,[[2, 1, 1, 3], [1, 1, 1, 2]]))
    assert expected==set(map(tuple, result))



