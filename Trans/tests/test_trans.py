import unittest
import numpy as np
import pandas as pd
import networkx as nx


#from BNS_JT.variable import Variable
from Trans.trans import get_arcs_length, do_branch

np.set_printoptions(precision=3)
#pd.set_option.display_precision = 3

class Test(unittest.TestCase):
    """
    @classmethod
    def setUpClass(cls):

        B = np.array([[1, 0], [0, 1], [1, 1]])
        value = ['survival', 'fail']

        cls.kwargs = {'B': B,
                      'value': value,
                      }
    """
    def test_get_arcs_length(self):

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

    def test_get_all_paths_and_times(self):

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

        for org, dest in ODs:
            for _path in nx.all_simple_paths(G, org, dest):
                val = nx.path_weight(G, _path, weight='time')

                edges_path = []
                for x in list(zip(_path, _path[1:])):
                    edges_path.append(G[x[0]][x[1]]['label'])
                print(org, dest, edges_path, val)

    def test_do_branch1(self):
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

        self.assertEqual([[1, 3, 3, 1]], result)

        group = [[1, 1, 1, 1],
                 [1, 2, 1, 1],
                 [1, 2, 2, 1],
                 [1, 1, 2, 1]]

        result = do_branch(group, complete, id_any=3)

        self.assertEqual([[1, 3, 3, 1]], result)

    def test_do_branch2(self):
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
        self.assertSetEqual(expected, set(map(tuple, result)))


        group = [[2, 1, 1, 1],
                 [2, 1, 1, 2],
                 [1, 1, 1, 2]]

        result = do_branch(group, complete, id_any=3)
        expected = set(map(tuple,[[2, 1, 1, 3], [1, 1, 1, 2]]))
        self.assertSetEqual(expected, set(map(tuple, result)))

    def test_dummy(self):

        groups = [1, 1, 1, 1, 1, 1]
        arc_paths = [1, 1, 1, 1, 1, 1]
        path_times = [0.0901, 0.2401]
        #unfinished = 0
        #inf_time = np.inf

        #branches_surv = [[2], [1, 3], [1], []]
        #branches_fail = [[], [2], [2, 3], [1, 2]]
        #states = [1, 2, 3, 3]
        #times = [0.0901, 0.2401, np.inf, np.inf]

        # Travel times (systems)
        c = np.array([
        [1,3,1,3,3,3,3],
        [2,1,2,1,3,3,3],
        [3,1,2,2,3,3,3],
        [3,2,2,3,3,3,3]])

        variables = [7, 1, 2, 3, 4, 5, 6]
        no_child = 1
        p = [1, 1, 1, 1]

    def test_do_branch3(self):

        # node 5 to 1 
        comp_inds = [[2], [1, 3]]
        state = [1]
        all_surv_state = [0.0901]
        unfinished_state = [0]

        paths = [2]
        time = 0.0901
        state = 1


    """
    def test_B1(self):

        f_B = [[1, 2], [0, 1], [1, 1]]
        with self.assertRaises(AssertionError):
            _ = Variable(**{'B': f_B,
                         'value': self.kwargs['value']})

    def test_B2(self):

        f_B = [[1, 2]]
        with self.assertRaises(AssertionError):
            _ = Variable(**{'B': f_B,
                         'value': self.kwargs['value']})
    """


if __name__=='__main__':
    unittest.main()

