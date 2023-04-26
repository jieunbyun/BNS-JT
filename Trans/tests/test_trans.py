import unittest
import numpy as np
import pandas as pd
import networkx as nx
import pdb

#from BNS_JT.variable import Variable
from Trans.trans import get_arcs_length, do_branch, get_all_paths_and_times
from Trans.bnb_fns import bnb_sys, bnb_next_comp, bnb_next_state
from BNS_JT.branch import get_cmat
from BNS_JT import variable
from run_bnb import run_bnb

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

    def test_bnb_sys1(self):

        comp_states = [1, 1, 1, 1, 1, 1]
        info = {'path': [[2], [3, 1]],
                'time': np.array([0.0901, 0.2401]),
                'arcs': np.array([1, 2, 3, 4, 5, 6]),
                }
        #pdb.set_trace()
        state, val, result = bnb_sys(comp_states, info)

        self.assertEqual(state, 3)
        self.assertEqual(val, np.inf)
        self.assertEqual(result, {'path': []})

    def test_bnb_sys2(self):

        comp_states = [1, 2, 1, 1, 1, 1]
        info = {'path': [[2], [3, 1]],
                'time': np.array([0.0901, 0.2401]),
                'arcs': np.array([1, 2, 3, 4, 5, 6]),
                }
        state, val, result = bnb_sys(comp_states, info)

        self.assertEqual(state, 1)
        self.assertEqual(val, 0.0901)
        self.assertEqual(result, {'path': [2]})

    def test_bnb_sys3(self):

        comp_states = [2, 2, 2, 2, 2, 2]
        info = {'path': [[2], [3, 1]],
                'time': np.array([0.0901, 0.2401]),
                'arcs': np.array([1, 2, 3, 4, 5, 6]),
                }
        #pdb.set_trace()
        state, val, result = bnb_sys(comp_states, info)

        self.assertEqual(state, 1)
        self.assertEqual(val, 0.0901)
        self.assertEqual(result, {'path': [2]})

    def test_bnb_sys4(self):

        comp_states = [1, 2, 2, 2, 2, 2]
        info = {'path': [[2], [3, 1]],
                'time': np.array([0.0901, 0.2401]),
                'arcs': np.array([1, 2, 3, 4, 5, 6]),
                }
        #pdb.set_trace()
        state, val, result = bnb_sys(comp_states, info)

        self.assertEqual(state, 1)
        self.assertEqual(val, 0.0901)
        self.assertEqual(result, {'path': [2]})

    def test_bnb_sys5(self):

        comp_states = [1, 1, 2, 2, 2, 2]
        info = {'path': [[2], [3, 1]],
                'time': np.array([0.0901, 0.2401]),
                'arcs': np.array([1, 2, 3, 4, 5, 6]),
                }
        #pdb.set_trace()
        state, val, result = bnb_sys(comp_states, info)

        self.assertEqual(state, 3)
        self.assertEqual(val, np.inf)
        self.assertEqual(result, {'path': []})

    def test_bnb_next_comp1(self):

        cand_comps = [1, 2, 3, 4, 5, 6]
        down_res = []
        up_res = [3, 1]
        info = {'path': [[2], [3, 1]],
                'time': np.array([0.0901, 0.2401]),
                'arcs': np.array([1, 2, 3, 4, 5, 6]),
                }

        next_comp = bnb_next_comp(cand_comps, down_res, up_res, info)

        self.assertEqual(next_comp, 1)

    def test_bnb_next_comp2(self):

        cand_comps = [2, 3, 4, 5, 6]
        down_res = []
        up_res = [2]
        info = {'path': [[2], [3, 1]],
                'time': np.array([0.0901, 0.2401]),
                'arcs': np.array([1, 2, 3, 4, 5, 6]),
                }

        next_comp = bnb_next_comp(cand_comps, down_res, up_res, info)

        self.assertEqual(next_comp, 2)

    def test_bnb_next_comp3(self):

        cand_comps = [3, 4, 5, 6]
        down_res = []
        up_res = [3, 1]
        info = {'path': [[2], [3, 1]],
                'time': np.array([0.0901, 0.2401]),
                'arcs': np.array([1, 2, 3, 4, 5, 6]),
                }

        next_comp = bnb_next_comp(cand_comps, down_res, up_res, info)

        self.assertEqual(next_comp, 3)

    def test_run_bnb(self):
        info = {'path': [[2], [3, 1]],
                'time': np.array([0.0901, 0.2401]),
                'arcs': np.array([1, 2, 3, 4, 5, 6])
                }
        max_state = 2
        comp_max_states = (max_state*np.ones_like(info['arcs'])).tolist()

        branches = run_bnb(sys_fn=bnb_sys,
                           next_comp_fn=bnb_next_comp,
                           next_state_fn=bnb_next_state,
                           info=info,
                           comp_max_states=comp_max_states)

        self.assertEqual(len(branches), 5)

        self.assertEqual(branches[0].down, [1, 1, 1, 1, 1, 1])
        self.assertEqual(branches[0].up, [1, 1, 2, 2, 2, 2])
        self.assertEqual(branches[0].is_complete, True)
        self.assertEqual(branches[0].down_state, 3)
        self.assertEqual(branches[0].up_state, 3)
        self.assertEqual(branches[0].down_val, np.inf)
        self.assertEqual(branches[0].up_val, np.inf)

        self.assertEqual(branches[1].down, [1, 2, 1, 1, 1, 1])
        self.assertEqual(branches[1].up, [1, 2, 2, 2, 2, 2])
        self.assertEqual(branches[1].is_complete, True)
        self.assertEqual(branches[1].down_state, 1)
        self.assertEqual(branches[1].up_state, 1)
        self.assertEqual(branches[1].down_val, 0.0901)
        self.assertEqual(branches[1].up_val, 0.0901)

        self.assertEqual(branches[2].down, [2, 2, 1, 1, 1, 1])
        self.assertEqual(branches[2].up, [2, 2, 2, 2, 2, 2])
        self.assertEqual(branches[2].is_complete, True)
        self.assertEqual(branches[2].down_state, 1)
        self.assertEqual(branches[2].up_state, 1)
        self.assertEqual(branches[2].down_val, 0.0901)
        self.assertEqual(branches[2].up_val, 0.0901)

        self.assertEqual(branches[3].down, [2, 1, 1, 1, 1, 1])
        self.assertEqual(branches[3].up, [2, 1, 1, 2, 2, 2])
        self.assertEqual(branches[3].is_complete, True)
        self.assertEqual(branches[3].down_state, 3)
        self.assertEqual(branches[3].up_state, 3)
        self.assertEqual(branches[3].down_val, np.inf)
        self.assertEqual(branches[3].up_val, np.inf)

        self.assertEqual(branches[4].down, [2, 1, 2, 1, 1, 1])
        self.assertEqual(branches[4].up, [2, 1, 2, 2, 2, 2])
        self.assertEqual(branches[4].is_complete, True)
        self.assertEqual(branches[4].down_state, 2)
        self.assertEqual(branches[4].up_state, 2)
        self.assertEqual(branches[4].down_val, 0.2401)
        self.assertEqual(branches[4].up_val, 0.2401)

    def test_get_cmat(self):

        info = {'path': [[2], [3, 1]],
                'time': np.array([0.0901, 0.2401]),
                'arcs': np.array([1, 2, 3, 4, 5, 6])
                }
        max_state = 2
        comp_max_states = (max_state*np.ones_like(info['arcs'])).tolist()

        branches = run_bnb(sys_fn=bnb_sys,
                           next_comp_fn=bnb_next_comp,
                           next_state_fn=bnb_next_state,
                           info=info,
                           comp_max_states=comp_max_states)

        varis = {}
        B = np.array([[1, 0], [0, 1], [1, 1]])
        for k in range(1, 7):
            varis[k] = variable.Variable(B=B, value=['Surv', 'Fail'])

        B_ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        varis[7] = variable.Variable(B=B_,
                value=[0.0901, 0.2401, np.inf])

        varis[8] = variable.Variable(B=B_,
                value=[0.0901, 0.2401, np.inf])

        varis[9] = variable.Variable(B=B_,
                value=[0.0943, 0.1761, np.inf])

        varis[10] = variable.Variable(B=B_,
                value=[0.0707, 0.1997, np.inf])

        for i in range(11, 15):
            varis[i] = variable.Variable(B=np.eye(2),
                value=['No disruption', 'Disruption'])

        C, varis = get_cmat(branches, info['arcs'], varis, False)

        expected = np.array([[3,2,2,3,3,3,3],
                             [1,2,1,3,3,3,3],
                             [1,1,1,3,3,3,3],
                             [3,1,2,2,3,3,3],
                             [2,1,2,1,3,3,3]])

        np.testing.assert_array_equal(C, expected)


if __name__=='__main__':
    unittest.main()

