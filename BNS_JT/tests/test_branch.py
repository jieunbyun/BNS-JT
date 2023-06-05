import numpy as np
import pandas as pd
import networkx as nx
import pdb
import pytest

#from BNS_JT.variable import Variable
from Trans.trans import get_arcs_length, do_branch, get_all_paths_and_times
from Trans.bnb_fns import bnb_sys, bnb_next_comp, bnb_next_state
from BNS_JT.branch import get_cmat, run_bnb, Branch, branch_and_bound
from BNS_JT import variable

np.set_printoptions(precision=3)
#pd.set_option.display_precision = 3

def test_run_bnb():
    info = {'path': [[2], [3, 1]],
            'time': np.array([0.0901, 0.2401]),
            'arcs': [1, 2, 3, 4, 5, 6],
            'max_state': 2
           }
    max_state = 2
    comp_max_states = (max_state*np.ones_like(info['arcs'])).tolist()
    #pdb.set_trace()
    branches = run_bnb(sys_fn=bnb_sys,
                       next_comp_fn=bnb_next_comp,
                       next_state_fn=bnb_next_state,
                       info=info,
                       comp_max_states=comp_max_states)

    assert len(branches)==5

    assert branches[0].down==[1, 1, 1, 1, 1, 1]
    assert branches[0].up==[1, 1, 2, 2, 2, 2]
    assert branches[0].is_complete==True
    assert branches[0].down_state==3-1
    assert branches[0].up_state==3-1
    assert branches[0].down_val==np.inf
    assert branches[0].up_val==np.inf

    assert branches[1].down==[1, 2, 1, 1, 1, 1]
    assert branches[1].up==[1, 2, 2, 2, 2, 2]
    assert branches[1].is_complete==True
    assert branches[1].down_state==1-1
    assert branches[1].up_state==1-1
    assert branches[1].down_val==0.0901
    assert branches[1].up_val==0.0901

    assert branches[2].down==[2, 2, 1, 1, 1, 1]
    assert branches[2].up, [2, 2, 2, 2, 2, 2]
    assert branches[2].is_complete==True
    assert branches[2].down_state==1-1
    assert branches[2].up_state==1-1
    assert branches[2].down_val==0.0901
    assert branches[2].up_val==0.0901

    assert branches[3].down==[2, 1, 1, 1, 1, 1]
    assert branches[3].up==[2, 1, 1, 2, 2, 2]
    assert branches[3].is_complete==True
    assert branches[3].down_state==3-1
    assert branches[3].up_state==3-1
    assert branches[3].down_val==np.inf
    assert branches[3].up_val==np.inf

    assert branches[4].down==[2, 1, 2, 1, 1, 1]
    assert branches[4].up==[2, 1, 2, 2, 2, 2]
    assert branches[4].is_complete==True
    assert branches[4].down_state==2-1
    assert branches[4].up_state==2-1
    assert branches[4].down_val==0.2401
    assert branches[4].up_val, 0.2401

def test_run_bnbs():
    info = {'path': [['2'], ['3', '1']],
            'time': np.array([0.0901, 0.2401]),
            'arcs': ['1', '2', '3', '4', '5', '6'],
            'max_state': 2
            }
    max_state = 2
    comp_max_states = (max_state*np.ones(len(info['arcs']))).tolist()

    #pdb.set_trace()
    branches = run_bnb(sys_fn=bnb_sys,
                       next_comp_fn=bnb_next_comp,
                       next_state_fn=bnb_next_state,
                       info=info,
                       comp_max_states=comp_max_states)

    assert branches[0].down==[1, 1, 1, 1, 1, 1]
    assert branches[0].up==[1, 1, 2, 2, 2, 2]
    assert branches[0].is_complete==True
    assert branches[0].down_state==3-1
    assert branches[0].up_state==3-1
    assert branches[0].down_val==np.inf
    assert branches[0].up_val==np.inf

    assert branches[1].down==[1, 2, 1, 1, 1, 1]
    assert branches[1].up==[1, 2, 2, 2, 2, 2]
    assert branches[1].is_complete==True
    assert branches[1].down_state==1-1
    assert branches[1].up_state==1-1
    assert branches[1].down_val==0.0901
    assert branches[1].up_val==0.0901

    assert branches[2].down==[2, 2, 1, 1, 1, 1]
    assert branches[2].up, [2, 2, 2, 2, 2, 2]
    assert branches[2].is_complete==True
    assert branches[2].down_state==1-1
    assert branches[2].up_state==1-1
    assert branches[2].down_val==0.0901
    assert branches[2].up_val==0.0901

    assert branches[3].down==[2, 1, 1, 1, 1, 1]
    assert branches[3].up==[2, 1, 1, 2, 2, 2]
    assert branches[3].is_complete==True
    assert branches[3].down_state==3-1
    assert branches[3].up_state==3-1
    assert branches[3].down_val==np.inf
    assert branches[3].up_val==np.inf

    assert branches[4].down==[2, 1, 2, 1, 1, 1]
    assert branches[4].up==[2, 1, 2, 2, 2, 2]
    assert branches[4].is_complete==True
    assert branches[4].down_state==2-1
    assert branches[4].up_state==2-1
    assert branches[4].down_val==0.2401
    assert branches[4].up_val, 0.2401

@pytest.fixture()
def setup_branch():

    branches = {}
    branches[0] = Branch(down=[1, 1, 1, 1, 1, 1],
    up=[1, 1, 2, 2, 2, 2],
    is_complete=True,
    down_state=3-1,
    up_state=3-1,
    down_val=np.inf,
    up_val=np.inf)

    branches[1] = Branch(down=[1, 2, 1, 1, 1, 1],
    up=[1, 2, 2, 2, 2, 2],
    is_complete=True,
    down_state=1-1,
    up_state=1-1,
    down_val=0.0901,
    up_val=0.0901)

    branches[2] = Branch(down=[2, 2, 1, 1, 1, 1],
    up= [2, 2, 2, 2, 2, 2],
    is_complete=True,
    down_state=1-1,
    up_state=1-1,
    down_val=0.0901,
    up_val=0.0901)

    branches[3] = Branch(down=[2, 1, 1, 1, 1, 1],
    up=[2, 1, 1, 2, 2, 2],
    is_complete=True,
    down_state=3-1,
    up_state=3-1,
    down_val=np.inf,
    up_val=np.inf)

    branches[4] = Branch(down=[2, 1, 2, 1, 1, 1],
    up=[2, 1, 2, 2, 2, 2],
    is_complete=True,
    down_state=2-1,
    up_state=2-1,
    down_val=0.2401,
    up_val= 0.2401)

    return list(branches.values())

def test_get_cmat1(setup_branch):

    branches = setup_branch

    varis = {}
    B = np.array([[1, 0], [0, 1], [1, 1]])
    for k in range(1, 7):
        varis[k] = variable.Variable(name=str(k), B=B, values=['Surv', 'Fail'])

    B_ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    varis[7] = variable.Variable(name='7', B=B_,
            values=[0.0901, 0.2401, np.inf])

    varis[8] = variable.Variable(name='8', B=B_,
            values=[0.0901, 0.2401, np.inf])

    varis[9] = variable.Variable(name='9', B=B_,
            values=[0.0943, 0.1761, np.inf])

    varis[10] = variable.Variable(name='10', B=B_,
            values=[0.0707, 0.1997, np.inf])

    for i in range(11, 15):
        varis[i] = variable.Variable(name=str(i), B=np.eye(2),
            values=['No disruption', 'Disruption'])

    info = {'path': [[2], [3, 1]],
            'time': np.array([0.0901, 0.2401]),
            'arcs': np.array([1, 2, 3, 4, 5, 6])
            }

    C = get_cmat(branches, [varis[i] for i in info['arcs']], False)

    expected = np.array([[3,2,2,3,3,3,3],
                         [1,2,1,3,3,3,3],
                         [1,1,1,3,3,3,3],
                         [3,1,2,2,3,3,3],
                         [2,1,2,1,3,3,3]]) - 1

    np.testing.assert_array_equal(C, expected)


def test_get_cmat2(setup_branch):
    #FIXME: test get_cmat with True flag

    info = {'path': [[2], [3, 1]],
            'time': np.array([0.0901, 0.2401]),
            'arcs': np.array([1, 2, 3, 4, 5, 6])
            }
    max_state = 2
    comp_max_states = (max_state*np.ones_like(info['arcs'])).tolist()

    branches = setup_branch

    varis = {}
    B = np.array([[1, 0], [0, 1], [1, 1]])
    for k in range(1, 7):
        varis[k] = variable.Variable(name=str(k), B=B, values=['Surv', 'Fail'])

    B_ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    varis[7] = variable.Variable(name='7', B=B_,
            values=[0.0901, 0.2401, np.inf])

    varis[8] = variable.Variable(name='8', B=B_,
            values=[0.0901, 0.2401, np.inf])

    varis[9] = variable.Variable(name='9', B=B_,
            values=[0.0943, 0.1761, np.inf])

    varis[10] = variable.Variable(name='10', B=B_,
            values=[0.0707, 0.1997, np.inf])

    for i in range(11, 15):
        varis[i] = variable.Variable(name=str(i), B=np.eye(2),
            values=['No disruption', 'Disruption'])

    C = get_cmat(branches, [varis[i] for i in info['arcs']], True)

    expected = np.array([[3,1,1,3,3,3,3],
                         [1,1,2,3,3,3,3],
                         [1,2,2,3,3,3,3],
                         [3,2,1,1,3,3,3],
                         [2,2,1,2,3,3,3]]) - 1

    np.testing.assert_array_equal(C, expected)


def test_get_cmat1s(setup_branch):

    info = {'path': [['2'], ['3', '1']],
            'time': np.array([0.0901, 0.2401]),
            'arcs': np.array(['1', '2', '3', '4', '5', '6'])
            }
    max_state = 2
    comp_max_states = (max_state*np.ones(len(info['arcs']))).tolist()

    branches = setup_branch

    varis = {}
    B = np.array([[1, 0], [0, 1], [1, 1]])
    for k in range(1, 7):
        varis[str(k)] = variable.Variable(name=str(k), B=B, values=['Surv', 'Fail'])

    B_ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    varis['7'] = variable.Variable(name='7', B=B_,
            values=[0.0901, 0.2401, np.inf])

    varis['8'] = variable.Variable(name='8', B=B_,
            values=[0.0901, 0.2401, np.inf])

    varis['9'] = variable.Variable(name='9', B=B_,
            values=[0.0943, 0.1761, np.inf])

    varis['10'] = variable.Variable(name='10', B=B_,
            values=[0.0707, 0.1997, np.inf])

    for i in range(11, 15):
        varis[str(i)] = variable.Variable(name=str(i), B=np.eye(2),
            values=['No disruption', 'Disruption'])
    #pdb.set_trace()
    C= get_cmat(branches, [varis[i] for i in info['arcs']], False)

    expected = np.array([[3,2,2,3,3,3,3],
                         [1,2,1,3,3,3,3],
                         [1,1,1,3,3,3,3],
                         [3,1,2,2,3,3,3],
                         [2,1,2,1,3,3,3]]) - 1

    np.testing.assert_array_equal(C, expected)


def test_branch_and_bound():

    path_time_idx =[([], np.inf, 0), (['e2'], 0.0901, 2), (['e1', 'e3'], 0.2401, 1)]

    # init
    arcs = [f'e{i}' for i in range(1, 7)]

    lower = {x: 0 for x in arcs}
    upper = {x: 1 for x in arcs}
    arc_condn = 1

    sb = branch_and_bound(path_time_idx, lower, upper, arc_condn)

    expected =[({'e1': 0, 'e2': 1, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 1, 'e2': 1, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 2, 2),
               ({'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 0, 'e2': 0, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 0, 0),
               ({'e1': 1, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 1, 'e2': 0, 'e3': 0, 'e4': 1, 'e5': 1, 'e6': 1}, 0, 0),
               ({'e1': 1, 'e2': 0, 'e3': 1, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 1, 'e2': 0, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 1, 1)]

    assert len(sb) == 4
    assert all([x in sb for x in expected])

    path_time_idx =[([], np.inf, 0), (['e2'], 0.0901, 2), (['e3', 'e1'], 0.2401, 1)]

    # init
    arcs = [f'e{i}' for i in range(1, 7)]

    lower = {x: 0 for x in arcs}
    upper = {x: 1 for x in arcs}
    arc_condn = 1

    sb = branch_and_bound(path_time_idx, lower, upper, arc_condn)

    expected =[({'e1': 0, 'e2': 1, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 1, 'e2': 1, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 2, 2),
               ({'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 1, 'e2': 0, 'e3': 0, 'e4': 1, 'e5': 1, 'e6': 1}, 0, 0),
               ({'e1': 0, 'e2': 0, 'e3': 1, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 0, 'e2': 0, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 0, 0),
               ({'e1': 1, 'e2': 0, 'e3': 1, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 1, 'e2': 0, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 1, 1)]

    assert len(sb) == 4
    assert all([x in sb for x in expected])


