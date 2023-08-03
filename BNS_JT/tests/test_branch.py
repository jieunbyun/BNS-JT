import numpy as np
import pandas as pd
import networkx as nx
import pdb
import time
import pytest
import json
from pathlib import Path
from dask.distributed import Client, LocalCluster, Variable, worker_client

from BNS_JT import variable, trans, bnb_fns, branch, config

HOME = Path(__file__).absolute().parent
PROJ = HOME.joinpath('../../')
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
    branches = branch.run_bnb(sys_fn=bnb_fns.bnb_sys,
                       next_comp_fn=bnb_fns.bnb_next_comp,
                       next_state_fn=bnb_fns.bnb_next_state,
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
    branches = branch.run_bnb(sys_fn=bnb_fns.bnb_sys,
                       next_comp_fn=bnb_fns.bnb_next_comp,
                       next_state_fn=bnb_fns.bnb_next_state,
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
    branches[0] = branch.Branch(down=[1, 1, 1, 1, 1, 1],
    up=[1, 1, 2, 2, 2, 2],
    is_complete=True,
    down_state=3-1,
    up_state=3-1,
    down_val=np.inf,
    up_val=np.inf)

    branches[1] = branch.Branch(down=[1, 2, 1, 1, 1, 1],
    up=[1, 2, 2, 2, 2, 2],
    is_complete=True,
    down_state=1-1,
    up_state=1-1,
    down_val=0.0901,
    up_val=0.0901)

    branches[2] = branch.Branch(down=[2, 2, 1, 1, 1, 1],
    up= [2, 2, 2, 2, 2, 2],
    is_complete=True,
    down_state=1-1,
    up_state=1-1,
    down_val=0.0901,
    up_val=0.0901)

    branches[3] = branch.Branch(down=[2, 1, 1, 1, 1, 1],
    up=[2, 1, 1, 2, 2, 2],
    is_complete=True,
    down_state=3-1,
    up_state=3-1,
    down_val=np.inf,
    up_val=np.inf)

    branches[4] = branch.Branch(down=[2, 1, 2, 1, 1, 1],
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

    C = branch.get_cmat(branches, [varis[i] for i in info['arcs']], False)

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

    C = branch.get_cmat(branches, [varis[i] for i in info['arcs']], True)

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
    C = branch.get_cmat(branches, [varis[i] for i in info['arcs']], False)

    expected = np.array([[3,2,2,3,3,3,3],
                         [1,2,1,3,3,3,3],
                         [1,1,1,3,3,3,3],
                         [3,1,2,2,3,3,3],
                         [2,1,2,1,3,3,3]]) - 1

    np.testing.assert_array_equal(C, expected)


def test_branch_and_bound():

    path_time_idx = [([], np.inf, 0), (['e2'], 0.0901, 2), (['e1', 'e3'], 0.2401, 1)]

    arc_condn = 1

    # init
    arcs = [f'e{i}' for i in range(1, 7)]

    lower = {x: 0 for x in arcs}
    upper = {x: 1 for x in arcs}

    fl = trans.eval_sys_state(path_time_idx, arcs_state=lower, arc_cond=1)
    fu = trans.eval_sys_state(path_time_idx, arcs_state=upper, arc_cond=1)

    # selecting a branch from sb such that fl /= fu
    bstars = [(lower, upper, fl, fu)]

    sb = branch.branch_and_bound(bstars, path_time_idx, arc_condn)

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

    fl = trans.eval_sys_state(path_time_idx, arcs_state=lower, arc_cond=1)
    fu = trans.eval_sys_state(path_time_idx, arcs_state=upper, arc_cond=1)

    # selecting a branch from sb such that fl /= fu
    bstars = [(lower, upper, fl, fu)]

    sb = branch.branch_and_bound(bstars, path_time_idx, arc_condn)

    expected =[({'e1': 0, 'e2': 1, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 1, 'e2': 1, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 2, 2),
               ({'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 1, 'e2': 0, 'e3': 0, 'e4': 1, 'e5': 1, 'e6': 1}, 0, 0),
               ({'e1': 0, 'e2': 0, 'e3': 1, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 0, 'e2': 0, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 0, 0),
               ({'e1': 1, 'e2': 0, 'e3': 1, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 1, 'e2': 0, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 1, 1)]

    assert len(sb) == 4
    assert all([x in sb for x in expected])


def test_get_cmat_from_branches():

    # variables
    variables = {}
    B = np.array([[1, 0], [0, 1], [1, 1]])
    for i in range(1, 7):
        variables[f'e{i}'] = variable.Variable(name=f'e{i}', B=B, values=['Fail', 'Surv'])

    branches =[({'e1': 0, 'e2': 1, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0},
                {'e1': 1, 'e2': 1, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 2, 2),
               ({'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0},
                {'e1': 0, 'e2': 0, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 0, 0),
               ({'e1': 1, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0},
                {'e1': 1, 'e2': 0, 'e3': 0, 'e4': 1, 'e5': 1, 'e6': 1}, 0, 0),
               ({'e1': 1, 'e2': 0, 'e3': 1, 'e4': 0, 'e5': 0, 'e6': 0},
                {'e1': 1, 'e2': 0, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 1, 1)]

    result = branch.get_cmat_from_branches(branches, variables)

    expected = np.array([[0,0,0,2,2,2,2],
                         [0,1,0,0,2,2,2],
                         [1,1,0,1,2,2,2],
                         [2,2,1,2,2,2,2]])

    np.testing.assert_array_equal(result, expected)

    # ('e3', 'e1') instead of ('e1', 'e3')
    branches =[({'e1': 0, 'e2': 1, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0},
                {'e1': 1, 'e2': 1, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 2, 2),
               ({'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0},
                {'e1': 1, 'e2': 0, 'e3': 0, 'e4': 1, 'e5': 1, 'e6': 1}, 0, 0),
               ({'e1': 0, 'e2': 0, 'e3': 1, 'e4': 0, 'e5': 0, 'e6': 0},
                {'e1': 0, 'e2': 0, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 0, 0),
               ({'e1': 1, 'e2': 0, 'e3': 1, 'e4': 0, 'e5': 0, 'e6': 0},
                {'e1': 1, 'e2': 0, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 1, 1)]

    result = branch.get_cmat_from_branches(branches, variables)

    expected = np.array([[0,2,0,0,2,2,2],
                         [0,0,0,1,2,2,2],
                         [1,1,0,1,2,2,2],
                         [2,2,1,2,2,2,2]])

    np.testing.assert_array_equal(result, expected)


@pytest.fixture()
def setup_client():

    cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    #client = Client(cluster)

    return cluster

@pytest.mark.skip('removed')
def test_branch_and_bound_dask(setup_client):

    cluster = setup_client

    # 0, 1, 2 corresponds to index of Variable.values
    path_time_idx =[([], np.inf, 0), (['e2'], 0.0901, 2), (['e1', 'e3'], 0.2401, 1)]
    # FIXME
    #path_time_idx =[(['e1', 'e3'], 0.2401, 1), ([], np.inf, 0), (['e2'], 0.0901, 2)] # not working 

    # init
    arcs = [f'e{i}' for i in range(1, 7)]

    lower = {x: 0 for x in arcs}  # Fail
    upper = {x: 1 for x in arcs}  # surv
    arc_condn = 1

    with Client(cluster) as client:
        branch.branch_and_bound_dask(path_time_idx, lower, upper, arc_condn, client, 's1')

    sb = branch.get_sb_saved_from_job(PROJ, 's1')

    expected =[({'e1': 0, 'e2': 1, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 1, 'e2': 1, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 2, 2),
               ({'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 0, 'e2': 0, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 0, 0),
               ({'e1': 1, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 1, 'e2': 0, 'e3': 0, 'e4': 1, 'e5': 1, 'e6': 1}, 0, 0),
               ({'e1': 1, 'e2': 0, 'e3': 1, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 1, 'e2': 0, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 1, 1)]

    assert len(sb) == 4
    assert all([x in sb for x in expected])

    # ('e3', 'e1') instead of ('e1', 'e3')
    path_time_idx =[([], np.inf, 0), (['e2'], 0.0901, 2), (['e3', 'e1'], 0.2401, 1)]

    with Client(cluster) as client:
        branch.branch_and_bound_dask(path_time_idx, lower, upper, arc_condn, client, 's2')


@pytest.mark.skip('removed')
def test_branch_and_bound_using_fn():

    # 0, 1, 2 corresponds to index of Variable.values
    path_time_idx =[([], np.inf, 0), (['e2'], 0.0901, 2), (['e1', 'e3'], 0.2401, 1)]
    # FIXME
    #path_time_idx =[(['e1', 'e3'], 0.2401, 1), ([], np.inf, 0), (['e2'], 0.0901, 2)] # not working 

    # init
    arcs = [f'e{i}' for i in range(1, 7)]

    lower = {x: 0 for x in arcs}  # Fail
    upper = {x: 1 for x in arcs}  # surv
    arc_condn = 1

    #pdb.set_trace()
    sb = branch.branch_and_bound_using_fn(path_time_idx, lower, upper, arc_condn)

    expected =[({'e1': 0, 'e2': 1, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 1, 'e2': 1, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 2, 2),
               ({'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 0, 'e2': 0, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 0, 0),
               ({'e1': 1, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 1, 'e2': 0, 'e3': 0, 'e4': 1, 'e5': 1, 'e6': 1}, 0, 0),
               ({'e1': 1, 'e2': 0, 'e3': 1, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 1, 'e2': 0, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 1, 1)]

    assert len(sb) == 4
    assert all([x in sb for x in expected])

    # ('e3', 'e1') instead of ('e1', 'e3')
    path_time_idx =[([], np.inf, 0), (['e2'], 0.0901, 2), (['e3', 'e1'], 0.2401, 1)]

    sb = branch.branch_and_bound_using_fn(path_time_idx, lower, upper, arc_condn)

    expected =[({'e1': 0, 'e2': 1, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 1, 'e2': 1, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 2, 2),
               ({'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 1, 'e2': 0, 'e3': 0, 'e4': 1, 'e5': 1, 'e6': 1}, 0, 0),
               ({'e1': 0, 'e2': 0, 'e3': 1, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 0, 'e2': 0, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 0, 0),
               ({'e1': 1, 'e2': 0, 'e3': 1, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 1, 'e2': 0, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 1, 1)]

    assert len(sb) == 4
    assert all([x in sb for x in expected])


@pytest.fixture()
def setup_rbd():

    # 0, 1, 2 corresponds to index of Variable.values
    path_time_idx = [([], np.inf, 0),
           (['e3', 'e12', 'e8', 'e9'], 0.30219999999999997, 4),
           (['e1', 'e10', 'e8', 'e9'], 0.3621, 3),
           (['e2', 'e11', 'e8', 'e9'], 0.40219999999999995, 2),
           (['e4', 'e5', 'e6', 'e7', 'e8', 'e9'], 0.48249999999999993, 1)  ]

    # init
    arcs = [f'e{i}' for i in range(1, 13)]

    lower = {x: 0 for x in arcs}  # Fail
    upper = {x: 1 for x in arcs}  # surv
    arc_cond = 1

    fl = trans.eval_sys_state(path_time_idx, lower, 1)
    fu = trans.eval_sys_state(path_time_idx, upper, 1)
    bstars = [(lower, upper, fl, fu)]

    return path_time_idx, bstars, arc_cond


@pytest.mark.skip('dask1 removed')
def test_branch_and_bound_using_rbd(setup_client, setup_rbd):

    cluster = setup_client

    # 0, 1, 2 corresponds to index of Variable.values
    path_time_idx, bstars, arc_condn = setup_rbd

    # init
    lower, upper, _, _ = bstars[0]

    #pdb.set_trace()
    sb = branch.branch_and_bound_using_fn(path_time_idx, lower, upper, arc_condn)

    varis = {}
    B = np.array([[1, 0], [0, 1], [1, 1]])
    for k in range(1, 13):
        varis[f'e{k}'] = variable.Variable(name=f'e{k}', B=B, values=['Surv', 'Fail'])

    C = branch.get_cmat_from_branches(sb, varis)
    C = C.astype(int)
    C = C[C[:, 0].argsort()]
    np.savetxt('./C_rbd.txt', C, fmt='%d')

    with Client(cluster) as client:
        branch.branch_and_bound_dask1(path_time_idx, lower, upper, arc_condn, client, key='rds')
    sb_dask = get_sb_saved_from_job(PROJ, key='rds')
    C = branch.get_cmat_from_branches(sb_dask, varis)
    C = C.astype(int)
    C = C[C[:, 0].argsort()]
    np.savetxt('./C_rbd_dask.txt', C, fmt='%d')


def test_branch_and_bound_using_rbd2(setup_client, setup_rbd):

    cluster = setup_client

    # 0, 1, 2 corresponds to index of Variable.values
    path_time_idx, bstars, arc_condn = setup_rbd

    # init
    #lower, upper, _, _ = bstars[0]

    #bstars = [(lower, upper, fl, fu)]
    #pdb.set_trace()

    with Client(cluster) as client:
        branch.branch_and_bound_dask(client, bstars, path_time_idx, arc_condn, key='rds2')

    sb_dask = branch.get_sb_saved_from_job(PROJ, key='rds2')

    varis = {}
    B = np.array([[1, 0], [0, 1], [1, 1]])
    for k in range(1, 13):
        varis[f'e{k}'] = variable.Variable(name=f'e{k}', B=B, values=['Surv', 'Fail'])

    C = branch.get_cmat_from_branches(sb_dask, varis)
    #C = C[C[:, 0].argsort()]
    np.savetxt('./C_rbd_dask2.txt', C, fmt='%d')


def test_branch_and_bound_using_rbd0(setup_rbd):

    path_time_idx, bstars, arc_condn = setup_rbd

    #lower, upper, _, _ = bstars[0]

    #pdb.set_trace()
    sb = branch.branch_and_bound(bstars, path_time_idx, arc_condn)

    varis = {}
    B = np.array([[1, 0], [0, 1], [1, 1]])
    for k in range(1, 13):
        varis[f'e{k}'] = variable.Variable(name=f'e{k}', B=B, values=['Surv', 'Fail'])

    C = branch.get_cmat_from_branches(sb, varis)
    C = C.astype(int)
    C = C[C[:, 0].argsort()]
    np.savetxt('./C_rbd_orig.txt', C, fmt='%d')


def test_get_arcs_given_bstar():

    bstars = branch.get_bstars_from_sb_dump(HOME.joinpath('./sb_dump_1.json'))
    bstar = bstars[0]

    with open(HOME.joinpath('../demos/SF/path_time_idx.json'), 'r') as fid:
        _dic = json.load(fid)

    path_time_idx = _dic['od1']

    result = branch.get_arcs_given_bstar(bstar, path_time_idx, 1)

    expected = ['e6_8', 'e8_9', 'e5_9', 'e4_5', 'e3_4', 'e3_12', 'e11_12']

    assert result == expected


def test_get_sb_given_arcs():

    bstars = branch.get_bstars_from_sb_dump(HOME.joinpath('./sb_dump_1.json'))
    bstar = bstars[0]

    with open(HOME.joinpath('../demos/SF/path_time_idx.json'), 'r') as fid:
        _dic = json.load(fid)

    path_time_idx = _dic['od1']

    arcs = ['e6_8', 'e8_9', 'e5_9', 'e4_5', 'e3_4', 'e3_12', 'e11_12']
    _path = ['e1_2', 'e2_6', 'e6_8', 'e8_9', 'e5_9', 'e4_5', 'e3_4', 'e3_12', 'e11_12']

    lower, upper, c_fl, c_fu = bstar
    sb = []
    tic = time.time()
    result = branch.get_sb_given_arcs(lower, upper, arcs, path_time_idx, c_fl, c_fu, 1, sb)
    print(f'elapsed: {time.time()-tic}')

    tic = time.time()
    expected = branch.fn_dummy(bstars[0], _path, 1, path_time_idx)
    print(f'elapsed: {time.time()-tic}')
    assert result == expected


#@pytest.mark.skip("NYI")
def test_branch_and_bound_dask_split(setup_client, setup_rbd):

    cluster = setup_client

    path_time_idx, bstars, arc_cond = setup_rbd

    key = 'rds3'
    with Client(cluster) as client:
        print(client.dashboard_link)
        g_arc_cond = Variable('arc_cond')
        g_arc_cond.set(arc_cond)

        branch.branch_and_bound_dask_split(client, bstars, path_time_idx, g_arc_cond, key, output_path=HOME)

    sb_dask = branch.get_sb_saved_from_job(HOME, key='rds3')

    assert len(sb_dask) == 77


#@pytest.mark.skip("NYI")
def test_branch_and_bound_dask4(setup_client, setup_rbd):

    cluster = setup_client

    path_time_idx, _, arc_cond = setup_rbd

    bstars = branch.get_bstars_from_sb_dump(HOME.joinpath('./sb_dump_rds2_2.json'))

    key = 'rds4'
    with Client(cluster) as client:
        print(client.dashboard_link)
        g_arc_cond = Variable('arc_cond')
        g_arc_cond.set(arc_cond)

        branch.branch_and_bound_dask_split(client, bstars, path_time_idx, g_arc_cond, key, output_path=HOME)

    sb_dask = branch.get_sb_saved_from_sb_dump(HOME.joinpath(f'./sb_dump_{key}_1.json'))

    assert len(sb_dask) == 28


#@pytest.mark.skip('FIXME')
def test_branch_and_bound_new3(setup_rbd):

    path_time_idx, bstars, arc_cond = setup_rbd

    #pdb.set_trace()
    key = 'new3'

    branch.branch_and_bound_new3(bstars, path_time_idx, arc_cond, key)

    sb_dask = branch.get_sb_saved_from_job(PROJ, key='new3')

    assert len(sb_dask) == 77


def fib2(n):
    if n==0:
        return (0, [0])
    elif n==1:
        return (1, [0, 1])
    else:
        with worker_client() as client:
            a_f = client.submit(fib2, n-1)
            b_f = client.submit(fib2, n-2)
            a, b = client.gather([a_f, b_f])
        return (a[0]+b[0], a[1] + [a[0]+b[0]])


def test_dask_fib(setup_client):

    cluster = setup_client

    with Client(cluster) as client:
        future = client.submit(fib2, 10)
        result = client.gather(future)

    assert result[0] == 55
    assert result[1] == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]


def test_get_set_branches_no_iteration(setup_client):

    cluster = setup_client

    # using road
    cfg = config.Config(HOME.joinpath('./config_roads.json'))

    path_times = trans.get_all_paths_and_times(cfg.infra['ODs'].values(), cfg.infra['G'], key='time')

    expected =[({'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 1, 'e2': 0, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 0, 1),
               ({'e1': 0, 'e2': 1, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 1, 'e2': 1, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 2, 2)]

    k, v = 'od1', ('n5', 'n1')
    arc_cond = 1
    values = [np.inf] + sorted([y for _, y in path_times[v]], reverse=True)
    varis = variable.Variable(name=k, B=np.eye(len(values)), values=values)
    path_time_idx = trans.get_path_time_idx(path_times[v], varis)

    lower = {k: 0 for k, _ in cfg.infra['edges'].items()}
    upper = {k: 1 for k, _ in cfg.infra['edges'].items()}

    fl = trans.eval_sys_state(path_time_idx, lower, arc_cond)
    fu = trans.eval_sys_state(path_time_idx, upper, arc_cond)

    bstar = (lower, upper, fl, fu)

    arcs = branch.get_arcs_given_bstar(bstar, path_time_idx, arc_cond)

    with Client(cluster) as client:
        sb = branch.get_set_branches_no_iteration(bstar, arcs, path_time_idx, arc_cond)
        sb = client.gather(sb)
    assert len(sb) == 2

    assert sb == expected

    sb1 = branch.get_set_of_branches(bstar, arcs, path_time_idx, arc_cond)
    assert sb == sb1

