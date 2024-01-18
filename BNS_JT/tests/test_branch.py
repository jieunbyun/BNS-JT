import pandas as pd
import numpy as np
import networkx as nx
import pdb
import time
import pytest
import json
from pathlib import Path
#from dask.distributed import Client, LocalCluster, Variable, worker_client

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
    assert branches[0].names == info['arcs']

    assert branches[1].down==[1, 2, 1, 1, 1, 1]
    assert branches[1].up==[1, 2, 2, 2, 2, 2]
    assert branches[1].is_complete==True
    assert branches[1].down_state==1-1
    assert branches[1].up_state==1-1
    assert branches[1].down_val==0.0901
    assert branches[1].up_val==0.0901
    assert branches[1].names == info['arcs']

    assert branches[2].down==[2, 2, 1, 1, 1, 1]
    assert branches[2].up, [2, 2, 2, 2, 2, 2]
    assert branches[2].is_complete==True
    assert branches[2].down_state==1-1
    assert branches[2].up_state==1-1
    assert branches[2].down_val==0.0901
    assert branches[2].up_val==0.0901
    assert branches[2].names == info['arcs']

    assert branches[3].down==[2, 1, 1, 1, 1, 1]
    assert branches[3].up==[2, 1, 1, 2, 2, 2]
    assert branches[3].is_complete==True
    assert branches[3].down_state==3-1
    assert branches[3].up_state==3-1
    assert branches[3].down_val==np.inf
    assert branches[3].up_val==np.inf
    assert branches[3].names == info['arcs']

    assert branches[4].down==[2, 1, 2, 1, 1, 1]
    assert branches[4].up==[2, 1, 2, 2, 2, 2]
    assert branches[4].is_complete==True
    assert branches[4].down_state==2-1
    assert branches[4].up_state==2-1
    assert branches[4].down_val==0.2401
    assert branches[4].up_val, 0.2401
    assert branches[4].names == info['arcs']

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
    assert branches[0].names == info['arcs']

    assert branches[1].down==[1, 2, 1, 1, 1, 1]
    assert branches[1].up==[1, 2, 2, 2, 2, 2]
    assert branches[1].is_complete==True
    assert branches[1].down_state==1-1
    assert branches[1].up_state==1-1
    assert branches[1].down_val==0.0901
    assert branches[1].up_val==0.0901
    assert branches[1].names == info['arcs']

    assert branches[2].down==[2, 2, 1, 1, 1, 1]
    assert branches[2].up, [2, 2, 2, 2, 2, 2]
    assert branches[2].is_complete==True
    assert branches[2].down_state==1-1
    assert branches[2].up_state==1-1
    assert branches[2].down_val==0.0901
    assert branches[2].up_val==0.0901
    assert branches[2].names == info['arcs']

    assert branches[3].down==[2, 1, 1, 1, 1, 1]
    assert branches[3].up==[2, 1, 1, 2, 2, 2]
    assert branches[3].is_complete==True
    assert branches[3].down_state==3-1
    assert branches[3].up_state==3-1
    assert branches[3].down_val==np.inf
    assert branches[3].up_val==np.inf
    assert branches[3].names == info['arcs']

    assert branches[4].down==[2, 1, 2, 1, 1, 1]
    assert branches[4].up==[2, 1, 2, 2, 2, 2]
    assert branches[4].is_complete==True
    assert branches[4].down_state==2-1
    assert branches[4].up_state==2-1
    assert branches[4].down_val==0.2401
    assert branches[4].up_val, 0.2401
    assert branches[4].names == info['arcs']


@pytest.fixture()
def setup_branch():

    names = [1, 2, 3, 4, 5, 6]

    branches = {}
    branches[0] = branch.Branch_old(down=[1, 1, 1, 1, 1, 1],
    up=[1, 1, 2, 2, 2, 2],
    is_complete=True,
    down_state=3-1,
    up_state=3-1,
    down_val=np.inf,
    up_val=np.inf,
    names=names)

    branches[1] = branch.Branch_old(down=[1, 2, 1, 1, 1, 1],
    up=[1, 2, 2, 2, 2, 2],
    is_complete=True,
    down_state=1-1,
    up_state=1-1,
    down_val=0.0901,
    up_val=0.0901,
    names=names)

    branches[2] = branch.Branch_old(down=[2, 2, 1, 1, 1, 1],
    up= [2, 2, 2, 2, 2, 2],
    is_complete=True,
    down_state=1-1,
    up_state=1-1,
    down_val=0.0901,
    up_val=0.0901,
    names=names)

    branches[3] = branch.Branch_old(down=[2, 1, 1, 1, 1, 1],
    up=[2, 1, 1, 2, 2, 2],
    is_complete=True,
    down_state=3-1,
    up_state=3-1,
    down_val=np.inf,
    up_val=np.inf,
    names=names)

    branches[4] = branch.Branch_old(down=[2, 1, 2, 1, 1, 1],
    up=[2, 1, 2, 2, 2, 2],
    is_complete=True,
    down_state=2-1,
    up_state=2-1,
    down_val=0.2401,
    up_val= 0.2401,
    names=names)

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


def test_branch_and_bound_org():

    path_time_idx = [([], np.inf, 0), (['e2'], 0.0901, 2), (['e1', 'e3'], 0.2401, 1)]

    arc_cond = 1

    # init
    arcs = [f'e{i}' for i in range(1, 7)]

    lower = {x: 0 for x in arcs}
    upper = {x: 1 for x in arcs}

    fl = trans.eval_sys_state(path_time_idx, arcs_state=lower, arc_cond=1)
    fu = trans.eval_sys_state(path_time_idx, arcs_state=upper, arc_cond=1)

    # selecting a branch from sb such that fl /= fu
    bstars = [(lower, upper, fl, fu)]

    sb = branch.branch_and_bound_org(bstars, path_time_idx, arc_cond)

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
    arc_cond = 1

    fl = trans.eval_sys_state(path_time_idx, arcs_state=lower, arc_cond=1)
    fu = trans.eval_sys_state(path_time_idx, arcs_state=upper, arc_cond=1)

    # selecting a branch from sb such that fl /= fu
    bstars = [(lower, upper, fl, fu)]

    sb = branch.branch_and_bound_org(bstars, path_time_idx, arc_cond)

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


@pytest.mark.skip('NODASK')
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
    arc_cond = 1

    with Client(cluster) as client:
        branch.branch_and_bound_dask(path_time_idx, lower, upper, arc_cond, client, 's1')

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
        branch.branch_and_bound_dask(path_time_idx, lower, upper, arc_cond, client, 's2')


def test_branch_and_bound():

    # 0, 1, 2 corresponds to index of Variable.values
    path_time_idx =[([], np.inf, 0), (['e2'], 0.0901, 2), (['e1', 'e3'], 0.2401, 1)]
    # FIXME
    #path_time_idx =[(['e1', 'e3'], 0.2401, 1), ([], np.inf, 0), (['e2'], 0.0901, 2)] # not working 

    # init
    arcs = [f'e{i}' for i in range(1, 7)]

    lower = {x: 0 for x in arcs}  # Fail
    upper = {x: 1 for x in arcs}  # surv
    arc_cond = 1

    bstars = [(lower, upper, 0, 2)]

    branch.branch_and_bound(bstars, path_time_idx, arc_cond, output_path=HOME, key='test')

    sb = branch.get_sb_saved_from_job(HOME, 'test')

    expected =[({'e1': 0, 'e2': 1, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 1, 'e2': 1, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 2, 2),
               ({'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 0, 'e2': 0, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 0, 0),
               ({'e1': 1, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 1, 'e2': 0, 'e3': 0, 'e4': 1, 'e5': 1, 'e6': 1}, 0, 0),
               ({'e1': 1, 'e2': 0, 'e3': 1, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 1, 'e2': 0, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 1, 1)]

    assert len(sb) == 4
    assert all([x in sb for x in expected])

    # ('e3', 'e1') instead of ('e1', 'e3')
    path_time_idx =[([], np.inf, 0), (['e2'], 0.0901, 2), (['e3', 'e1'], 0.2401, 1)]
    bstars = [(lower, upper, 0, 2)]

    branch.branch_and_bound(bstars, path_time_idx, arc_cond, output_path=HOME, key='test')

    sb = branch.get_sb_saved_from_job(HOME, 'test')

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

    output_path = HOME

    key = 'rbd'

    return path_time_idx, bstars, arc_cond, output_path, key

@pytest.mark.skip('NODASK')
def test_branch_and_bound_using_rbd(setup_client, setup_rbd):

    # 0, 1, 2 corresponds to index of Variable.values
    path_time_idx, bstars, arc_cond, output_path, key = setup_rbd

    varis = {}
    B = np.array([[1, 0], [0, 1], [1, 1]])
    for k in range(1, 13):
        varis[f'e{k}'] = variable.Variable(name=f'e{k}', B=B, values=['Surv', 'Fail'])

    sb_org = branch.branch_and_bound_org(bstars, path_time_idx, arc_cond)
    C1 = branch.get_cmat_from_branches(sb_org, varis)
    C1 = C1.astype(int)
    C1 = C1[C1[:, 0].argsort()]
    np.savetxt(HOME.joinpath('./C_rbd_org.txt'), C1, fmt='%d')
    assert C1.shape[0] == 77

    branch.branch_and_bound(bstars, path_time_idx, arc_cond, output_path, key)

    sb = branch.get_sb_saved_from_job(output_path, key)

    C = branch.get_cmat_from_branches(sb, varis)
    C = C.astype(int)
    C = C[C[:, 0].argsort()]
    np.savetxt(HOME.joinpath('./C_rbd.txt'), C, fmt='%d')
    #assert C.shape[0] == 77
    assert all((C[:, None] == C1).all(-1).any(-1))


def test_get_arcs_given_bstar():

    bstars = branch.get_bstars_from_sb_dump(HOME.joinpath('./sb_dump_1.json'))
    bstar = bstars[0]

    with open(HOME.joinpath('../demos/SF/path_time_idx.json'), 'r') as fid:
        _dic = json.load(fid)

    path_time_idx = _dic['od1']
    result = branch.get_arcs_given_bstar(bstar, path_time_idx, 1)
    expected = ['e6_8', 'e8_9', 'e5_9', 'e4_5', 'e3_4', 'e3_12', 'e11_12']
    assert result == expected

    result = branch.get_arcs_given_bstar_nobreak(bstar, path_time_idx, 1)
    assert result == expected


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


@pytest.mark.skip('NODASK')
def test_dask_fib(setup_client):

    cluster = setup_client

    with Client(cluster) as client:
        future = client.submit(fib2, 10)
        result = client.gather(future)

    assert result[0] == 55
    assert result[1] == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]


def test_create_arc_state_given_cond():

    arc = 'e3'
    arc_state = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}
    expected = {'e1': 0, 'e2': 0, 'e3': 1, 'e4': 0, 'e5': 0, 'e6': 0}

    result = branch.create_arc_state_given_cond(arc, value=1, arc_state=arc_state)
    assert result==expected

    arc = 'e5'
    arc_state = {'e1': 1, 'e2': 1, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}
    expected = {'e1': 1, 'e2': 1, 'e3': 1, 'e4': 1, 'e5': 0, 'e6': 1}

    result = branch.create_arc_state_given_cond(arc, value=0, arc_state=arc_state)

    assert result==expected

@pytest.mark.skip('NWY')
def test_get_set_branches_no_iteration():

    #cluster = setup_client

    # using road
    cfg = config.Config(HOME.joinpath('../demos/road/config_road.json'))

    path_times = trans.get_all_paths_and_times(cfg.infra['ODs'].values(), cfg.infra['G'], key='time')

    expected =[({'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 1, 'e2': 0, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 0, 1),
               ({'e1': 0, 'e2': 1, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, {'e1': 1, 'e2': 1, 'e3': 1, 'e4': 1, 'e5': 1, 'e6': 1}, 2, 2)]

    k, v = 'od1', ('n1', 'n3')
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

    sb = branch.get_set_of_branches(bstar, arcs, path_time_idx, arc_cond)

    assert len(sb) == 2
    assert sb == expected


def test_get_set_branches_no_iteration_sf():

    cfg = config.Config(HOME.joinpath('../demos/SF/config.json'))

    with open(HOME.joinpath('../demos/SF/path_time_idx.json'), 'r') as fid:
        _dic = json.load(fid)
    path_time_idx = _dic['od1']

    arc_cond = 1
    lower = {k: 0 for k, _ in cfg.infra['edges'].items()}
    upper = {k: 1 for k, _ in cfg.infra['edges'].items()}

    fl = trans.eval_sys_state(path_time_idx, lower, arc_cond)
    fu = trans.eval_sys_state(path_time_idx, upper, arc_cond)

    bstar = (lower, upper, fl, fu)

    arcs = branch.get_arcs_given_bstar(bstar, path_time_idx, arc_cond)
    sb = branch.get_set_of_branches(bstar, arcs, path_time_idx, arc_cond)

