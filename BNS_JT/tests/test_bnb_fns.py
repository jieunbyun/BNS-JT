import numpy as np

from BNS_JT import bnb_fns


def test_bnb_sys1():

    comp_states = [1, 1, 1, 1, 1, 1]
    info = {'path': [[2], [3, 1]],
            'time': np.array([0.0901, 0.2401]),
            'arcs': [1, 2, 3, 4, 5, 6],
            'max_state': 2}
    #pdb.set_trace()
    state, val, result = bnb_fns.bnb_sys(comp_states, info)

    assert state== 3-1
    assert val== np.inf
    assert result== {'path': []}


def test_bnb_sys2():

    comp_states = [1, 2, 1, 1, 1, 1]
    info = {'path': [[2], [3, 1]],
            'time': np.array([0.0901, 0.2401]),
            'arcs': [1, 2, 3, 4, 5, 6],
            'max_state': 2}
    state, val, result = bnb_fns.bnb_sys(comp_states, info)

    assert state== 1-1
    assert val== 0.0901
    assert result=={'path': [2]}


def test_bnb_sys3():

    comp_states = [2, 2, 2, 2, 2, 2]
    info = {'path': [[2], [3, 1]],
            'time': np.array([0.0901, 0.2401]),
            'arcs': [1, 2, 3, 4, 5, 6],
            'max_state': 2}
    #pdb.set_trace()
    state, val, result = bnb_fns.bnb_sys(comp_states, info)

    assert state==1-1
    assert val==0.0901
    assert result=={'path': [2]}


def test_bnb_sys4():

    comp_states = [1, 2, 2, 2, 2, 2]
    info = {'path': [[2], [3, 1]],
            'time': np.array([0.0901, 0.2401]),
            'arcs': [1, 2, 3, 4, 5, 6],
            'max_state': 2
            }
    #pdb.set_trace()
    state, val, result = bnb_fns.bnb_sys(comp_states, info)

    assert state==1-1
    assert val==0.0901
    assert result=={'path': [2]}


def test_bnb_sys5():

    comp_states = [1, 1, 2, 2, 2, 2]
    info = {'path': [[2], [3, 1]],
            'time': np.array([0.0901, 0.2401]),
            'arcs': [1, 2, 3, 4, 5, 6],
            'max_state': 2}
    #pdb.set_trace()
    state, val, result = bnb_fns.bnb_sys(comp_states, info)

    assert state== 3-1
    assert val== np.inf
    assert result== {'path': []}


def test_bnb_sys6():
    # 2: survival, 1: failure
    comp_states = [2, 1, 2, 1, 1, 1]
    info = {'path': [[2], [3, 1]],
            'time': np.array([0.0901, 0.2401]),
            'arcs': [1, 2, 3, 4, 5, 6],
            'max_state': 2,
            }
    #pdb.set_trace()
    state, val, result = bnb_fns.bnb_sys(comp_states, info)

    assert state==2-1
    assert val==0.2401
    assert result=={'path': [3, 1]}


def test_bnb_next_comp1():

    cand_comps = [1, 2, 3, 4, 5, 6]
    down_res = []
    up_res = [3, 1]
    info = {'path': [[2], [3, 1]],
            'time': np.array([0.0901, 0.2401]),
            'arcs': [1, 2, 3, 4, 5, 6],
            }

    next_comp = bnb_fns.bnb_next_comp(cand_comps, down_res, up_res, info)

    assert next_comp== 1


def test_bnb_next_comp2():

    cand_comps = [2, 3, 4, 5, 6]
    down_res = []
    up_res = [2]
    info = {'path': [[2], [3, 1]],
            'time': np.array([0.0901, 0.2401]),
            'arcs': np.array([1, 2, 3, 4, 5, 6]),
            }

    next_comp = bnb_fns.bnb_next_comp(cand_comps, down_res, up_res, info)

    assert next_comp== 2


def test_bnb_next_comp3():

    cand_comps = [3, 4, 5, 6]
    down_res = []
    up_res = [3, 1]
    info = {'path': [[2], [3, 1]],
            'time': np.array([0.0901, 0.2401]),
            'arcs': np.array([1, 2, 3, 4, 5, 6]),
            }

    next_comp = bnb_fns.bnb_next_comp(cand_comps, down_res, up_res, info)

    assert next_comp== 3


