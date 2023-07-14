import numpy as np

from BNS_JT.cpm import ismember


def bnb_sys(comp_states, info):
    """
    Parameters
    ----------
    "info": a dict that contains any information about a given problem
        path
        path_time
        arcs
        max_state
    comp_states: list-like

    Output:
    "state": a positve integer indicating a computed system state
    "val": any value that shows what the "state" means if unnecessary, can be left empty
    "result": a structure with any field that is required for the functions "nextComp" and "nextState"
    """
    result = {}

    assert isinstance(info, dict), 'info should be a dict'
    assert isinstance(info['path'], list), 'path should be a list'
    assert isinstance(info['time'], (list, np.ndarray)), 'path_time should be a list-like'
    assert isinstance(info['arcs'], (list, np.ndarray)), 'arcs should be a list-like'

    if isinstance(comp_states, list):
        comp_states = np.array(comp_states)

    if isinstance(info['time'], list):
        info['time'] = np.array(info['time'])

    if isinstance(info['arcs'], np.ndarray):
        info['arcs'] = info['arcs'].tolist()


    path_time = info['time']

    # Ensure shorter paths to be considered first
    path_time = np.sort(path_time)
    path_sort_idx = path_time.argsort().tolist()
    path = [info['path'][i] for i in path_sort_idx]

    # Find the shortest path possible
    idx = np.where(comp_states==info['max_state'])[0]
    try:
        surv_comps = [info['arcs'][i] for i in idx]
    except TypeError:
        surv_comps = []
    is_path_conn = [all(ismember(x, surv_comps)[0]) for x in path]
    is_path_conn = np.where(is_path_conn)[0].tolist()

    # Result
    if is_path_conn:
        is_path_conn = is_path_conn[0] # take the first
        #FIXME
        state = path_sort_idx[is_path_conn]
        val = path_time[is_path_conn]
        result['path'] = path[is_path_conn]
    else:
        #FIXME
        state = len(path_time) # there is no path available
        val = np.inf
        result['path'] = []

    return state, val, result


def bnb_next_comp(cand_comps, down_res, up_res, info):
    """
    Parameters
    ----------
    "info": a dict that contains any information about a given problem
        path
        path_time
        arcs

    comp_states: list-like

    Output:
    "state": a positve integer indicating a computed system state
    "val": any value that shows what the "state" means if unnecessary, can be left empty
    "result": a structure with any field that is required for the functions "nextComp" and "nextState"
    """
    assert isinstance(info['path'], list), 'path should be a list'
    assert isinstance(info['time'], (list, np.ndarray)), 'path_time should be a list-like'
    assert isinstance(info['arcs'], (list, np.ndarray)), 'arcs should be a list-like'

    if isinstance(info['time'], list):
        info['time'] = np.array(info['time'])

    path = info['path']
    path_time = info['time']

    path_sort_idx = path_time.argsort()
    path_time = np.sort(path_time)

    comps_order = []
    _diff = []
    for _path in path:
        # Do not change the order so that components on shorter paths come forward.
        [comps_order.append(x) for x in _path if x not in comps_order]

    [_diff.append(x) for x in info['arcs'] if x not in comps_order]
    [comps_order.append(x) for x in _diff if x not in comps_order]

    next_comp_idx = ismember(cand_comps, comps_order)[0]
    next_comp_idx = np.where(next_comp_idx)[0].tolist()[0]
    next_comp = cand_comps[next_comp_idx]

    return next_comp


def bnb_next_state(next_comp, bound, down_res, up_res, info):
    # In this example, there is only two states, so there's nothing much to do
    next_state = 1

    return next_state
