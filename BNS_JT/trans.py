import numpy as np
import networkx as nx
from BNS_JT.variable import Variable

def get_all_paths_and_times(ODs, G, key='time'):
    """
    ODs: list of OD pairs
    G: instance of networkx.Graph
    key: keyword for weight
    """

    path_time = {}
    for org, dest in ODs:
        for _path in nx.all_simple_paths(G, org, dest):
            val = nx.path_weight(G, _path, weight=key)

            edges_path = []
            for x in list(zip(_path, _path[1:])):
                edges_path.append(G[x[0]][x[1]]['label'])

            path_time.setdefault((org, dest), []).append((edges_path, val))

    return path_time


def get_path_time_idx(path_time, vari):
    """
    path_time: a list of tuple
    vari: instance of Variable

    """
    assert isinstance(path_time, list)
    assert all([isinstance(x, tuple) for x in path_time])
    assert all([len(x)==2 for x in path_time])
    assert isinstance(vari, Variable)

    path_timex = path_time[:]

    # refering variable
    path_time_idx = []
    for x in path_timex:
        idx = [i for i, y in enumerate(vari.values) if np.isclose(x[1], y)]
        try:
            path_time_idx.append((*x, idx[0]))
        except IndexError:
            print('path_time incompatible with variable')

    # sort by increasing number of edges
    path_time_idx = sorted(path_time_idx, key=lambda x: x[2], reverse=True)

    if not any([np.inf in x for x in path_timex]):
        path_time_idx.insert(0, ([], np.inf, 0))

    return path_time_idx


def get_arcs_length(arcs, node_coords):
    """
    if arcs.shape[1] == 2:
        nArc = arcs.shape[0]
    elif arcs.shape[0] == 2:
        arcs = arcs.T
        nArc = arcs.shape[0]
    else:
        print('"arcs" must have either two columns or two rows (each noting start and end points)')

    if node_coords.shape[1] != 2:
        if node_coords.shape[0] == 2:
            node_coords = nodeCoords.T
        else:
            print('"node_coords" must have either two columns or two rows (each noting coordinates of x and y).')
    """
    arc_len = {}
    for k, v in arcs.items():
        diff = np.array(node_coords[v[0]]) - np.array(node_coords[v[1]])
        arc_len[k] = np.sqrt(np.sum(diff**2))

    return arc_len


def get_match(a, b, complete, idx_any):

    mask = np.equal(a, b)

    if sum(mask):

        res = [x if z else 'x' for x, z in zip(a, mask)]

        idx = res.index('x')

        if (res.count('x') == 1) and (set([a[idx], b[idx]]) == set(complete[idx])):

            res[idx] = idx_any

            return res


def do_branch(group, complete, id_any):
    """

    """

    while len(group) > 1:

        copied = group.copy()

        a = group.pop(0)
        b = group.pop(0)

        res = get_match(a, b, complete, id_any)

        if res:
            group.append(res)

        else:

            group.append(a)
            group.append(b)

            if group == copied:

                return group

    return group


def eval_sys_route_old(OD, G, arcs_state, arc_cond, key='time'):

    path_time = get_all_paths_and_times([OD], G, key)[OD]
    path_time = sorted(path_time, key=lambda x: x[1])

    sys_state = 0  # no path available
    for state, (edges, _time) in enumerate(path_time, 1):

        path_is_surv = [arcs_state[i]==arc_cond for i in edges]
        if all(path_is_surv):
            sys_state = len(path_time) - state + 1
            break

    return sys_state


def eval_sys_state(path_time_idx, arcs_state, arc_cond):
    """
    path_time_idx: a list of tuple (path, time, idx)
    arcs_state: dict or frozenset
    arc_cond: value for survival (row index)
    """

    sys_state = path_time_idx[0][2]  # no path available

    for edges, _, state in path_time_idx:

        path_is_surv = [arcs_state[i]==arc_cond for i in edges]
        if path_is_surv and all(path_is_surv):
            sys_state = state
            break

    return sys_state


