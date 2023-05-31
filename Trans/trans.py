import numpy as np
import networkx as nx


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


def eval_sys_route(OD1,G, arcs_state1, arc_surv, arc_fail, key='time'):

    
    path_time = get_all_paths_and_times([OD1], G, key)
    path_time[OD1].append(([], float('inf')))


    for state in range(0, len(path_time[OD1])):

        path_state = [arcs_state1[i-1] for i in path_time[OD1][state][0]]
        path_is_surv = [path_state1 == arc_surv for path_state1 in path_state]

        if all(path_is_surv):
            sys_state = state
            break

    return sys_state
