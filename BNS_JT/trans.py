import numpy as np
import dask
import json
import networkx as nx
import socket
import matplotlib

from BNS_JT import variable
from scipy.stats import lognorm

if 'gadi' in socket.gethostname():
    matplotlib.use('Agg')
else:
    matplotlib.use("TKAgg")
    import matplotlib.pyplot as plt


system_meta = {'system_meta': {
        "MODEL_NAME": {
            "value": "Basic Linear Network",
            "notes": None
        },
        "INFRASTRUCTURE_LEVEL": {
            "value": "network",
            "notes": None
        },
        "SYSTEM_CLASS": {
            "value": "RailNetwork",
            "notes": None
        },
        "SYSTEM_SUBCLASS": {
            "value": "Regional Rail Network",
            "notes": None
        },
        "SYSTEM_COMPONENT_LOCATION_CONF": {
            "value": "defined",
            "notes": None
        },
        "RESTORATION_TIME_UNIT": {
            "value": "days",
            "notes": None
        },
        "HAZARD_INTENSITY_MEASURE_PARAM": {
            "value": "PGA",
            "notes": None
        },
        "HAZARD_INTENSITY_MEASURE_UNIT": {
            "value": "g",
            "notes": None
        }
        }
        }


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
    vari: instance of variable.Variable

    """
    assert isinstance(path_time, list)
    assert all([isinstance(x, tuple) for x in path_time])
    assert all([len(x)==2 for x in path_time])
    assert isinstance(vari, variable.Variable)

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


def eval_sys_state_given_arc(arcs_state, **kwargs):
    """
    arcs_state: dict or frozenset
    path_time_idx: a list of tuple (path, time, idx)
    arc_cond: value for survival (row index)
    """

    path_time_idx = kwargs['path_time_idx']
    arc_cond = kwargs['arc_cond']

    sys_state = path_time_idx[0][2]  # no path available

    for edges, _, state in path_time_idx:

        path_is_surv = [arcs_state[i]==arc_cond for i in edges]
        if path_is_surv and all(path_is_surv):
            sys_state = state
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


def get_node_conn_df(arcs, node_coords, avg_speed_by_arc, flag=False):

    distance_by_arc = get_arcs_length(arcs, node_coords)

    #arcTimes_h = arcLens_km ./ arcs_Vavg_kmh
    _dic = {'node_conn_df': {}}
    for k, v in arcs.items():

        _dic['node_conn_df'][k] = {'origin': v[0],
                   'destination': v[1],
                   'link_capacity': None,
                   'weight': distance_by_arc[k]/avg_speed_by_arc[k]}

    if flag:
        wfile = './node_conn_df.json'
        with open(wfile, 'w') as w:
            json.dump(_dic, w, indent=4)

        print(f'{wfile} is written')

    return _dic


def create_model_json_for_tranportation_network(arcs, node_coords, avg_speed_by_arc, ODs, key=None):

    _dic = {}
    _dic.update(system_meta)

    sysout_setup = {'sysout_setup': {}}
    for k, v in ODs.items():
        sysout_setup['sysout_setup'][k] = {'origin': v[0],
                           'destination': v[1],
                           'output_node_capacity': None,
                           'capacity_fraction': None,
                           'priorty': None
                           }
    _dic.update(sysout_setup)

    node_conn_df = get_node_conn_df(arcs, node_coords, avg_speed_by_arc)
    _dic.update(node_conn_df)

    component_list = {'component_list': {}}
    for k, v in node_coords.items():
        component_list['component_list'][k] = {'component_type': None,
                             'component_class': None,
                             'cost_fraction': None,
                             'cost_fraction': None,
                             'node_type': None,
                             'node_cluster': None,
                             'operation_capacity': None,
                             'pos_x': v[0],
                             'pos_y': v[1],
                             'damages_states_constructor': None}
    _dic.update(component_list)

    if key:
        wfile = f'./model_{key}.json'
        with open(wfile, 'w') as w:
            json.dump(_dic, w, indent=4)
        print(f'{wfile} is written')

    return _dic

def create_scenario_json_for_tranportation_network(damage_states, arcs, type_by_arc, frag_by_type, obs_by_arc, key=None):
    """
    damage_states: list of string
    arcs: dict
    type_by_arc: dict
    frag_by_type: dict (only support lognorm.cdf, 'std', 'med')
    obs_by_arc: dict
    """
    _dic = {'damage_states': damage_states}
    s1_list = {'scenarios': {'s1': {}}}

    for k in arcs.keys():

        _type = type_by_arc[k]

        prob = lognorm.cdf(obs_by_arc[k],
                           frag_by_type[_type]['std'],
                           scale=frag_by_type[_type]['med'])

        s1_list['scenarios']['s1'][k] = [prob, 1-prob]

    _dic.update(s1_list)

    if key:
        wfile = f'./scenarios_{key}.json'
        with open(wfile, 'w') as w:
            json.dump(_dic, w, indent=4)
        print(f'{wfile} is written')

    return _dic


def plot_graph(G, filename=None):

    # plot graph
    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = nx.get_edge_attributes(G, 'label')
    fig = plt.figure()
    ax = fig.add_subplot()
    nx.draw(G, pos, with_labels=True, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

    if filename:
       fig.savefig(filename, dpi=200)
       print(f'{filename} is created')

