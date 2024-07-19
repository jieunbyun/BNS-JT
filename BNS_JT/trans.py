import numpy as np
#import dask
import json
import networkx as nx
import socket
import matplotlib
import copy

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


def sys_fun_wrap(G, od_pair, varis=None, thres=None):

    def sys_fun2(comps_st):
        return sf_min_path(comps_st, G, od_pair, varis=varis, thres=thres)
    return sys_fun2


def sf_min_path(comps_st, G, od_pair, varis=None, thres=None):
    """
    comps_st:
    G:
    od_pair:
    varis:
    thres:
    """

    if thres:

        if isinstance(od_pair, (list, tuple)):
            d_time, path_n, path_e = get_time_and_path_given_comps(comps_st, G, od_pair, varis)

        elif isinstance(od_pair, dict): # multiple destinations, e.g. {origin: n1, dests: [n2, n3, n4]}
            d_time, path_n, path_e = get_time_and_path_multi_dest(comps_st, G, od_pair['origin'], od_pair['dests'], varis)

        if d_time > thres:
            sys_st = 'f'
        else:
            sys_st= 's'

    else:
        # no threshold, only check connectivity
        # varis is not used; comps_st has the information (0: fail, 1: success)
        d_time = None

        # comps_st can be either node or edge, or combined
        path_e, path_n = get_connectivity_given_comps(comps_st, G, od_pair)

        if path_n:
            sys_st = 's'
        else:
            sys_st= 'f'

    min_comps_st = {}
    if sys_st == 's':
        # edges
        min_comps_st.update({k: comps_st[k] for k in path_e if k in comps_st})

        # nodes
        min_comps_st.update({k: comps_st[k] for k in path_n if k in comps_st})

    return d_time, sys_st, min_comps_st


def get_connectivity_given_comps(comps_st, G, od_pair):
    """
    return path on connectivity given od_pair including od_pair
    it works on either node or edge
    comps_st: starting from 0: (0: failure, 1: intact) only binary status
              comps_st can be for node, edge or combined
    G: graph
    od_pair:
    arcs:
    vari:
    """
    assert isinstance(comps_st, dict)
    #assert all([v < len(vari[k].values) for k, v in comps_st.items()])

    # node of od_pair should be intact regardless of comps_st
    #_comps_st = comps_st.copy()
    #_comps_st.update({k: 1 for k in od_pair})

    H = update_G_given_comps_st(G, comps_st)

    # shortest path in edges
    try:
        # shortest path in nodes
        s_path = nx.shortest_path(H, source = od_pair[0], target = od_pair[1])

    except (nx.NodeNotFound, nx.exception.NetworkXNoPath):
        s_path, s_path_edge = [], []

    else:
        s_path_edge = get_edge_from_nodes(H, s_path)

    return s_path_edge, s_path


def get_edge_from_nodes(G, path_nodes):

    # path in edges
    path_edges = []
    if isinstance(G, nx.MultiGraph):
        for p in nx.utils.pairwise(path_nodes):
            edges = (G.get_edge_data(*p).items())
            sorted_edges = sorted(edges, key=lambda x: x[1].get('weight'))
            path_edges.append(sorted_edges[0][1]['label'])
    else:
        for p in nx.utils.pairwise(path_nodes):
            path_edges.append(G.get_edge_data(*p)['label'])

    return path_edges


def update_G_given_comps_st(G, comps_st, varis=None):

    assert isinstance(G, nx.Graph), f'G must be an instance of networkx.Graph'

    H = copy.deepcopy(G)

    if varis:

        attributes = {}
        if isinstance(H, nx.MultiGraph):
            for e in H.edges(data=True):
                k = e[2]['label']
                attributes[(e[0], e[1], 0)] = {'weight': varis[k].values[comps_st[k]]}
        else:
            for e in H.edges(data=True):
                k = e[2]['label']
                attributes[(e[0], e[1])] = {'weight': varis[k].values[comps_st[k]]}

         # update H
        nx.set_edge_attributes(H, attributes)

    else:

        # edge
        for n1, n2, d in list(H.edges(data=True)):
            try:
                s = comps_st[d['label']]
            except KeyError:
                pass
            else:
                if not s:
                    try:
                        H.remove_edge(n1, n2, key=d['label'])
                    except TypeError:
                        H.remove_edge(n1, n2)

        # node
        for n0 in list(H.nodes):
            try:
                s = comps_st[n0]
            except KeyError:
                pass
            else:
                if not s:
                    H.remove_node(n0)

    return H


def get_time_and_path_given_comps(comps_st, G, od_pair, varis):
    """
    comps_st: starting from 0
    od_pair:
    arcs:
    vari:
    """
    assert isinstance(comps_st, dict)
    assert all([comps_st[k] < len(v.values) for k, v in varis.items()])

    H = update_G_given_comps_st(G, comps_st, varis)

    path = nx.shortest_path(H, source=od_pair[0], target=od_pair[1], weight='weight')
    d_time = nx.shortest_path_length(H, source=od_pair[0], target=od_pair[1], weight='weight')
    path_e = get_edge_from_nodes(H, path)

    return d_time, path, path_e


def get_time_and_path_multi_dest(comps_st, G, origin, dests, varis):
    """
    Compute time and path given multiple destinations--NB: works only for bidirectional graph
    """
    assert isinstance(dests, (list, tuple)), f'dests must be a list-like: {type(dests)}'
    assert isinstance(comps_st, dict)
    assert all([comps_st[k] < len(varis[k].values) for k in comps_st.keys()])

    # G must be a undirected (either Graph or MultiGraph)
    assert not isinstance(G, (nx.DiGraph, nx.MultiDiGraph)), f'G must not be a directed graph'

    H = update_G_given_comps_st(G, comps_st, varis)

    try:
        d_time, path_nodes = nx.multi_source_dijkstra(H, sources=dests, target=origin, weight='weight')
    except nx.exception.NetworkXNoPath:
        d_time, path_nodes, path_edges = np.inf, [], []
    else:
        path_nodes = path_nodes[::-1] # reverse as origin was used as the target
        path_edges = get_edge_from_nodes(H, path_nodes)

    return d_time, path_nodes, path_edges


def get_all_paths_and_times(ODs, G, key='weight'):
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

    # sort by elapsed time
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


def get_node_conn_df(arcs, node_coords, avg_speed_by_arc, outfile=None):

    distance_by_arc = get_arcs_length(arcs, node_coords)

    #arcTimes_h = arcLens_km ./ arcs_Vavg_kmh
    _dic = {'node_conn_df': {}}
    for k, v in arcs.items():

        _dic['node_conn_df'][k] = {'origin': v[0],
                   'destination': v[1],
                   'link_capacity': None,
                   'weight': distance_by_arc[k]/avg_speed_by_arc[k]}

    if outfile:
        with open(outfile, 'w') as w:
            json.dump(_dic, w, indent=4)
        print(f'{outfile} is written')

    return _dic


def create_model_json_for_graph_network(arcs, node_coords, ODs, outfile=None):

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

    #arcTimes_h = arcLens_km ./ arcs_Vavg_kmh
    _dic['node_conn_df'] = {}
    for k, v in arcs.items():
        _dic['node_conn_df'][k] = {'origin': v[0],
                   'destination': v[1],
                   'link_capacity': None,
                   'weight': 1.0}

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

    if outfile:
        with open(outfile, 'w') as w:
            json.dump(_dic, w, indent=4)
        print(f'{outfile} is written')

    return _dic


def create_model_json_for_tranportation_network(arcs, node_coords, avg_speed_by_arc, ODs, outfile=None):

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

    if outfile:
        with open(outfile, 'w') as w:
            json.dump(_dic, w, indent=4)
        print(f'{outfile} is written')

    return _dic


def create_scenario_json_for_trans_network(damage_states, scenarios_dic, outfile=None):

    assert isinstance(damage_states, list), 'damage_states should be a list'
    assert isinstance(scenarios_dic, dict), 'scenarios_dic should be a dict'

    _dic = {'damage_states': damage_states}

    scenarios = {'scenarios': {}}
    for k, v in scenarios_dic.items():

        scenarios['scenarios'][k] = v

    _dic.update(scenarios)

    if outfile:
        with open(outfile, 'w') as w:
            json.dump(_dic, w, indent=4)
        print(f'{outfile} is written')

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

