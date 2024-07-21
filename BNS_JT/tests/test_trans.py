'''
Ji-Eun Byun (ji-eun.byun@glasgow.ac.uk)
Created: 27 Mar 2023

A small, hypothetical bridge system
'''
import pytest
import numpy as np
import pandas as pd
import networkx as nx
import socket
import matplotlib
import pdb
import copy

from scipy.stats import lognorm
from pathlib import Path
from math import isclose

np.set_printoptions(precision=3)
#pd.set_option.display_precision = 3

if 'gadi' in socket.gethostname():
    matplotlib.use('Agg')
else:
    matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

from BNS_JT import cpm, variable, config, branch, model, trans, operation


HOME = Path(__file__).absolute().parent


@pytest.fixture(scope='package')
def data_bridge():
    """
    system with 6 edges, 5 nodes
    (see graph.png)
    """

    data = {}

    # Network
    data['node_coords'] = {'n1': (-2, 3),
                   'n2': (-2, -3),
                   'n3': (2, -2),
                   'n4': (1, 1),
                   'n5': (0, 0)}

    data['arcs'] = {'e1': ['n1', 'n2'],
            'e2': ['n1', 'n5'],
            'e3': ['n2', 'n5'],
            'e4': ['n3', 'n4'],
            'e5': ['n3', 'n5'],
            'e6': ['n4', 'n5']}

    data['arcs_avg_kmh'] = {'e1': 40,
                    'e2': 40,
                    'e3': 40,
                    'e4': 30,
                    'e5': 30,
                    'e6': 20}

    data['frag'] = {'major': {'med': 60.0, 'std': 0.7},
            'urban' : {'med': 24.0, 'std': 0.7},
            'bridge': {'med': 1.1, 'std': 3.9},
            }

    data['arcs_type'] = {'e1': 'major',
             'e2': 'major',
             'e3': 'major',
             'e4': 'urban',
             'e5': 'bridge',
             'e6': 'bridge'}

    data['var_ODs'] = {'od1': ('n5', 'n1'),
               'od2': ('n5', 'n2'),
               'od3': ('n5', 'n3'),
               'od4': ('n5', 'n4')}

    # For the moment, we assume that ground motions are observed. Later, hazard nodes will be added.
    data['GM_obs'] = {'e1': 30.0,
              'e2': 20.0,
              'e3': 10.0,
              'e4': 2.0,
              'e5': 0.9,
              'e6': 0.6}

    return data


@pytest.fixture(scope='package')
def expected_probs():

    probs = {}
    probs['disconn'] = np.array([0.0096, 0.0011, 0.2102, 0.2102])
    probs['delay'] = np.array([0.0583, 0.0052, 0.4795, 0.4382])
    probs['damage'] = np.array([0.1610,  1,  1,  0.0002,   0,  0.4382])

    return probs


@pytest.fixture(scope='package')
def setup_bridge(data_bridge):
    """
    system with 6 edges, 5 nodes
    (see graph.png)
    """
    arcs = data_bridge['arcs']
    node_coords = data_bridge['node_coords']
    arcs_avg_kmh = data_bridge['arcs_avg_kmh']
    arcs_type = data_bridge['arcs_type']
    GM_obs = data_bridge['GM_obs']
    frag = data_bridge['frag']
    var_ODs = data_bridge['var_ODs']

    # Arcs' states index compatible with variable B index, and C
    #arc_surv = 0
    #arc_fail = 1
    #arc_either = 2

    arc_lens_km = trans.get_arcs_length(arcs, node_coords)

    #arcTimes_h = arcLens_km ./ arcs_Vavg_kmh
    arc_times_h = {k: v/arcs_avg_kmh[k] for k, v in arc_lens_km.items()}

    # create a graph
    G = nx.Graph()
    for k, x in arcs.items():
        G.add_edge(x[0], x[1], weight=arc_times_h[k], label=k)

    for k, v in node_coords.items():
        G.add_node(k, pos=v, label=k)

    # plot graph
    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = nx.get_edge_attributes(G, 'label')

    fig = plt.figure()
    ax = fig.add_subplot()
    nx.draw(G, pos, with_labels=True, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    fig.savefig(HOME.joinpath('graph_bridge.png'), dpi=100)

    # Arcs (components): P(X_i | GM = GM_ob ), i = 1 .. N (= nArc)
    cpms_arc = {}
    vars_arc = {}

    #path_time = trans.get_all_paths_and_times(var_ODs.values(), G, key='time')

    # number of component states: 2 ('surv' or 'fail')
    for k in arcs.keys():
        #vars_arc[k] = variable.Variable(name=str(k), B=[{0}, {1}, {0, 1}], values=['Fail', 'Surv'])
        vars_arc[k] = variable.Variable(name=str(k), values=['Fail', 'Surv'])

        _type = arcs_type[k]
        prob = lognorm.cdf(GM_obs[k], frag[_type]['std'], scale=frag[_type]['med'])
        C = np.array([[0, 1]]).T
        p = np.array([prob, 1-prob])
        cpms_arc[k] = cpm.Cpm(variables = [vars_arc[k]],
                              no_child = 1,
                              C = C,
                              p = p)

    # Travel times (systems): P(OD_j | X1, ... Xn) j = 1 ... nOD
    # e.g., for 'od1': 'e2': 0.0901, 'e3'-'e1': 0.2401
    vars_arc['od1'] = variable.Variable(name='od1',
            values=[np.inf, 0.2401, 0.0901])

    vars_arc['od2'] = variable.Variable(name='od2',
            values=[np.inf, 0.2401, 0.0901])

    vars_arc['od3'] = variable.Variable(name='od3',
            values=[np.inf, 0.1761, 0.0943])

    vars_arc['od4'] = variable.Variable(name='od4',
            values=[np.inf, 0.1997, 0.0707])

    _variables = [vars_arc[k] for k in ['od1', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6']]
    c7 = np.array([[0,2,0,0,2,2,2],
                   [0,0,0,1,2,2,2],
                   [1,1,0,1,2,2,2],
                   [2,2,1,2,2,2,2]])
    cpms_arc['od1'] = cpm.Cpm(variables= _variables,
                           no_child = 1,
                           C = c7,
                           p = [1, 1, 1, 1],
                           )

    _variables = [vars_arc[k] for k in ['od2', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6']]
    c8 = np.array([[0,2,0,0,2,2,2],
                  [0,0,1,0,2,2,2],
                  [1,1,1,0,2,2,2],
                  [2,2,2,1,2,2,2]])
    cpms_arc['od2'] = cpm.Cpm(variables= _variables,
                           no_child = 1,
                           C = c8,
                           p = [1, 1, 1, 1],
                           )

    _variables = [vars_arc[k] for k in ['od3', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6']]
    c9 = np.array([[0,2,2,2,2,0,0],
                   [0,2,2,2,0,0,1],
                   [1,2,2,2,1,0,1],
                   [2,2,2,2,2,1,2]])
    cpms_arc['od3'] = cpm.Cpm(variables= _variables,
                           no_child = 1,
                           C = c9,
                           p = [1, 1, 1, 1],
                           )

    _variables = [vars_arc[k] for k in ['od4', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6']]
    c10 = np.array([[0,2,2,2,2,0,0],
                   [0,2,2,2,0,1,0],
                   [1,2,2,2,1,1,0],
                   [2,2,2,2,2,2,1]])
    cpms_arc['od4'] = cpm.Cpm(variables= _variables,
                           no_child = 1,
                           C = c10,
                           p = [1, 1, 1, 1],
                           )
    #probs = {k: cpms_arc[k].p[:, 0].tolist() for k in arcs.keys()}

    return cpms_arc, vars_arc, arcs, var_ODs


@pytest.fixture(scope='package')
def setup_bridge_alt(data_bridge):
    """
    Note the difference of arc's state from the setup_bridge
    """

    arcs = data_bridge['arcs']
    node_coords = data_bridge['node_coords']
    arcs_avg_kmh = data_bridge['arcs_avg_kmh']
    arcs_type = data_bridge['arcs_type']
    GM_obs = data_bridge['GM_obs']
    frag = data_bridge['frag']
    var_ODs = data_bridge['var_ODs']

    # Arcs' states index compatible with variable B index, and C
    #arc_fail = 0
    #arc_surv = 1
    #arc_either = 2

    arc_lens_km = trans.get_arcs_length(arcs, node_coords)

    #arcTimes_h = arcLens_km ./ arcs_Vavg_kmh
    arc_times_h = {k: v/arcs_avg_kmh[k] for k, v in arc_lens_km.items()}

    # create a graph
    G = nx.Graph()
    for k, x in arcs.items():
        G.add_edge(x[0], x[1], time=arc_times_h[k], label=k)

    for k, v in node_coords.items():
        G.add_node(k, pos=v)

    #path_time = trans.get_all_paths_and_times(var_ODs.values(), G, key='time')
    """
    # plot graph
    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = nx.get_edge_attributes(G, 'label')

    fig = plt.figure()
    ax = fig.add_subplot()
    nx.draw(G, pos, with_labels=True, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    fig.savefig(HOME.joinpath('graph.png'), dpi=200)
    """
    # Arcs (components): P(X_i | GM = GM_ob ), i = 1 .. N (= nArc)
    cpms_arc = {}
    vars_arc = {}

    for k in arcs.keys():
        #vars_arc[k] = variable.Variable(name=str(k), B=[{0}, {1}, {0, 1}], values=['Fail', 'Surv'])
        vars_arc[k] = variable.Variable(name=str(k), values=['Fail', 'Surv'])

        _type = arcs_type[k]
        prob = lognorm.cdf(GM_obs[k], frag[_type]['std'], scale=frag[_type]['med'])
        C = np.array([[0, 1]]).T
        p = np.array([prob, 1-prob])
        cpms_arc[k] = cpm.Cpm(variables = [vars_arc[k]],
                              no_child = 1,
                              C = C,
                              p = p)

    # Travel times (systems): P(OD_j | X1, ... Xn) j = 1 ... nOD
    # e.g., for 'od1': 'e2': 0.0901, 'e3'-'e1': 0.2401
    vars_arc['od1'] = variable.Variable(name='od1',
            values=[0.0901, 0.2401, np.inf])

    vars_arc['od2'] = variable.Variable(name='od2',
            values=[0.0901, 0.2401, np.inf])

    vars_arc['od3'] = variable.Variable(name='od3',
            values=[0.0943, 0.1761, np.inf])

    vars_arc['od4'] = variable.Variable(name='od4',
            values=[0.0707, 0.1997, np.inf])

    _variables = [vars_arc[k] for k in ['od1', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6']]
    c7 = np.array([
    [1,3,2,3,3,3,3],
    [2,2,1,2,3,3,3],
    [3,2,1,1,3,3,3],
    [3,1,1,3,3,3,3]]) - 1

    cpms_arc['od1'] = cpm.Cpm(variables= _variables,
                           no_child = 1,
                           C = c7,
                           p = [1, 1, 1, 1],
                           )

    _variables = [vars_arc[k] for k in ['od2', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6']]
    c8 = np.array([
    [1,3,3,2,3,3,3],
    [2,2,2,1,3,3,3],
    [3,2,1,1,3,3,3],
    [3,1,3,1,3,3,3]]) - 1
    cpms_arc['od2'] = cpm.Cpm(variables= _variables,
                           no_child = 1,
                           C = c8,
                           p = [1, 1, 1, 1],
                           )

    _variables = [vars_arc[k] for k in ['od3', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6']]
    c9 = np.array([
    [1,3,3,3,3,2,3],
    [2,3,3,3,2,1,2],
    [3,3,3,3,2,1,1],
    [3,3,3,3,1,1,3]]) - 1
    cpms_arc['od3'] = cpm.Cpm(variables= _variables,
                           no_child = 1,
                           C = c9,
                           p = [1, 1, 1, 1],
                           )

    _variables = [vars_arc[k] for k in ['od4', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6']]
    c10 = np.array([
    [1,3,3,3,3,3,2],
    [2,3,3,3,2,2,1],
    [3,3,3,3,2,1,1],
    [3,3,3,3,1,3,1]]) - 1
    cpms_arc['od4'] = cpm.Cpm(variables= _variables,
                           no_child = 1,
                           C = c10,
                           p = [1, 1, 1, 1],
                           )

    return cpms_arc, vars_arc, arcs, var_ODs


@pytest.fixture
def setup_multi_dest():
    # see graph.png 

    arcs = {'e1': ['n1', 'n2'],
            'e2': ['n1', 'n5'],
            'e3': ['n2', 'n5'],
            'e4': ['n3', 'n4'],
            'e5': ['n3', 'n5'],
            'e6': ['n4', 'n5']}

    arc_times_h = {'e1': [np.inf, 0.15], 'e2': [np.inf, 0.0901], 'e3': [np.inf, 0.0901], 'e4': [np.inf, 0.1054],
                   'e5': [np.inf, 0.0943], 'e6': [np.inf, 0.0707]}

    # create a graph
    G = nx.Graph()
    for k, x in arcs.items():
        G.add_edge(x[0], x[1], weight=arc_times_h[k][1], label=k)

    [G.add_node(f'n{k}') for k in range(1, 6)]

    varis = {k: variable.Variable(name=k, values=v) for k, v in arc_times_h.items()}

    origin = 'n1'
    dests = ['n3', 'n4']

    return G, origin, dests, arcs, varis


def test_prob_delay1(setup_bridge, expected_probs):

    ## Inference - by variable elimination (would not work for large-scale systems)
    # Probability of delay and disconnection
    # Becomes P(OD_1, ..., OD_M) since X_1, ..., X_N are eliminated
    #cpms_arc_cp = cpms_arc.values()

    d_cpms_arc, d_vars_arc, arcs, var_ODs = setup_bridge
    cpms_arc = copy.deepcopy(d_cpms_arc)
    vars_arc = copy.deepcopy(d_vars_arc)

    cpms_arc_cp = list(cpms_arc.values())
    nODs = len(var_ODs)

    # prod_cpms
    # get different variables order
    for i in arcs.keys():

        is_inscope = operation.isinscope([vars_arc[i]], cpms_arc_cp)
        cpm_sel = [y for x, y in zip(is_inscope, cpms_arc_cp) if x]
        cpm_mult = cpm.product(cpm_sel)
        cpm_mult = cpm_mult.sum([vars_arc[i]])

        cpms_arc_cp = [y for x, y in zip(is_inscope, cpms_arc_cp) if x == False]
        cpms_arc_cp.insert(0, cpm_mult)

    # cpms_arc_cp[0].variables = [10, 9, 7, 8]
    #print([x.name for x in cpms_arc_cp[0].variables])

    # Retrieve example results
    # P( OD_j = 3 ), j = 1, ..., M, where State 3 indicates disconnection
    ODs_prob_disconn = np.zeros(nODs)
    # P( (OD_j = 2) U (OD_j = 2) ), j = 1, ..., M, where State 2 indicates the use of the second shortest path (or equivalently, P( (OD_j = 1)^c ), where State 1 indicates the use of the shortest path)
    ODs_prob_delay = np.zeros(nODs)

    #disconn_state = 3 - 1
    disconn_state = 0

    for j, idx in enumerate(var_ODs):

        # Prob. of disconnection
        [cpm_ve] = operation.condition(cpms_arc_cp,
                                       cnd_vars=[vars_arc[idx]],
                                       cnd_states=[disconn_state])
        ODs_prob_disconn[j] = cpm_ve.p.sum(axis=0).item()

        # Prob. of delay
        var_loc = cpms_arc_cp[0].variables.index(vars_arc[idx])
        # except the shortest path (which is state 0)
        rows_to_keep = np.where(cpms_arc_cp[0].C[:, var_loc] < 2)[0]
        cpm_ve = cpms_arc_cp[0].get_subset(rows_to_keep)
        ODs_prob_delay[j] = cpm_ve.p.sum(axis=0).item()

        # Prob. of disconnection alternative
        rows_to_keep = np.where(cpms_arc_cp[0].C[:, var_loc] == disconn_state)[0]
        assert cpms_arc_cp[0].get_subset(rows_to_keep).p.sum() == ODs_prob_disconn[j]

    #plot_delay(ODs_prob_delay, ODs_prob_disconn, var_ODs)

    # Check if the results are the same
    np.testing.assert_array_almost_equal(ODs_prob_disconn, expected_probs['disconn'], decimal=4)
    np.testing.assert_array_almost_equal(ODs_prob_delay, expected_probs['delay'], decimal=4)


def test_prob_delay2(setup_bridge, expected_probs):

    d_cpms_arc, d_vars_arc, arcs, var_ODs = setup_bridge

    cpms_arc = copy.deepcopy(d_cpms_arc)
    vars_arc = copy.deepcopy(d_vars_arc)

    nODs = len(var_ODs)

    ## Repeat inferences again using new functions -- the results must be the same.
    # Probability of delay and disconnection
    M = [cpms_arc[k] for k in list(arcs.keys()) + list(var_ODs.keys())]
    var_elim_order = [vars_arc[i] for i in arcs.keys()]
    M_VE2 = operation.variable_elim(M, var_elim_order)

    # "M_VE2" same as "M_VE"
    # Retrieve example results
    ODs_prob_disconn2 = np.zeros(nODs)
    ODs_prob_delay2 = np.zeros(nODs)

    disconn_state = 0
    for j, idx in enumerate(var_ODs):

        # Prob. of disconnection
        # FIXME 2 -> 3?? 
        #disconn_state = vars_arc[idx].values.index(np.inf) + 1
        # the state of disconnection is assigned an arbitrarily large number 100
        ODs_prob_disconn2[j] = M_VE2.get_prob([vars_arc[idx]], [0])

        # Prob. of delay
        #ODs_prob_delay2[j] = cpm.get_prob(M_VE2, [vars_arc[idx]], [1-1], flag=False) # Any state greater than 1 means delay.
        ODs_prob_delay2[j] = M_VE2.get_prob([vars_arc[idx]], [0]) + M_VE2.get_prob([vars_arc[idx]], [1]) # 0, 1

    # Check if the results are the same
    np.testing.assert_array_almost_equal(ODs_prob_disconn2, expected_probs['disconn'], decimal=4)
    np.testing.assert_array_almost_equal(ODs_prob_delay2, expected_probs['delay'], decimal=4)


def test_prob_delay3(setup_bridge_alt, expected_probs):
    """ same as delay2 but only using one OD"""
    cpms_arc, vars_arc, arcs, var_ODs = setup_bridge_alt
    nODs = len(var_ODs)

    ## Repeat inferences again using new functions -- the results must be the same.
    # Probability of delay and disconnection
    #M = [cpms_arc[k] for k in list(arcs.keys()) + list(var_ODs.keys())]
    M = [cpms_arc[k] for k in list(arcs.keys()) + ['od1']]
    var_elim_order = [vars_arc[i] for i in arcs.keys()]
    M_VE2 = operation.variable_elim(M, var_elim_order)

    # "M_VE2" same as "M_VE"
    # Retrieve example results
    ODs_prob_disconn2 = np.zeros(nODs)
    ODs_prob_delay2 = np.zeros(nODs)

    #disconn_state = 2
    for j, idx in enumerate(['od1']):

        # Prob. of disconnection
        # FIXME 2 -> 3?? 
        #disconn_state = vars_arc[idx].values.index(np.inf) + 1
        # the state of disconnection is assigned an arbitrarily large number 100
        ODs_prob_disconn2[j] = M_VE2.get_prob([vars_arc[idx]], [2])

        # Prob. of delay
        ODs_prob_delay2[j] = M_VE2.get_prob([vars_arc[idx]], [1-1], flag=False) # Any state greater than 1 means delay.

    # Check if the results are the same
    np.testing.assert_array_almost_equal(ODs_prob_disconn2[0], expected_probs['disconn'][0], decimal=4)
    np.testing.assert_array_almost_equal(ODs_prob_delay2[0], expected_probs['delay'][0], decimal=4)


def test_prob_damage(setup_bridge, expected_probs):

    #arc_fail = 1
    #arc_surv = 0

    d_cpms_arc, d_vars_arc, arcs, var_ODs = setup_bridge
    cpms_arc = copy.deepcopy(d_cpms_arc)
    vars_arc = copy.deepcopy(d_vars_arc)

    # City 1 (od1) and 2 (od2) experienced a disruption in getting resources, City 3 (od3) was okay and 4 (od4) is unknown. Probability of damage of roads?
    # A composite state needs be created for City 1 and City 2
    # cf. 
    #B_ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    #vars_arc['OD1'] = variable.Variable(name='OD1', B=B_,
    #        values=[0.0901, 0.2401, np.inf])
    for idx in var_ODs.keys():
        vars_arc[idx].B.append({1, 2})

    # FIXME: sorting of variables in product
    # # Add observation nodes P( O_j | OD_j ), j = 1, ..., M
    var_ODs_obs = [f'od_obs{i}' for i, _ in enumerate(var_ODs.keys(), 1)]
    # column 0 for od_obs: No disruption or Disruption
    # column 1 for od: 2 (shortest), 1 (alternative), 0 (except the shortest path) 
    C = np.array([[0, 2], [1, 0], [1, 1]])
    for j, idx in zip(var_ODs_obs, var_ODs.keys()):

        vars_arc[j] = variable.Variable(name=j,
            #B=[{0}, {1}],
            values=['No disruption', 'Disruption'])

        _variables = [vars_arc[k] for k in [j, idx]]
        cpms_arc[j]= cpm.Cpm(variables=_variables,
                             no_child=1,
                             C= C,
                             p= [1, 1, 1])

    # using string
    cpm_ve = operation.condition(cpms_arc,
                           cnd_vars=['od_obs1', 'od_obs2', 'od_obs3'],
                           cnd_states=[1, 1, 0])

    """
    assert len(cpm_ve) == 14
    assert [x.name for x in cpm_ve[10].variables] == ['od_obs1', 'od1']
    np.testing.assert_array_equal(cpm_ve[10].C, np.array([[1, 1], [1, 2]]))
    np.testing.assert_array_almost_equal(cpm_ve[10].p, np.array([[1, 1]]).T)

    assert [x.name for x in cpm_ve[11].variables] == ['od_obs2', 'od2']
    np.testing.assert_array_equal(cpm_ve[11].C, np.array([[1, 1], [1, 2]]))
    np.testing.assert_array_almost_equal(cpm_ve[11].p, np.array([[1, 1]]).T)

    assert [x.name for x in cpm_ve[12].variables] == ['od_obs3', 'od3']
    np.testing.assert_array_equal(cpm_ve[12].C, np.array([[0, 0]]))
    np.testing.assert_array_almost_equal(cpm_ve[12].p, np.array([[1]]).T)

    assert [x.name for x in cpm_ve[13].variables] == ['od_obs4', 'od4']
    np.testing.assert_array_equal(cpm_ve[13].C, np.array([[0, 0], [1, 1], [1, 2]]))
    np.testing.assert_array_almost_equal(cpm_ve[13].p, np.array([[1, 1, 1]]).T)
    """

    Mcond_mult = cpm.product(cpm_ve)

    assert Mcond_mult.C.shape == (8, 14)
    #assert [x.name for x in Mcond_mult.variables] == ['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'od1', 'od2', 'od3', 'od4', 'od_obs1', 'od_obs2', 'od_obs3', 'od_obs4']

    expected_p = np.array([[7.48743285822800e-05,
1.43718130993790e-05,
1.44320505816953e-08,
2.77017153313071e-09,
5.84121979269559e-05,
1.12119762170237e-05,
1.12589696766821e-08,
2.16111197186942e-09]]).T

    #np.testing.assert_array_almost_equal(Mcond_mult.p, expected_p)

    # using string
    Mcond_mult_sum = Mcond_mult.sum(list(var_ODs.keys()) + var_ODs_obs)

    assert Mcond_mult_sum.C.shape == (8, 6)
    #assert [x.name for x in Mcond_mult_sum.variables] == ['e1', 'e2', 'e3', 'e4', 'e5', 'e6']

    expected_p = np.array([[7.48743285822800e-05,
1.43718130993790e-05,
1.44320505816953e-08,
2.77017153313071e-09,
5.84121979269559e-05,
1.12119762170237e-05,
1.12589696766821e-08,
2.16111197186942e-09]]).T

    #np.testing.assert_array_almost_equal(Mcond_mult_sum.p, expected_p)

    # P( X_i = 2 | OD_1 = 2, OD_2 = 2, OD_3 = 1 ), i = 1, ..., N
    arc_fail = 0
    arcs_prob_damage = np.zeros(len(arcs))
    for j, i in enumerate(arcs.keys()):

        iM = Mcond_mult_sum.sum([vars_arc[i]], 0)
        iM_fail = iM.condition(cnd_vars=[vars_arc[i]],
                               cnd_states=[arc_fail])
        fail_prob = iM_fail.p / iM.p.sum(axis=0)
        if fail_prob.size:
            arcs_prob_damage[j] = fail_prob.item()

    # Check if the results are the same
    np.testing.assert_array_almost_equal(arcs_prob_damage,  expected_probs['damage'], decimal=4)

    """
    plt.figure()

    barWidth = 0.25
    r1 = np.arange(len(arcs))
    r2 = [x + barWidth for x in r1]

    plt.bar(r1, 1-arcs_prob_damage, color='#7f6d5f', width=barWidth, edgecolor='white', label='Survival')
    plt.bar(r2, arcs_prob_damage, color='#557f2d', width=barWidth, edgecolor='white', label='Failure')

    plt.xlabel( 'Arc' )
    plt.ylabel( 'Probability' )
    plt.legend()
    plt.savefig(HOME.joinpath('figure2s.png'), dpi=200)
    plt.close()
    """

def plot_delay(ODs_prob_delay, ODs_prob_disconn, var_ODs):

    # plot
    plt.figure()
    nODs = len(var_ODs)
    # set width of bars
    barWidth = 0.25
    # Set position of bar on X axis
    r1 = np.arange(nODs)
    r2 = [x + barWidth for x in r1]

    # Make the plot
    plt.bar(r1, ODs_prob_disconn, color='#7f6d5f', width=barWidth, edgecolor='white', label='Disconnection')
    plt.bar(r2, ODs_prob_delay, color='#557f2d', width=barWidth, edgecolor='white', label='Delay')

    # Add xticks on the middle of the group bars
    plt.xlabel('OD pair', fontweight='bold')
    plt.xticks([r + 0.5*barWidth for r in range(nODs)], [f'{x}' for x in var_ODs.keys()])

    # Create legend & Show graphic
    plt.legend()
    plt.savefig(HOME.joinpath('figure1s.png'), dpi=200)
    plt.close()


def test_get_arcs_length():

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

    result = trans.get_arcs_length(arcs, node_coords)

    expected = {1: 6.0,
                2: 3.6056,
                3: 3.6056,
                4: 3.1623,
                5: 2.8284,
                6: 1.4142}

    pd.testing.assert_series_equal(pd.Series(result), pd.Series(expected), rtol=1.0e-3)


def test_get_all_paths_and_times():

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
        G.add_edge(x[0], x[1], weight=arc_times_h[k], label=k, key=k)

    ODs = [(5, 1), (5, 2), (5, 3), (5, 4)]

    path_time = trans.get_all_paths_and_times(ODs, G)

    expected = {(5, 1): [([2], 0.0901),
                         ([3, 1], 0.2401)],
                (5, 2): [([2, 1], 0.2401),
                         ([3], 0.0901)],
                (5, 3): [([5], 0.0943),
                         ([6, 4], 0.1761)],
                (5, 4): [([5, 4], 0.1997),
                         ([6], 0.0707)],
                }


def test_get_path_time_idx1():

    path_time =[(['e2'], 0.0901), (['e3', 'e1'], 0.24009999999999998)]

    vari = variable.Variable(name='od1', values=[np.inf, 0.2401, 0.0901])

    result = trans.get_path_time_idx(path_time, vari)

    expected = [([], np.inf, 0), (['e2'], 0.0901, 2), (['e3', 'e1'], 0.24009999999999998, 1)]

    assert result==expected

    path_time =[(['e3', 'e1'], 0.24009999999999998), (['e2'], 0.0901)]

    result = trans.get_path_time_idx(path_time, vari)

    assert result==expected


def test_get_path_time_idx2():

    path_time =[(['e2'], 0.0901), (['e3', 'e1'], 0.24009999999999998)]

    vari = variable.Variable(name='od1', values=[np.inf, 0.2401, 0.0901])

    result = trans.get_path_time_idx(path_time, vari)

    expected = [([], np.inf, 0), (['e2'], 0.0901, 2), (['e3', 'e1'], 0.24009999999999998, 1)]

    assert result==expected


def test_eval_sys_state():

    arc_surv = 1
    arc_fail = 0

    arcs = {'e1': ['n1', 'n2'],
            'e2': ['n1', 'n5'],
            'e3': ['n2', 'n5'],
            'e4': ['n3', 'n4'],
            'e5': ['n3', 'n5'],
            'e6': ['n4', 'n5']}

    arc_times_h = {'e1': 0.15, 'e2': 0.0901, 'e3': 0.0901, 'e4': 0.1054,
                   'e5': 0.0943, 'e6': 0.0707}

    G = nx.Graph()
    for k, x in arcs.items():
        G.add_edge(x[0], x[1], weight=arc_times_h[k], label=k)

    ODs = [('n5', 'n1'), ('n5', 'n2'), ('n5', 'n3'), ('n5', 'n4')]

    arcs_state = {'e1': arc_surv,
                  'e2': arc_surv,
                  'e3': arc_surv,
                  'e4': arc_surv,
                  'e5': arc_surv,
                  'e6': arc_surv}

    vars_od1 = variable.Variable(name='od1',
            values=[np.inf, 0.2401, 0.0901])

    path_time = trans.get_all_paths_and_times([ODs[0]], G, 'weight')[ODs[0]]
    path_time.append(([], np.inf))

    # refering variable
    path_time_idx = []
    for x in path_time:
        idx = [i for i, y in enumerate(vars_od1.values) if isclose(x[1], y)]
        try:
            path_time_idx.append((*x, idx[0]))
        except IndexError:
            print('path_time incompatible with variable')

    # sort by increasing number of edges
    path_time_idx = sorted(path_time_idx, key=lambda x: len(x[0]))

    # The shortest route (2) available
    sys_state = trans.eval_sys_state(path_time_idx, arcs_state, arc_surv)
    assert sys_state == 2

    arcs_state = {'e1': arc_surv,
                  'e2': arc_fail,
                  'e3': arc_surv,
                  'e4': arc_surv,
                  'e5': arc_surv,
                  'e6': arc_surv}

    # The second shortest route (1,3) available
    sys_state = trans.eval_sys_state(path_time_idx, arcs_state, arc_surv)
    assert sys_state == 1

    arcs_state = {'e1': arc_surv,
                  'e2': arc_fail,
                  'e3': arc_fail,
                  'e4': arc_surv,
                  'e5': arc_surv,
                  'e6': arc_surv}

    # No path available
    sys_state = trans.eval_sys_state(path_time_idx, arcs_state, arc_surv)
    assert sys_state == 0


def test_get_connectivity_given_comps1(setup_sys_rbd):

    varis, arcs, G = setup_sys_rbd

    comps_st = {f'x{i}': 1 for i in range(1, 9)}
    #comps_st.update({'sink': 1, 'source': 1})
    od_pair = ('source', 'sink')

    #_varis = {k: varis[k] for k in comps_st.keys()}
    path_edges, path_nodes = trans.get_connectivity_given_comps(comps_st,  G, od_pair)
    assert path_nodes == ['source', 'x1', 'x7', 'x8', 'sink']
    assert path_edges == ['e1', 'e10', 'e8', 'e9']

    # failure
    _comps_st = comps_st.copy()
    _comps_st['x7'] = 0
    path_edges, path_nodes = trans.get_connectivity_given_comps(_comps_st,  G, od_pair)
    assert path_edges == []

    # success
    _comps_st = comps_st.copy()
    _comps_st.update({'x2': 0, 'x3': 0, 'x1': 0})
    path_edges, path_nodes = trans.get_connectivity_given_comps(_comps_st, G, od_pair)
    assert path_nodes == ['source', 'x4', 'x5', 'x6', 'x7', 'x8', 'sink']
    assert path_edges == ['e4', 'e5', 'e6', 'e7', 'e8', 'e9']

    # failure
    _comps_st = comps_st.copy()
    _comps_st.update({'x4': 0, 'x2': 0, 'x1': 0, 'x3': 0})
    path_edges, path_nodes = trans.get_connectivity_given_comps(_comps_st, G, od_pair)
    assert path_edges == []

    # success
    _comps_st = comps_st.copy()
    _comps_st.update({'x1': 0, 'x2': 0, 'x3': 0})
    od_pair = ('x4', 'x6')
    path_edges, path_nodes = trans.get_connectivity_given_comps(_comps_st, G, od_pair)
    assert path_nodes == ['x4', 'x5', 'x6']
    assert path_edges == ['e5', 'e6']


def test_get_connectivity_given_comps2(setup_sys_rbd):

    varis, arcs, G = setup_sys_rbd

    comps_st = {f'e{i}': 1 for i in range(1, 13)}
    od_pair = ('source', 'sink')
    #_varis = {k: varis[k] for k in comps_st.keys()}
    path_e, path_n = trans.get_connectivity_given_comps(comps_st,  G, od_pair)
    assert path_n == ['source', 'x1', 'x7', 'x8', 'sink']
    assert path_e == ['e1', 'e10', 'e8', 'e9']

    # failure
    _comps_st = comps_st.copy()
    _comps_st['e8'] = 0
    path_e, path_n = trans.get_connectivity_given_comps(_comps_st,  G, od_pair)
    assert path_n == []

    # failure from node and edge failure
    _comps_st = comps_st.copy()
    _comps_st['x7'] = 0
    #_varis.update({'x7': varis['x7']})
    path_e, path_n = trans.get_connectivity_given_comps(_comps_st,  G, od_pair)
    assert path_n == []


def test_get_connectivity_given_comps4():

    x_star = {'e1': 1, 'e2': 0, 'e3': 1}

    varis = {}
    for i in range(1, 4):
        varis[f'e{i}'] = variable.Variable(name=f'e{i}', values=['Fail', 'Survive'])

    od_pair = ('n1', 'n3')
    edges = {
         'e1': {'origin': 'n1', 'destination': 'n2', 'link_capacity': None, 'weight': 1.0},
         'e2': {'origin': 'n2', 'destination': 'n3', 'link_capacity': None, 'weight': 1.0},
         'e3': {'origin': 'n2', 'destination': 'n3', 'link_capacity': None, 'weight': 1.0},
         }

    G = nx.MultiDiGraph()
    G.add_edge('n1', 'n2', label='e1', key='e1', weight=1)
    G.add_edge('n2', 'n3', label='e2', key='e2', weight=1)
    G.add_edge('n2', 'n3', label='e3', key='e3', weight=1)
    G.add_node('n1', key='n1', label='n1')
    G.add_node('n2', key='n2', label='n2')
    G.add_node('n3', key='n3', label='n3')

    s_path_edges, s_path_nodes = trans.get_connectivity_given_comps(x_star, G, od_pair)

    assert s_path_edges == ['e1', 'e3']
    assert s_path_nodes == ['n1', 'n2', 'n3']

    #sys_fun = trans.sys_fun_wrap(od_pair, edges, varis)

    #sys_val, sys_st, comp_st_min = sys_fun(x_star)

    #assert sys_st == 's'
    #assert sys_val == None
    #assert comp_st_min == {'e1': 1, 'e3': 1}


def test_sf_min_path(setup_sys_rbd):

    varis, arcs, G = setup_sys_rbd

    # edges
    comps_st = {f'e{i}': 1 for i in range(1, 13)}
    od_pair = ('source', 'sink')

    d_time, sys_st, min_comps_st = trans.sf_min_path(comps_st, G, od_pair)
    assert sys_st == 's'
    assert min_comps_st == {'e1': 1, 'e10': 1, 'e8': 1, 'e9':1}

    # nodes
    comps_st = {f'x{i}': 1 for i in range(1, 9)}
    d_time, sys_st, min_comps_st = trans.sf_min_path(comps_st, G, od_pair)
    assert sys_st == 's'
    assert min_comps_st == {'x1': 1, 'x7': 1, 'x8': 1}

    # nodes: failure
    _comps_st = {f'x{i}': 1 for i in range(1, 9)}
    _comps_st['x7'] = 0
    d_time, sys_st, min_comps_st = trans.sf_min_path(_comps_st, G, od_pair)
    assert sys_st == 'f'
    assert min_comps_st == {}


def test_get_time_and_path_multi_dest1(setup_multi_dest):

    G, origin, dests, _, varis = setup_multi_dest

    arcs_state = {'e1': 1,
                  'e2': 0,
                  'e3': 1,
                  'e4': 1,
                  'e5': 1,
                  'e6': 0}

    d_time1, path1, path1e = trans.get_time_and_path_multi_dest(arcs_state, G, origin, dests, varis)

    assert d_time1 == pytest.approx(0.3344, rel=1.0e-4)
    assert path1 == ['n1', 'n2', 'n5', 'n3']
    assert path1e == ['e1', 'e3', 'e5']

    arcs_state['e3'] = 0 # no path available in this case
    d_time2, path2, path2_e = trans.get_time_and_path_multi_dest(arcs_state, G, origin, dests, varis)

    assert d_time2 == np.inf


def test_sys_fun_wrap1(setup_multi_dest):

    G, origin, dests, arcs, varis = setup_multi_dest

    arcs_state = {'e1': 1,
                  'e2': 0,
                  'e3': 1,
                  'e4': 1,
                  'e5': 1,
                  'e6': 0}

    od_pair = {'origin': origin, 'dests': dests}
    sys_fun = trans.sys_fun_wrap(G, od_pair, varis=varis, thres=0.5)
    d_time1, sys_st1, min_comps_st1 = sys_fun(arcs_state)

    assert d_time1 == pytest.approx(0.3344, rel=1.0e-4)
    assert sys_st1 == 's'
    assert min_comps_st1 == {'e1': 1, 'e3': 1, 'e5': 1}

    arcs_state['e3'] = 0
    d_time2, sys_st2, min_comps_st2 = sys_fun(arcs_state)

    assert d_time2 == np.inf
    assert sys_st2 == 'f'
    assert min_comps_st2 == {}


def test_get_edge_from_nodes():

    G = nx.MultiGraph()
    G.add_edge('n1', 'n2', weight=1, label='e1')
    G.add_edge('n2', 'n3', weight=1, label='e2')
    G.add_edge('n2', 'n3', weight=1, label='e3')

    G.add_node('n1', pos=(0, 0), label='n1')
    G.add_node('n2', pos=(1, 0), label='n2')
    G.add_node('n3', pos=(2, 0), label='n3')

    comps_st = {'e1': 1, 'e2': 1, 'e3': 1}
    od_pair = ('n1', 'n3')
    path_nodes = ['n1', 'n2', 'n3']
    path_edge = trans.get_edge_from_nodes(G, path_nodes)

    assert path_edge == ['e1', 'e2']


