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

from BNS_JT import cpm, variable, config, branch, model, trans


HOME = Path(__file__).absolute().parent


@pytest.fixture(scope='package')
def data_bridge():
    """
    system with 6 edges, 5 nodes
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
    """
    arcs = data_bridge['arcs']
    node_coords = data_bridge['node_coords']
    arcs_avg_kmh = data_bridge['arcs_avg_kmh']
    arcs_type = data_bridge['arcs_type']
    GM_obs = data_bridge['GM_obs']
    frag = data_bridge['frag']
    var_ODs = data_bridge['var_ODs']

    # Arcs' states index compatible with variable B index, and C
    arc_surv = 0
    arc_fail = 1
    arc_either = 2

    arc_lens_km = trans.get_arcs_length(arcs, node_coords)

    #arcTimes_h = arcLens_km ./ arcs_Vavg_kmh
    arc_times_h = {k: v/arcs_avg_kmh[k] for k, v in arc_lens_km.items()}

    # create a graph
    G = nx.Graph()
    for k, x in arcs.items():
        G.add_edge(x[0], x[1], time=arc_times_h[k], label=k)

    for k, v in node_coords.items():
        G.add_node(k, pos=v)

    # plot graph
    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = nx.get_edge_attributes(G, 'label')

    fig = plt.figure()
    ax = fig.add_subplot()
    nx.draw(G, pos, with_labels=True, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    fig.savefig(HOME.joinpath('graph.png'), dpi=200)

    # Arcs (components): P(X_i | GM = GM_ob ), i = 1 .. N (= nArc)
    cpms_arc = {}
    vars_arc = {}

    #path_time = trans.get_all_paths_and_times(var_ODs.values(), G, key='time')

    # number of component states: 2 ('surv' or 'fail')
    for k in arcs.keys():
        vars_arc[k] = variable.Variable(name=str(k), B=[{0}, {1}, {0, 1}], values=['Surv', 'Fail'])

        _type = arcs_type[k]
        prob = lognorm.cdf(GM_obs[k], frag[_type]['std'], scale=frag[_type]['med'])
        C = np.array([[arc_surv, arc_fail]]).T
        p = np.array([1-prob, prob])
        cpms_arc[k] = cpm.Cpm(variables = [vars_arc[k]],
                              no_child = 1,
                              C = C,
                              p = p)

    # Travel times (systems): P(OD_j | X1, ... Xn) j = 1 ... nOD
    # e.g., for 'od1': 'e2': 0.0901, 'e3'-'e1': 0.2401
    vars_arc['od1'] = variable.Variable(name='od1', B=[{0}, {1}, {2}],
            values=[0.0901, 0.2401, np.inf])

    vars_arc['od2'] = variable.Variable(name='od2', B=[{0}, {1}, {2}],
            values=[0.0901, 0.2401, np.inf])

    vars_arc['od3'] = variable.Variable(name='od3', B=[{0}, {1}, {2}],
            values=[0.0943, 0.1761, np.inf])

    vars_arc['od4'] = variable.Variable(name='od4', B=[{0}, {1}, {2}],
            values=[0.0707, 0.1997, np.inf])

    _variables = [vars_arc[k] for k in ['od1', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6']]
    c7 = np.array([
    [1,3,1,3,3,3,3],
    [2,1,2,1,3,3,3],
    [3,1,2,2,3,3,3],
    [3,2,2,3,3,3,3]]) - 1

    cpms_arc['od1'] = cpm.Cpm(variables= _variables,
                           no_child = 1,
                           C = c7,
                           p = [1, 1, 1, 1],
                           )

    _variables = [vars_arc[k] for k in ['od2', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6']]
    c8 = np.array([
    [1,3,3,1,3,3,3],
    [2,1,1,2,3,3,3],
    [3,1,2,2,3,3,3],
    [3,2,3,2,3,3,3]]) - 1

    cpms_arc['od2'] = cpm.Cpm(variables= _variables,
                           no_child = 1,
                           C = c8,
                           p = [1, 1, 1, 1],
                           )

    _variables = [vars_arc[k] for k in ['od3', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6']]
    c9 = np.array([
    [1,3,3,3,3,1,3],
    [2,3,3,3,1,2,1],
    [3,3,3,3,1,2,2],
    [3,3,3,3,2,2,3]]) - 1
    cpms_arc['od3'] = cpm.Cpm(variables= _variables,
                           no_child = 1,
                           C = c9,
                           p = [1, 1, 1, 1],
                           )

    _variables = [vars_arc[k] for k in ['od4', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6']]
    c10 = np.array([
    [1,3,3,3,3,3,1],
    [2,3,3,3,1,1,2],
    [3,3,3,3,1,2,2],
    [3,3,3,3,2,3,2]]) - 1
    cpms_arc['od4'] = cpm.Cpm(variables= _variables,
                           no_child = 1,
                           C = c10,
                           p = [1, 1, 1, 1],
                           )

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
    arc_fail = 0
    arc_surv = 1
    arc_either = 2

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
        vars_arc[k] = variable.Variable(name=str(k), B=[{0}, {1}, {0, 1}], values=['Fail', 'Surv'])

        _type = arcs_type[k]
        prob = lognorm.cdf(GM_obs[k], frag[_type]['std'], scale=frag[_type]['med'])
        C = np.array([[arc_fail, arc_surv]]).T
        p = np.array([prob, 1-prob])
        cpms_arc[k] = cpm.Cpm(variables = [vars_arc[k]],
                              no_child = 1,
                              C = C,
                              p = p)

    # Travel times (systems): P(OD_j | X1, ... Xn) j = 1 ... nOD
    # e.g., for 'od1': 'e2': 0.0901, 'e3'-'e1': 0.2401
    vars_arc['od1'] = variable.Variable(name='od1', B=[{0}, {1}, {2}],
            values=[0.0901, 0.2401, np.inf])

    vars_arc['od2'] = variable.Variable(name='od2', B=[{0}, {1}, {2}],
            values=[0.0901, 0.2401, np.inf])

    vars_arc['od3'] = variable.Variable(name='od3', B=[{0}, {1}, {2}],
            values=[0.0943, 0.1761, np.inf])

    vars_arc['od4'] = variable.Variable(name='od4', B=[{0}, {1}, {2}],
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


def test_prob_delay1(setup_bridge, expected_probs):

    ## Inference - by variable elimination (would not work for large-scale systems)
    # Probability of delay and disconnection
    # Becomes P(OD_1, ..., OD_M) since X_1, ..., X_N are eliminated
    #cpms_arc_cp = cpms_arc.values()

    cpms_arc, vars_arc, arcs, var_ODs = setup_bridge
    cpms_arc_cp = list(cpms_arc.values())
    nODs = len(var_ODs)

    # prod_cpms
    # get different variables order
    for i in arcs.keys():

        is_inscope = cpm.isinscope([vars_arc[i]], cpms_arc_cp)
        cpm_sel = [y for x, y in zip(is_inscope, cpms_arc_cp) if x]
        cpm_mult = cpm.prod_cpms(cpm_sel)
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

    disconn_state = 3 - 1
    for j, idx in enumerate(var_ODs):

        # Prob. of disconnection
        [cpm_ve] = cpm.condition(cpms_arc_cp,
                               cnd_vars=[vars_arc[idx]],
                               cnd_states=[disconn_state])
        ODs_prob_disconn[j] = cpm_ve.p.sum(axis=0)

        # Prob. of delay
        var_loc = cpms_arc_cp[0].variables.index(vars_arc[idx])
        # except the shortest path (which is state 0)
        rows_to_keep = np.where(cpms_arc_cp[0].C[:, var_loc] > 0)[0]
        cpm_ve = cpms_arc_cp[0].get_subset(rows_to_keep)
        ODs_prob_delay[j] = cpm_ve.p.sum(axis=0)

        # Prob. of disconnection alternative
        rows_to_keep = np.where(cpms_arc_cp[0].C[:, var_loc] == disconn_state)[0]
        assert cpms_arc_cp[0].get_subset(rows_to_keep).p.sum() == ODs_prob_disconn[j]

    #plot_delay(ODs_prob_delay, ODs_prob_disconn, var_ODs)

    # Check if the results are the same
    np.testing.assert_array_almost_equal(ODs_prob_disconn, expected_probs['disconn'], decimal=4)
    np.testing.assert_array_almost_equal(ODs_prob_delay, expected_probs['delay'], decimal=4)


def test_prob_delay2(setup_bridge, expected_probs):

    cpms_arc, vars_arc, arcs, var_ODs = setup_bridge

    nODs = len(var_ODs)

    ## Repeat inferences again using new functions -- the results must be the same.
    # Probability of delay and disconnection
    M = [cpms_arc[k] for k in list(arcs.keys()) + list(var_ODs.keys())]
    var_elim_order = [vars_arc[i] for i in arcs.keys()]
    M_VE2 = cpm.variable_elim(M, var_elim_order)

    # "M_VE2" same as "M_VE"
    # Retrieve example results
    ODs_prob_disconn2 = np.zeros(nODs)
    ODs_prob_delay2 = np.zeros(nODs)

    disconn_state = 2
    for j, idx in enumerate(var_ODs):

        # Prob. of disconnection
        # FIXME 2 -> 3?? 
        #disconn_state = vars_arc[idx].values.index(np.inf) + 1
        # the state of disconnection is assigned an arbitrarily large number 100
        ODs_prob_disconn2[j] = cpm.get_prob(M_VE2, [vars_arc[idx]], [2])

        # Prob. of delay
        ODs_prob_delay2[j] = cpm.get_prob(M_VE2, [vars_arc[idx]], [1-1], flag=False) # Any state greater than 1 means delay.

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
    M_VE2 = cpm.variable_elim(M, var_elim_order)

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
        ODs_prob_disconn2[j] = cpm.get_prob(M_VE2, [vars_arc[idx]], [2])

        # Prob. of delay
        ODs_prob_delay2[j] = cpm.get_prob(M_VE2, [vars_arc[idx]], [1-1], flag=False) # Any state greater than 1 means delay.

    # Check if the results are the same
    np.testing.assert_array_almost_equal(ODs_prob_delay2[0], expected_probs['delay'][0], decimal=4)
    np.testing.assert_array_almost_equal(ODs_prob_disconn2[0], expected_probs['disconn'][0], decimal=4)


def test_prob_damage(setup_bridge, expected_probs):

    arc_fail = 1
    arc_surv = 0

    cpms_arc, vars_arc, arcs, var_ODs = setup_bridge

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
    # column 1 for od: 0 (shortest), 1 (alternative), 2 (except the shortest path) 
    C = np.array([[1, 1], [2, 2], [2, 3]]) - 1
    for j, idx in zip(var_ODs_obs, var_ODs.keys()):

        vars_arc[j] = variable.Variable(name=j,
            B=[{0}, {1}],
            values=['No disruption', 'Disruption'])

        _variables = [vars_arc[k] for k in [j, idx]]
        cpms_arc[j]= cpm.Cpm(variables=_variables,
                             no_child=1,
                             C= C,
                             p= [1, 1, 1])

    # using string
    cpm_ve = cpm.condition(cpms_arc,
                           cnd_vars=['od_obs1', 'od_obs2', 'od_obs3'],
                           cnd_states=[1, 1, 0])

    assert len(cpm_ve) == 14
    assert [x.name for x in cpm_ve[10].variables] == ['od_obs1', 'od1']
    np.testing.assert_array_equal(cpm_ve[10].C, np.array([[2, 2], [2, 3]]) - 1)
    np.testing.assert_array_almost_equal(cpm_ve[10].p, np.array([[1, 1]]).T)

    assert [x.name for x in cpm_ve[11].variables] == ['od_obs2', 'od2']
    np.testing.assert_array_equal(cpm_ve[11].C, np.array([[2, 2], [2, 3]]) - 1)
    np.testing.assert_array_almost_equal(cpm_ve[11].p, np.array([[1, 1]]).T)

    assert [x.name for x in cpm_ve[12].variables] == ['od_obs3', 'od3']
    np.testing.assert_array_equal(cpm_ve[12].C, np.array([[1, 1]]) - 1)
    np.testing.assert_array_almost_equal(cpm_ve[12].p, np.array([[1]]).T)

    assert [x.name for x in cpm_ve[13].variables] == ['od_obs4', 'od4']
    np.testing.assert_array_equal(cpm_ve[13].C, np.array([[1, 1], [2, 2], [2, 3]]) - 1)
    np.testing.assert_array_almost_equal(cpm_ve[13].p, np.array([[1, 1, 1]]).T)


    Mcond_mult = cpm.prod_cpms(cpm_ve)

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
    arcs_prob_damage = np.zeros(len(arcs))
    for j, i in enumerate(arcs.keys()):

        iM = Mcond_mult_sum.sum([vars_arc[i]], 0)
        [iM_fail] = cpm.condition(iM,
                               cnd_vars=[vars_arc[i]],
                               cnd_states=[arc_fail],
                               )
        fail_prob = iM_fail.p / iM.p.sum(axis=0)
        if fail_prob.size:
            arcs_prob_damage[j] = fail_prob

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
        G.add_edge(x[0], x[1], time=arc_times_h[k], label=k)

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

    vari = variable.Variable(name='od1', B=[{0}, {1}, {2}], values=[np.inf, 0.2401, 0.0901])

    result = trans.get_path_time_idx(path_time, vari)

    expected = [([], np.inf, 0), (['e2'], 0.0901, 2), (['e3', 'e1'], 0.24009999999999998, 1)]

    assert result==expected

    path_time =[(['e3', 'e1'], 0.24009999999999998), (['e2'], 0.0901)]

    result = trans.get_path_time_idx(path_time, vari)

    assert result==expected


def test_get_path_time_idx2():

    path_time =[(['e2'], 0.0901), (['e3', 'e1'], 0.24009999999999998)]

    vari = variable.Variable(name='od1', B=[{0}, {1}, {2}], values=[np.inf, 0.2401, 0.0901])

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
        G.add_edge(x[0], x[1], time=arc_times_h[k], label=k)

    ODs = [('n5', 'n1'), ('n5', 'n2'), ('n5', 'n3'), ('n5', 'n4')]

    arcs_state = {'e1': arc_surv,
                  'e2': arc_surv,
                  'e3': arc_surv,
                  'e4': arc_surv,
                  'e5': arc_surv,
                  'e6': arc_surv}

    vars_od1 = variable.Variable(name='od1', B=[{0}, {1}, {2}],
            values=[np.inf, 0.2401, 0.0901])

    path_time = trans.get_all_paths_and_times([ODs[0]], G, 'time')[ODs[0]]
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

