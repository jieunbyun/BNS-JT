'''
Ji-Eun Byun (ji-eun.byun@glasgow.ac.uk)
Created: 27 Mar 2023

A small, hypothetical bridge system
'''
import pytest
import numpy as np
import networkx as nx
from scipy.stats import lognorm
from pathlib import Path
import matplotlib
import pdb

matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

from BNS_JT import cpm, variable
from Trans.trans import get_arcs_length, get_all_paths_and_times

HOME = Path(__file__).absolute().parent

expected_disconn = np.array([0.0096, 0.0011, 0.2102, 0.2102])
expected_delay = np.array([0.0583, 0.0052, 0.4795, 0.4382])
expected_damage = np.array([0.1610,  1,  1,  0.0002,   0,  0.4382])

## Data
# Network
node_coords = {'n1': (-2, 3),
               'n2': (-2, -3),
               'n3': (2, -2),
               'n4': (1, 1),
               'n5': (0, 0)}

arcs = {'e1': ['n1', 'n2'],
	'e2': ['n1', 'n5'],
	'e3': ['n2', 'n5'],
	'e4': ['n3', 'n4'],
	'e5': ['n3', 'n5'],
	'e6': ['n4', 'n5']}

# Fragility curves -- From HAZUS-EQ model (roads are regarded as disconnected when being extensively or completely damaged)

frag = {'major': {'med': 60.0, 'std': 0.7},
        'urban' : {'med': 24.0, 'std': 0.7},
        'bridge': {'med': 1.1, 'std': 3.9},
        }

arcs_type = {'e1': 'major',
             'e2': 'major',
             'e3': 'major',
             'e4': 'urban',
             'e5': 'bridge',
             'e6': 'bridge'}

arcs_avg_kmh = {'e1': 40,
                'e2': 40,
                'e3': 40,
                'e4': 30,
                'e5': 30,
                'e6': 20}

var_ODs = {'OD1': ('n5', 'n1'),
           'OD2': ('n5', 'n2'),
           'OD3': ('n5', 'n3'),
           'OD4': ('n5', 'n4')}

nODs = len(var_ODs)

# For the moment, we assume that ground motions are observed. Later, hazard nodes will be added.
GM_obs = {'e1': 30.0,
          'e2': 20.0,
          'e3': 10.0,
          'e4': 2.0,
          'e5': 0.9,
          'e6': 0.6}

# Arcs' states index compatible with variable B index, and C
arc_surv = 1 - 1
arc_fail = 2 - 1
arc_either = 3 - 1

@pytest.fixture(scope='package')
def setup_bridge():

    arc_lens_km = get_arcs_length(arcs, node_coords)

    #arcTimes_h = arcLens_km ./ arcs_Vavg_kmh
    arc_times_h = {k: v/arcs_avg_kmh[k] for k, v in arc_lens_km.items()}

    # create a graph
    G = nx.Graph()
    for k, x in arcs.items():
        G.add_edge(x[0], x[1], time=arc_times_h[k], label=k)

    for k, v in node_coords.items():
        G.add_node(k, pos=v)

    path_time = get_all_paths_and_times(var_ODs.values(), G, key='time')

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

    B = np.array([[1, 0], [0, 1], [1, 1]])

    for k in arcs.keys():
        vars_arc[k] = variable.Variable(name=str(k), B=B, values=['Surv', 'Fail'])

        _type = arcs_type[k]
        prob = lognorm.cdf(GM_obs[k], frag[_type]['std'], scale=frag[_type]['med'])
        C = np.array([[arc_surv, arc_fail]]).T
        p = np.array([1-prob, prob])
        cpms_arc[k] = cpm.Cpm(variables = [vars_arc[k]],
                              no_child = 1,
                              C = C,
                              p = p)

    # Travel times (systems): P(OD_j | X1, ... Xn) j = 1 ... nOD
    B_ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    vars_arc['OD1'] = variable.Variable(name='OD1', B=B_,
            values=[0.0901, 0.2401, np.inf])

    vars_arc['OD2'] = variable.Variable(name='OD2', B=B_,
            values=[0.0901, 0.2401, np.inf])

    vars_arc['OD3'] = variable.Variable(name='OD3', B=B_,
            values=[0.0943, 0.1761, np.inf])

    vars_arc['OD4'] = variable.Variable(name='OD4', B=B_,
            values=[0.0707, 0.1997, np.inf])

    _variables = [vars_arc[k] for k in ['OD1', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6']]
    c7 = np.array([
    [1,3,1,3,3,3,3],
    [2,1,2,1,3,3,3],
    [3,1,2,2,3,3,3],
    [3,2,2,3,3,3,3]]) - 1
    cpms_arc['OD1'] = cpm.Cpm(variables= _variables,
                           no_child = 1,
                           C = c7,
                           p = [1, 1, 1, 1],
                           )

    _variables = [vars_arc[k] for k in ['OD2', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6']]
    c8 = np.array([
    [1,3,3,1,3,3,3],
    [2,1,1,2,3,3,3],
    [3,1,2,2,3,3,3],
    [3,2,3,2,3,3,3]]) - 1
    cpms_arc['OD2'] = cpm.Cpm(variables= _variables,
                           no_child = 1,
                           C = c8,
                           p = [1, 1, 1, 1],
                           )

    _variables = [vars_arc[k] for k in ['OD3', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6']]
    c9 = np.array([
    [1,3,3,3,3,1,3],
    [2,3,3,3,1,2,1],
    [3,3,3,3,1,2,2],
    [3,3,3,3,2,2,3]]) - 1
    cpms_arc['OD3'] = cpm.Cpm(variables= _variables,
                           no_child = 1,
                           C = c9,
                           p = [1, 1, 1, 1],
                           )

    _variables = [vars_arc[k] for k in ['OD4', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6']]
    c10 = np.array([
    [1,3,3,3,3,3,1],
    [2,3,3,3,1,1,2],
    [3,3,3,3,1,2,2],
    [3,3,3,3,2,3,2]]) - 1
    cpms_arc['OD4'] = cpm.Cpm(variables= _variables,
                           no_child = 1,
                           C = c10,
                           p = [1, 1, 1, 1],
                           )

    return cpms_arc, vars_arc


def test_prob_delay1(setup_bridge):

    ## Inference - by variable elimination (would not work for large-scale systems)
    # Probability of delay and disconnection
    # Becomes P(OD_1, ..., OD_M) since X_1, ..., X_N are eliminated
    #cpms_arc_cp = cpms_arc.values()

    cpms_arc, vars_arc = setup_bridge
    cpms_arc_cp = list(cpms_arc.values())

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
        #disconn_state = vars_arc[idx].B.shape[0] - 1
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

    plot_delay(ODs_prob_delay, ODs_prob_disconn)


def test_prob_delay2(setup_bridge):

    cpms_arc, vars_arc = setup_bridge

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
    np.testing.assert_array_almost_equal(ODs_prob_disconn2, expected_disconn, decimal=4)
    np.testing.assert_array_almost_equal(ODs_prob_delay2, expected_delay, decimal=4)


@pytest.mark.skip('FIXME')
def test_prob_damage(setup_bridge):

    cpms_arc, vars_arc = setup_bridge

    # City 1 and 2 experienced a disruption in getting resources, City 3 was okay and 4 is unknown. Probability of damage of roads?
    # A composite state needs be created for City 1 and City 2
    # cf. 
    #B_ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    #vars_arc['OD1'] = variable.Variable(name='OD1', B=B_,
    #        values=[0.0901, 0.2401, np.inf])
    for idx in var_ODs.keys():
        _B = np.vstack([vars_arc[idx].B, [0, 1, 1]])
        vars_arc[idx] = variable.Variable(
                name=idx,
                B=_B,
                values=vars_arc[idx].values)

    # # Add observation nodes P( O_j | OD_j ), j = 1, ..., M
    var_ODs_obs = [f'{i}_obs' for i in var_ODs.keys()]
    # column 0: No disruption or Disruption
    # column 1: 0 (shortest), 1 (alternative), 2 (except the shortest path) 
    C = np.array([[1, 1], [2, 2], [2, 3]]) - 1
    for j, idx in zip(var_ODs_obs, var_ODs.keys()):

        vars_arc[j] = variable.Variable(name=j,
            B=np.eye(2, dtype=int),
            values=['No disruption', 'Disruption'])

        _variables = [vars_arc[k] for k in [j, idx]]
        cpms_arc[j]= cpm.Cpm(variables=_variables,
                             no_child=1,
                             C= C,
                             p= [1, 1, 1])

    cpm_ve = cpm.condition(cpms_arc,
                           cnd_vars=[vars_arc[k] for k in ['OD1_obs', 'OD2_obs', 'OD3_obs']],
                           cnd_states=[1, 1, 0])
    Mcond_mult = cpm.prod_cpms(cpm_ve)
    Mcond_mult_sum = Mcond_mult.sum([vars_arc[k] for k in list(var_ODs.keys()) + var_ODs_obs])

    # P( X_i = 2 | OD_1 = 2, OD_2 = 2, OD_3 = 1 ), i = 1, ..., N
    arcs_prob_damage = np.zeros(len(arcs))
    for j, i in enumerate(arcs.keys()):

        iM = Mcond_mult_sum.sum([vars_arc[i]], 0)
        [iM_fail] = cpm.condition(iM,
                               cnd_vars=[vars_arc[i]],
                               cnd_states=[arc_fail - 1],
                               )
        fail_prob = iM_fail.p / iM.p.sum(axis=0)
        if fail_prob.any():
            arcs_prob_damage[j] = fail_prob

    # Check if the results are the same
    np.testing.assert_array_almost_equal(arcs_prob_damage,  expected_damage, decimal=4)

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


def plot_delay(ODs_prob_delay, ODs_prob_disconn):

    # plot
    plt.figure()

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

