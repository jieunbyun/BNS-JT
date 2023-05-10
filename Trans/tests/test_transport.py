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

## Data
# Network
node_coords = {'1': (-2, 3),
               '2': (-2, -3),
               '3': (2, -2),
               '4': (1, 1),
               '5': (0, 0)}

arcs = {'1': ['1', '2'],
	'2': ['1', '5'],
	'3': ['2', '5'],
	'4': ['3', '4'],
	'5': ['3', '5'],
	'6': ['4', '5']}

# Fragility curves -- From HAZUS-EQ model (roads are regarded as disconnected when being extensively or completely damaged)

frag = {'major': {'med': 60.0, 'std': 0.7},
        'urban' : {'med': 24.0, 'std': 0.7},
        'bridge': {'med': 1.1, 'std': 3.9},
        }

arcs_type = {'1': 'major',
             '2': 'major',
             '3': 'major',
             '4': 'urban',
             '5': 'bridge',
             '6': 'bridge'}

arcs_avg_kmh = {'1': 40,
                '2': 40,
                '3': 40,
                '4': 30,
                '5': 30,
                '6': 20}

ODs = [('5', '1'), ('5', '2'), ('5', '3'), ('5', '4')]

# For the moment, we assume that ground motions are observed. Later, hazard nodes will be added.
GM_obs = {'1': 30.0,
          '2': 20.0,
          '3': 10.0,
          '4': 2.0,
          '5': 0.9,
          '6': 0.6}

# Arcs' states index
arc_surv = 1
arc_fail = 2
arc_either = 3

@pytest.fixture(scope='package')
def main_bridge():

    arc_lens_km = get_arcs_length(arcs, node_coords)

    #arcTimes_h = arcLens_km ./ arcs_Vavg_kmh
    arc_times_h = {k: v/arcs_avg_kmh[k] for k, v in arc_lens_km.items()}

    # create a graph
    G = nx.Graph()
    for k, x in arcs.items():
        G.add_edge(x[0], x[1], time=arc_times_h[k], label=k)

    for k, v in node_coords.items():
        G.add_node(k, pos=v)

    path_time = get_all_paths_and_times(ODs, G, key='time')

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
    vars_arc['7'] = variable.Variable(name='7', B=B_,
            values=[0.0901, 0.2401, np.inf])

    vars_arc['8'] = variable.Variable(name='8', B=B_,
            values=[0.0901, 0.2401, np.inf])

    vars_arc['9'] = variable.Variable(name='9', B=B_,
            values=[0.0943, 0.1761, np.inf])

    vars_arc['10'] = variable.Variable(name='10', B=B_,
            values=[0.0707, 0.1997, np.inf])

    c7 = np.array([
    [1,3,1,3,3,3,3],
    [2,1,2,1,3,3,3],
    [3,1,2,2,3,3,3],
    [3,2,2,3,3,3,3]])

    c8 = np.array([
    [1,3,3,1,3,3,3],
    [2,1,1,2,3,3,3],
    [3,1,2,2,3,3,3],
    [3,2,3,2,3,3,3]])

    c9 = np.array([
    [1,3,3,3,3,1,3],
    [2,3,3,3,1,2,1],
    [3,3,3,3,1,2,2],
    [3,3,3,3,2,2,3]])

    c10 = np.array([
    [1,3,3,3,3,3,1],
    [2,3,3,3,1,1,2],
    [3,3,3,3,1,2,2],
    [3,3,3,3,2,3,2]])

    _variables = [vars_arc[k] for k in ['7', '1', '2', '3', '4', '5', '6']]
    cpms_arc['7'] = cpm.Cpm(variables= _variables,
                           no_child = 1,
                           C = c7,
                           p = [1, 1, 1, 1],
                           )

    _variables = [vars_arc[k] for k in ['8', '1', '2', '3', '4', '5', '6']]
    cpms_arc['8'] = cpm.Cpm(variables= _variables,
                           no_child = 1,
                           C = c8,
                           p = [1, 1, 1, 1],
                           )

    _variables = [vars_arc[k] for k in ['9', '1', '2', '3', '4', '5', '6']]
    cpms_arc['9'] = cpm.Cpm(variables= _variables,
                           no_child = 1,
                           C = c9,
                           p = [1, 1, 1, 1],
                           )

    _variables = [vars_arc[k] for k in ['10', '1', '2', '3', '4', '5', '6']]
    cpms_arc['10'] = cpm.Cpm(variables= _variables,
                           no_child = 1,
                           C = c10,
                           p = [1, 1, 1, 1],
                           )

    ## Inference - by variable elimination (would not work for large-scale systems)
    # Probability of delay and disconnection
    # Becomes P(OD_1, ..., OD_M) since X_1, ..., X_N are eliminated
    cpms_arc_cp = cpms_arc.values()

    #pdb.set_trace()

    # prod_cpms
    # get different variables order
    for i in arcs.keys():

        is_inscope = cpm.isinscope([vars_arc[i]], cpms_arc_cp)
        cpm_sel = [y for x, y in zip(is_inscope, cpms_arc_cp) if x]
        cpm_mult = cpm.prod_cpms(cpm_sel)
        cpm_mult = cpm_mult.sum([vars_arc[i]])

        cpms_arc_cp = [y for x, y in zip(is_inscope, cpms_arc_cp) if x == False]
        cpms_arc_cp.insert(0, cpm_mult)

    # Retrieve example results
    # P( OD_j = 3 ), j = 1, ..., M, where State 3 indicates disconnection
    ODs_prob_disconn = np.zeros(len(ODs))
    # P( (OD_j = 2) U (OD_j = 2) ), j = 1, ..., M, where State 2 indicates the use of the second shortest path (or equivalently, P( (OD_j = 1)^c ), where State 1 indicates the use of the shortest path)
    ODs_prob_delay = np.zeros(len(ODs))

    var_OD = ['7', '8', '9', '10']
    for j, idx in enumerate(var_OD):

        # Prob. of disconnection
        disconn_state = vars_arc[idx].B.shape[0]
        [cpm_ve] = cpm.condition(cpms_arc_cp,
                               cnd_vars=[vars_arc[idx]],
                               cnd_states=[disconn_state])
        ODs_prob_disconn[j] = cpm_ve.p.sum(axis=0)

        # Prob. of delay
        var_loc = cpms_arc_cp[0].variables.index(vars_arc[idx])
        rows_to_keep = np.where(cpms_arc_cp[0].C[:, var_loc] > 1)[0]
        cpm_ve = cpms_arc_cp[0].get_subset(rows_to_keep)
        ODs_prob_delay[j] = cpm_ve.p.sum(axis=0)

    # City 1 and 2 experienced a disruption in getting resources, City 3 was okay and 4 is unknown. Probability of damage of roads?
    # A composite state needs be created for City 1 and City 2
    for idx in var_OD:
        _B = np.vstack([vars_arc[idx].B, [0, 1, 1]])
        vars_arc[idx] = variable.Variable(
                name=str(idx),
                B=_B,
                values=vars_arc[idx].values)

    # # Add observation nodes P( O_j | OD_j ), j = 1, ..., M
    var_OD_obs = []
    for j, (idx, od) in enumerate(zip(var_OD, ODs), start=11):

        #iPaths = arcPaths[iOdInd]
        #iTimes = arcPaths_time[iOdInd]
        vars_arc[str(j)] = variable.Variable(name=str(j), B=np.eye(2, dtype=int), values=['No disruption', 'Disruption'])

        _variables = [vars_arc[k] for k in [str(j), idx]]
        cpms_arc[str(j)]= cpm.Cpm(variables=_variables,
                               no_child=1,
                               C= np.array([[1, 1], [2, 2], [2, 3]]),
                               p= [1, 1, 1])
        var_OD_obs.append(str(j))

    cpm_ve = cpm.condition(cpms_arc,
                                     cnd_vars=[vars_arc[k] for k in ['11', '12', '13']],
                                     cnd_states=[2, 2, 1])
    Mcond_mult = cpm.prod_cpms(cpm_ve)
    Mcond_mult_sum = Mcond_mult.sum([vars_arc[k] for k in var_OD + var_OD_obs])

    # P( X_i = 2 | OD_1 = 2, OD_2 = 2, OD_3 = 1 ), i = 1, ..., N
    arcs_prob_damage = np.zeros(len(arcs))
    for j, i in enumerate(arcs.keys()):

        iM = Mcond_mult_sum.sum([vars_arc[i]], 0)
        [iM_fail] = cpm.condition(iM,
                               cnd_vars=[vars_arc[i]],
                               cnd_states=[arc_fail],
                               )
        fail_prob = iM_fail.p / iM.p.sum(axis=0)
        if fail_prob.any():
            arcs_prob_damage[j] = fail_prob


    ## Repeat inferences again using new functions -- the results must be the same.
    # Probability of delay and disconnection
    M = list(cpms_arc.values())[:10]
    var_elim_order = [vars_arc[str(i)] for i in range(1, 7)]
    M_VE2 = cpm.variable_elim(M, var_elim_order)

    # "M_VE2" same as "M_VE"
    # Retrieve example results
    ODs_prob_disconn2 = np.zeros(len(ODs))
    ODs_prob_delay2 = np.zeros(len(ODs))

    for j, idx in enumerate(var_OD):

        # Prob. of disconnection
        # FIXME 2 -> 3?? 
        disconn_state = np.where(vars_arc[idx].values == np.inf)[0] + 1
        # the state of disconnection is assigned an arbitrarily large number 100
        ODs_prob_disconn2[j] = cpm.get_prob(M_VE2, [vars_arc[idx]], np.array([[3]]))
        # Prob. of delay
        ODs_prob_delay2[j] = cpm.get_prob(M_VE2, [vars_arc[idx]], np.array([[1]]), flag=False) # Any state greater than 1 means delay.

    plot_figs(ODs_prob_delay, ODs_prob_disconn, arcs_prob_damage)

    return ODs_prob_delay, ODs_prob_disconn, ODs_prob_delay2, ODs_prob_disconn2, arcs_prob_damage, cpms_arc, vars_arc


def test_prob(main_bridge):

    ODs_prob_delay, ODs_prob_disconn, ODs_prob_delay2, ODs_prob_disconn2, _, _, _ = main_bridge

    # Check if the results are the same
    try:
        np.testing.assert_array_equal(ODs_prob_disconn, ODs_prob_disconn2)
    except AssertionError:
        print('Prob_disconn')
        print(ODs_prob_disconn)
        print(ODs_prob_disconn2)
    try:
        np.testing.assert_array_equal(ODs_prob_delay, ODs_prob_delay2)
    except AssertionError:
        print('Prob_delay')
        print(ODs_prob_delay)
        print(ODs_prob_delay2)


def plot_figs(ODs_prob_delay, ODs_prob_disconn, arcs_prob_damage):

    # plot
    plt.figure()

    # set width of bars
    barWidth = 0.25

    # Set position of bar on X axis
    r1 = np.arange(len(ODs))
    r2 = [x + barWidth for x in r1]

    # Make the plot
    plt.bar(r1, ODs_prob_disconn, color='#7f6d5f', width=barWidth, edgecolor='white', label='Disconnection')
    plt.bar(r2, ODs_prob_delay, color='#557f2d', width=barWidth, edgecolor='white', label='Delay')

    # Add xticks on the middle of the group bars
    plt.xlabel('OD pair', fontweight='bold')
    plt.xticks([r + 0.5*barWidth for r in range(len(ODs))], [f'{x}' for x in ODs])

    # Create legend & Show graphic
    plt.legend()
    plt.savefig(HOME.joinpath('figure1s.png'), dpi=200)
    plt.close()

    plt.figure()
    r1 = np.arange(len(arcs))
    r2 = [x + barWidth for x in r1]

    plt.bar(r1, 1-arcs_prob_damage, color='#7f6d5f', width=barWidth, edgecolor='white', label='Survival')
    plt.bar(r2, arcs_prob_damage, color='#557f2d', width=barWidth, edgecolor='white', label='Failure')

    plt.xlabel( 'Arc' )
    plt.ylabel( 'Probability' )
    plt.legend()
    plt.savefig(HOME.joinpath('figure2s.png'), dpi=200)
    plt.close()


