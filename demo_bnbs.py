"""
Ji-Eun Byun (ji-eun.byun@glasgow.ac.uk)
Created: 11 Apr 2023

Generalise Branch-and-Bound (BnB) operation to build CPMs
"""
import numpy as np
import networkx as nx
import pdb
from scipy.stats import lognorm
import matplotlib

matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

#import demo_transport
from Trans import bnb_fns
from BNS_JT.branch import get_cmat, run_bnb
from BNS_JT.cpm import variable_elim, Cpm, get_prob
from BNS_JT.variable import Variable
from Trans.trans import get_arcs_length, get_all_paths_and_times


#ODs_prob_delay, ODs_prob_disconn, _, cpms_arc, vars_arc = demo_transport.main()

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
fig.savefig("graph_bnb.png", dpi=200)

# Arcs (components): P(X_i | GM = GM_ob ), i = 1 .. N (= nArc)
cpms_arc = {}
vars_arc = {}
for k in arcs.keys():
    _type = arcs_type[k]
    prob = lognorm.cdf(GM_obs[k], frag[_type]['std'], scale=frag[_type]['med'])
    C = np.array([[arc_surv, arc_fail]]).T
    p = np.array([1-prob, prob])
    cpms_arc[k] = Cpm(variables = [f'{k}'],
                          no_child = 1,
                          C = C,
                          p = p)

    B = np.array([[1, 0], [0, 1], [1, 1]])
    vars_arc[f'{k}'] = Variable(B=B, value=['Surv', 'Fail'])

## Problem
odInd = 1

info = {'path': [['2'], ['3', '1']],
        'time': np.array([0.0901, 0.2401]),
        'arcs': np.array(['1', '2', '3', '4', '5', '6'])
        }

max_state = 2
comp_max_states = (max_state*np.ones(len(info['arcs']))).tolist()

branches = run_bnb(sys_fn=bnb_fns.bnb_sys,
                   next_comp_fn=bnb_fns.bnb_next_comp,
                   next_state_fn=bnb_fns.bnb_next_state,
                   info=info,
                   comp_max_states=comp_max_states)

[C_od, varis] = get_cmat(branches, info['arcs'], vars_arc, False)

# Check if the results are correct
# FIXME: index issue
od_var_id = 7 - 1
var_elim_order = list(range(1, 7))

M_bnb = list(cpms_arc.values())[:7]
#M_bnb[od_var_id].C = C_od
#M_bnb[od_var_id].p = np.ones(shape=(C_od.shape[0],1))
M_bnb_VE, vars_arc = variable_elim(M_bnb, var_elim_order, vars_arc)

# FIXME: index issue
disconn_state = 3 # max basic state
disconn_prob = get_prob(M_bnb_VE, [7], np.array([disconn_state]), vars_arc)
delay_prob = get_prob(M_bnb_VE, [7], np.array([1]), vars_arc, 0 )

# Check if the results are the same
# FIXME: index issue
np.testing.assert_array_almost_equal(ODs_prob_delay[0], delay_prob)
np.testing.assert_array_almost_equal(ODs_prob_disconn[0], disconn_prob)


