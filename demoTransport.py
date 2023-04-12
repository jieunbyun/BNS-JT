'''
Ji-Eun Byun (ji-eun.byun@glasgow.ac.uk)
Created: 27 Mar 2023

A small, hypothetical bridge system
'''
import numpy as np
import networkx as nx
from scipy.stats import lognorm
import matplotlib
import pdb


matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

from BNS_JT import cpm, variable
from Trans.trans import get_arcs_length, get_all_paths_and_times

## Data
# Network
node_coords = {1: (-2, 3),
               2: (-2, -3),
               3: (2, -2),
               4: (1, 1),
               5: (0, 0)}

arcs = {1: [1, 2],
	2: [1, 5],
	3: [2, 5],
	4: [3, 4],
	5: [3, 5],
	6: [4, 5]}

# Fragility curves -- From HAZUS-EQ model (roads are regarded as disconnected when being extensively or completely damaged)

frag = {'major': {'med': 60.0, 'std': 0.7},
        'urban' : {'med': 24.0, 'std': 0.7},
        'bridge': {'med': 1.1, 'std': 3.9},
        }

arcs_type = {1: 'major',
             2: 'major',
             3: 'major',
             4: 'urban',
             5: 'bridge',
             6: 'bridge'}

arcs_avg_kmh = {1: 40,
                2: 40,
                3: 40,
                4: 30,
                5: 30,
                6: 20}

ODs = [(5, 1), (5, 2), (5, 3), (5, 4)]

# For the moment, we assume that ground motions are observed. Later, hazard nodes will be added.
GM_obs = {1: 30.0,
          2: 20.0,
          3: 10.0,
          4: 2.0,
          5: 0.9,
          6: 0.6}

# Arcs' states index
arc_surv = 1
arc_fail = 2
arc_either = 3

arc_lens_km = get_arcs_length(arcs, node_coords)

#arcTimes_h = arcLens_km ./ arcs_Vavg_kmh
arc_times_h = {k: v/arcs_avg_kmh[k] for k, v in arc_lens_km.items()}

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
fig.savefig("graph.png", dpi=200)

# Arcs (components)
cpms_arc = {}
vars_arc = {}
for k in arcs.keys():
    _type = arcs_type[k]
    prob = lognorm.cdf(GM_obs[k], frag[_type]['std'], scale=frag[_type]['med'])
    C = np.array([[arc_surv, arc_fail]]).T
    p = np.array([1-prob, prob])
    cpms_arc[k] = cpm.Cpm(variables = [k],
                          no_child = 1,
                          C = C,
                          p = p)

    B = np.array([[1, 0], [0, 1], [1, 1]])
    vars_arc[k] = variable.Variable(B=B, value=['Surv', 'Fail'])

# Travel times (systems)
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

cpms_arc[7] = cpm.Cpm(variables= [7, 1, 2, 3, 4, 5, 6],
                       no_child = 1,
                       C = c7,
                       p = [1, 1, 1, 1],
                       )

cpms_arc[8] = cpm.Cpm(variables= [8, 1, 2, 3, 4, 5, 6],
                       no_child = 1,
                       C = c8,
                       p = [1, 1, 1, 1],
                       )

cpms_arc[9] = cpm.Cpm(variables= [9, 1, 2, 3, 4, 5, 6],
                       no_child = 1,
                       C = c9,
                       p = [1, 1, 1, 1],
                       )

cpms_arc[10] = cpm.Cpm(variables= [10, 1, 2, 3, 4, 5, 6],
                       no_child = 1,
                       C = c10,
                       p = [1, 1, 1, 1],
                       )

B_ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
vars_arc[7] = variable.Variable(B=B_,
        value=[0.0901, 0.2401, np.inf])

vars_arc[8] = variable.Variable(B=B_,
        value=[0.0901, 0.2401, np.inf])

vars_arc[9] = variable.Variable(B=B_,
        value=[0.0943, 0.1761, np.inf])

vars_arc[10] = variable.Variable(B=B_,
        value=[0.0707, 0.1997, np.inf])

## Inference - by variable elimination (would not work for large-scale systems)
# Probability of delay and disconnection
cpms_arc_cp = cpms_arc.values()

for i in arcs.keys():

    isinscope_tf = cpm.isinscope([i], cpms_arc_cp)
    cpm_sel = [y for x, y in zip(isinscope_tf, cpms_arc_cp) if x]
    cpm_mult, vars_arc = cpm.prod_cpms(cpm_sel, vars_arc)
    cpm_mult = cpm_mult.sum([i])

    cpms_arc_cp = [y for x, y in zip(isinscope_tf, cpms_arc_cp) if x == False]
    cpms_arc_cp.insert(0, cpm_mult)

# Retrieve example results
ODs_prob_disconn = np.zeros(len(ODs))
ODs_prob_delay = np.zeros(len(ODs))

var_OD = [7, 8, 9, 10]
for j, idx in enumerate(var_OD):

    # Prob. of disconnection
    disconn_state = vars_arc[idx].B.shape[0]
    [cpm_ve], vars_arc = cpm.condition(cpms_arc_cp,
                           cnd_vars=[idx],
                           cnd_states=[disconn_state],
                           var=vars_arc)
    ODs_prob_disconn[j] = cpm_ve.p.sum(axis=0)

    # Prob. of delay
    var_loc = np.where(cpms_arc_cp[0].variables == idx)[0]
    rows_to_keep = np.where(cpms_arc_cp[0].C[:, var_loc] > 1)[0]
    cpm_ve = cpms_arc_cp[0].get_subset(rows_to_keep)
    ODs_prob_delay[j] = cpm_ve.p.sum(axis=0)

# City 1 and 2 experienced a disruption in getting resources, City 3 was okay and 4 is unknown. Probability of damage of roads?
# A composite state needs be created for City 1 and City 2
for idx in var_OD:
    vars_arc[idx].B = np.vstack([vars_arc[idx].B, [0, 1, 1]])

# # Add observation nodes
var_OD_obs = []
for j, (idx, od) in enumerate(zip(var_OD, ODs), start=11):

    #iPaths = arcPaths[iOdInd]
    #iTimes = arcPaths_time[iOdInd]
    cpms_arc[j]= cpm.Cpm(variables=[j, idx],
                           no_child=1,
                           C= np.array([[1, 1], [2, 2], [2, 3]]),
                           p= [1, 1, 1])
    vars_arc[j] = variable.Variable(B=np.eye(2, dtype=int), value=['No disruption', 'Disruption'])

    var_OD_obs.append(j)

cpm_ve, vars_arc = cpm.condition(cpms_arc,
                                 cnd_vars=[11, 12, 13],
                                 cnd_states=[2, 2, 1],
                                 var=vars_arc)

#pdb.set_trace()

Mcond_mult, vars_arc = cpm.prod_cpms(cpm_ve, vars_arc)
Mcond_mult_sum = Mcond_mult.sum(var_OD + var_OD_obs)

arcs_prob_damage = np.zeros(len(arcs))
for j, i in enumerate(arcs.keys()):

    iM = Mcond_mult_sum.sum([i], 0)
    [iM_fail], vars_arc = cpm.condition(iM, [i], [arc_fail], vars_arc)
    fail_prob = iM_fail.p / iM.p.sum(axis=0)
    if fail_prob.any():
        arcs_prob_damage[j] = fail_prob

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
plt.savefig('./figure1.png', dpi=200)
plt.close()

plt.figure()

r1 = list(arcs.keys())
r2 = [x + barWidth for x in r1]

plt.bar(r1, 1-arcs_prob_damage, color='#7f6d5f', width=barWidth, edgecolor='white', label='Survival')
plt.bar(r2, arcs_prob_damage, color='#557f2d', width=barWidth, edgecolor='white', label='Failure')

plt.xlabel( 'Arc' )
plt.ylabel( 'Probability' )
plt.legend()
plt.savefig('./figure2.png', dpi=200)
plt.close()
