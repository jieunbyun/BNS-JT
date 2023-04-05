'''
Ji-Eun Byun (ji-eun.byun@glasgow.ac.uk)
Created: 27 Mar 2023

A small, hypothetical bridge system
'''
import numpy as np
import networkx as nx
from scipy.stats import lognorm
import matplotlib

matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

from inspect import getmembers, isfunction
from BNS_JT import cpm, variable
from Trans.trans import get_arcs_length

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

# plot graph
pos = nx.get_node_attributes(G, 'pos')
edge_labels = nx.get_edge_attributes(G, 'label')

fig = plt.figure()
ax = fig.add_subplot()
nx.draw(G, pos, with_labels=True, ax=ax)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
fig.savefig("graph.png", dpi=200)


# Arcs (components)
cpm_by_arc = {}
var_by_arc = {}
for k in arcs.keys():
    _type = arcs_type[k]
    prob = lognorm.cdf(GM_obs[k], frag[_type]['std'], scale=frag[_type]['med'])
    C = np.array([[arc_surv, arc_fail]]).T
    p = np.array([1-prob, prob])
    cpm_by_arc[k] = cpm.Cpm(variables = [k],
                            no_child = 1,
                            C = C,
                            p = p)

    B = np.array([[1, 0], [0, 1], [1, 1]])
    var_by_arc[k] = variable.Variable(B=B, value=['Surv', 'Fail'])

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

cpm_by_arc[7] = cpm.Cpm(variables= [7, 1, 2, 3, 4, 5, 6],
                       no_child = 1,
                       C = c7,
                       p = [1, 1, 1, 1],
                       )

cpm_by_arc[8] = cpm.Cpm(variables= [8, 1, 2, 3, 4, 5, 6],
                       no_child = 1,
                       C = c8,
                       p = [1, 1, 1, 1],
                       )

cpm_by_arc[9] = cpm.Cpm(variables= [9, 1, 2, 3, 4, 5, 6],
                       no_child = 1,
                       C = c9,
                       p = [1, 1, 1, 1],
                       )
    iprint(f'cpm_by: {cpm_by_arc_copied}')

cpm_by_arc[10] = cpm.Cpm(variables= [10, 1, 2, 3, 4, 5, 6],
                       no_child = 1,
                       C = c10,
                       p = [1, 1, 1, 1],
                       )

B_ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
var_by_arc[7] = variable.Variable(B=B_,
        value=[0.0901, 0.2401, np.inf])

var_by_arc[8] = variable.Variable(B=B_,
        value=[0.0901, 0.2401, np.inf])

var_by_arc[9] = variable.Variable(B=B_,
        value=[0.0943, 0.1761, np.inf])

var_by_arc[10] = variable.Variable(B=B_,
        value=[0.0707, 0.1997, np.inf])

## Inference - by variable elimination (would not work for large-scale systems)
# Probability of delay and disconnection
var_elim_order = sorted(arcs.keys())
cpm_by_arc_copied = [value for key, value in sorted(cpm_by_arc.items())]

for i in var_elim_order:
    print(i)
    isinscope_tf = cpm.isinscope([i], cpm_by_arc_copied)
    cpm_sel = [y for x, y in zip(isinscope_tf, cpm_by_arc_copied) if x]
    cpm_mult, var_by_arc = cpm.prod_cpms(cpm_sel, var_by_arc)
    cpm_mult = cpm_mult.sum([i])

    cpm_arc_copied = [y for x, y in zip(isinscope_tf, cpm_by_arc_copied) if x is False]
    cpm_by_arc_copied.insert(0, cpm_mult)

"""
G = graph(arcs(:,1), arcs(:,2), arcTimes_h)
[arcPaths, arcPaths_time] = funTrans.getAllPathsAndTimes( ODs, G, arcTimes_h )


# GM
GM_obs = [30 20 10 2 0.9 0.6]


## BN set up
varInd = 0
M = Cpm
vars = Variable


# Retrieve example results
ODs_prob_disconn = zeros(1,nOD)
ODs_prob_delay = zeros(1,nOD)
for iODInd = 1:nOD
    iVarInd = var_OD(iODInd)

    # Prob. of disconnection
    iDisconnState = size(vars(iVarInd).B,1)
    iM_VE = condition( M_VE, iVarInd, iDisconnState, vars )

    iDisconnProb = sum( iM_VE.p )
    ODs_prob_disconn(iODInd) = iDisconnProb

    # Prob. of delay
    iVarLocInC = find(M_VE.variables == iVarInd)
    iRowsIndToKeep = find( M_VE.C(:, iVarLocInC)> 1 )
    iM_VE = getCpmSubset( M_VE, iRowsIndToKeep )

    iDelayProb = sum( iM_VE.p )
    ODs_prob_delay(iODInd) = iDelayProb
end


figure
bar( [ODs_prob_disconn(:) ODs_prob_delay(:)] )

grid on
xlabel( 'OD pair' )
ylabel( 'Probability' )
legend( {'Disconnection' 'Delay'}, 'location', 'northwest' )


# City 1 and 2 experienced a disruption in getting resources, City 3 was okay and 4 is unknown. Probability of damage of roads?
# # A composite state needs be created for City 1 and City 2
vars( var_OD(1) ).B = [vars( var_OD(1) ).B 0 1 1]
vars( var_OD(2) ).B = [vars( var_OD(2) ).B 0 1 1]

[Mcond, vars] = condition(M,[var_OD(1) var_OD(2) var_OD(3)], [4 4 1], vars)
[Mcond_mult, vars] = multCPMs(Mcond, vars)
Mcond_mult_sum = sum( Mcond_mult, var_OD )

arcs_prob_damage = zeros(1,nArc)
for iArcInd = 1:nArc
    iVarInd = var_arcs(iArcInd)

    iM = sum( Mcond_mult_sum, iVarInd, 0 )
    [iM_fail, vars] = condition( iM, iVarInd, arc_fail, vars )
    
    iFailProb = iM_fail.p / sum(iM.p)
    if ~isempty(iFailProb)
        arcs_prob_damage(iArcInd) = iFailProb
    else
        arcs_prob_damage(iArcInd) = 0
    end
end


figure
bar( [1-arcs_prob_damage(:) arcs_prob_damage(:)] )

grid on
xlabel( 'Arc' )
ylabel( 'Probability' )
legend( {'Survive' 'Fail'}, 'location', 'northwest' )
"""
