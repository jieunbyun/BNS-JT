import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os

from BNS_JT import cpm, variable


## System
HOME = os.getcwd()

node_coords = {'n1': (0, 1),
                'n2': (0, 0),
                'n3': (0, -1),
                'n4': (2, 0),
                'n5': (2, 1),
                'n6': (1, 1),
                'n7': (2, -1),
                'n8': (1, -1),
                'n9': (1, -2),
                'n10': (3, 1),
                'n11': (3, -1)}

no_node_st = 2 # Number of a node's states
node_st_cp = [0, 2] # state index to actual capacity (e.g. state 1 stands for flow capacity 2, etc.)
varis = {}
for k, v in node_coords.items():
    varis[k] = variable.Variable( name=k, B = np.eye( no_node_st ), values = node_st_cp )

edges = {'e1': ['n1', 'n2'],
        'e2': ['n3', 'n2'],
        'e3': ['n2', 'n4'],
        'e4': ['n4', 'n5'],
        'e5': ['n5', 'n6'],
        'e6': ['n4', 'n7'],
        'e7': ['n7', 'n8'],
        'e8': ['n7', 'n9'],
        'e9': ['n6', 'n5'],
        'e10': ['n8', 'n7'],
        'e11': ['n9','n7'],
        'e12': ['n7','n4'],
        'e13': ['n5','n4'],
        'e14': ['n5','n10'],
        'e15': ['n7','n11'],
        'e16': ['n4', 'n5'],
        'e17': ['n4', 'n7']}

edges2comps = {}
c_idx = 0
for e, pair in edges.items():

    c_rev = [x1 for e1,x1 in edges2comps.items() if edges[e1] == pair or edges[e1]==[pair[1], pair[0]]]
    if len(c_rev) == 0:
        c_idx += 1
        edges2comps[e] = 'x' + str(c_idx)
    else:
        edges2comps[e] = c_rev[0]
no_comp = c_idx
    

es_idx = { }
idx = 0
for e in edges:
    idx += 1
    es_idx[e] = idx

no_comp_st = 3 # Number of a comp's states
comp_st_fval = [0, 1, 2] # state index to actual flow capacity (e.g. state 1 stands for flow capacity 0, etc.)
for e, x in edges2comps.items():
    if x not in varis:
        varis[x] = variable.Variable( name=k, B = np.eye( no_comp_st ), values = comp_st_fval )

# subsystems information
sub_bw_nodes = [['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9'],
                ['n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11']] # nodes in between subsystem i and (i+1)
sub_bw_edges = [['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8'],
                ['e16','e17','e9','e10','e11','e12','e13','e14','e15']]

no_sub = len(sub_bw_nodes) + 1

depots = [['n1', 'n3'], ['n6', 'n8', 'n9'], ['n10', 'n11']] # nodes that flows must stop by


# Plot the system
G = nx.DiGraph()
for k, x in edges.items():
    G.add_edge(x[0], x[1], label=k)

for k, v in node_coords.items():
    G.add_node(k, pos=v, label = k)

pos = nx.get_node_attributes(G, 'pos')
edge_labels = nx.get_edge_attributes(G, 'label')

fig = plt.figure()
ax = fig.add_subplot()
nx.draw(G, pos, with_labels=True, ax=ax)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
fig.savefig( os.path.join(HOME, 'graph_toy.png'), dpi=200)


##### System analysis
import pipes_sys

import importlib
importlib.reload(pipes_sys)

### Example 1: all in the highest state
comps_st = {n: len(varis[n].B[0]) for n in node_coords}
for c_idx in range(no_comp):
    c_name = 'x' + str(c_idx+1)
    comps_st[c_name] = len(varis[c_name].B[0])

res = pipes_sys.run_pipes_fun( comps_st, edges, node_coords, es_idx, edges2comps, depots, varis, sub_bw_nodes, sub_bw_edges )
print(res)


def sys_fun_pipes( comps_st, thres, edges, node_coords, es_idx, edges2comps, depots, varis, sub_bw_nodes, sub_bw_edges ):
    res = pipes_sys.run_pipes_fun( comps_st, edges, node_coords, es_idx, edges2comps, depots, varis, sub_bw_nodes, sub_bw_edges )

    if res.success == True:
        sys_val = -res.fun
    else:
        sys_val = 0
    
    if sys_val < thres:
        sys_st = 'fail'
    else:
        sys_st = 'surv'
    

    if sys_st == 'surv':

        min_comps_st = {}
        for e, pair in edges.items():
            
            e_idx = es_idx[e]

            if res.x[ e_idx-1 ] > 0:
                x_name = edges2comps[e]

                if x_name not in min_comps_st:
                    x_st = comps_st[x_name]

                    min_comps_st[x_name] = x_st

                if pair[0] not in min_comps_st:
                    n_st = comps_st[pair[0]]
                    min_comps_st[pair[0]] = n_st

                if pair[1] not in min_comps_st:
                    n_st = comps_st[pair[1]]
                    min_comps_st[pair[1]] = n_st

    else:
        min_comps_st = None

    return sys_val, sys_st, min_comps_st


thres = 2
sys_val, sys_st, min_comps_st = sys_fun_pipes( comps_st, thres, edges, node_coords, es_idx, edges2comps, depots, varis, sub_bw_nodes, sub_bw_edges )

print(sys_val, sys_st, min_comps_st)


###### Branch and Bound
import gen_bnb

sys_fun = lambda comps_st : sys_fun_pipes( comps_st, thres, edges, node_coords, es_idx, edges2comps, depots, varis, sub_bw_nodes, sub_bw_edges )

comps_name = [n for n in node_coords]
for i in range(no_comp):
    comps_name.append( 'x'+str(i+1) )
print(comps_name)

no_sf, rules, rules_st, brs, sys_res = gen_bnb.do_gen_bnb( sys_fun, varis, comps_name, max_br=100 )

# Result
print(no_sf)
print(len(rules))
print(len(brs))
print( min([len(r) for r in rules]) )
print( [r for r in rules if len(r)<3 ] )