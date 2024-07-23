import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pytest

from pathlib import Path
from numpy.random import choice
from BNS_JT import cpm, variable, branch, pipes_sys

HOME = Path(__file__).parent


@pytest.fixture(scope='session')
def sub_bw():
    # subsystems information
    sub_bw_nodes = [[f'n{i+1}' for i in range(42)],
                    [f'n{i+1}' for i in range(2, 44)],
                    ['n43', 'n44', 'n45']] # nodes in subsystem i and (i+1)
    sub_bw_edges = [[f'e{i+1}' for i in range(41)],
                    ['e3','e12', 'e23', 'e36'] + [f'e{i+1}' for i in range(45, 80)] + ['e42', 'e43'],
                    ['e44', 'e45']]

    #no_sub = len(sub_bw_nodes) + 1

    depots = [['n1', 'n2'], ['n4', 'n7', 'n9', 'n11', 'n12', 'n15', 'n17', 'n19', 'n20', 'n22', 'n23', 'n26', 'n27', 'n29', 'n30', 'n32', 'n33', 'n35', 'n36', 'n38', 'n40', 'n42'], ['n43', 'n44'], ['n45']] # nodes that flows must stop by

    return depots, sub_bw_nodes, sub_bw_edges


@pytest.fixture(scope='session')
def setup_sys(sub_bw):

    depots, sub_bw_nodes, sub_bw_edges = sub_bw

    # ## Network topology
    d_h1 = 1.0
    d_h2 = 0.2
    d_v1 = 1.0
    d_v2 = 0.4

    node_coords = {'n1': (7*d_h1, 2*d_v2),
                    'n2': (7*d_h1, 0.0),
                    'n3': (4*d_h1, d_v2),
                    'n4': (4*d_h1, 0.0),
                    'n5': (3.5*d_h1, d_v2),
                    'n6': (3*d_h1, d_v2),
                    'n7': (3*d_h1, 0.0),
                    'n8': (2*d_h1, d_v2),
                    'n9': (2*d_h1, 0.0),
                    'n10': (1*d_h1, d_v2),
                    'n11': (1*d_h1, 0.0),
                    'n12': (0.0, 0.0),
                    'n13': (3.5*d_h1, d_v1+d_v2),
                    'n14': (3*d_h1, d_v1+d_v2),
                    'n15': (3*d_h1, d_v1),
                    'n16': (2*d_h1, d_v1+d_v2),
                    'n17': (2*d_h1, d_v1),
                    'n18': (1*d_h1, d_v1+d_v2),
                    'n19': (1*d_h1, d_v1),
                    'n20': (0, d_v1),
                    'n21': (4*d_h1, d_v1+d_v2),
                    'n22': (4*d_h1, d_v1),
                    'n23': (5*d_h1, d_v1),
                    'n24': (3.5*d_h1, 2*d_v1+d_v2),
                    'n25': (3*d_h1, 2*d_v1+d_v2),
                    'n26': (3*d_h1-d_h2, 2*d_v1+2*d_v2),
                    'n27': (3*d_h1, 2*d_v1),
                    'n28': (2*d_h1, 2*d_v1+d_v2),
                    'n29': (2*d_h1-d_h2, 2*d_v1+2*d_v2),
                    'n30': (2*d_h1, 2*d_v1),
                    'n31': (1*d_h1, 2*d_v1+d_v2),
                    'n32': (1*d_h1-d_h2, 2*d_v1+2*d_v2),
                    'n33': (1*d_h1, 2*d_v1),
                    'n34': (0*d_h1, 2*d_v1+d_v2),
                    'n35': (0*d_h1-d_h2, 2*d_v1+2*d_v2),
                    'n36': (0*d_h1, 2*d_v1),
                    'n37': (4*d_h1, 2*d_v1+d_v2),
                    'n38': (4*d_h1, 2*d_v1),
                    'n39': (5*d_h1, 2*d_v1+d_v2),
                    'n40': (5*d_h1, 2*d_v1),
                    'n41': (6*d_h1, 2*d_v1+d_v2),
                    'n42': (6*d_h1+d_h2, 2*d_v1),
                    'n43': (4*d_h1, 3*d_v1),
                    'n44': (6*d_h1, 3*d_v1),
                    'n45': (3.5*d_h1, 3.5*d_v1)
                    }

    no_node_st = 2 # Number of a node's states
    nc = [10, 5, 1]
    node_capas = {'n1': nc[1],
                    'n2': nc[1],
                    'n3': nc[0],
                    'n4': nc[2],
                    'n5': nc[0],
                    'n6': nc[0],
                    'n7': nc[2],
                    'n8': nc[0],
                    'n9': nc[2],
                    'n10': nc[0],
                    'n11': nc[2],
                    'n12': nc[2],
                    'n13': nc[0],
                    'n14': nc[0],
                    'n15': nc[2],
                    'n16': nc[0],
                    'n17': nc[2],
                    'n18': nc[0],
                    'n19': nc[2],
                    'n20': nc[2],
                    'n21': nc[0],
                    'n22': nc[2],
                    'n23': nc[2],
                    'n24': nc[0],
                    'n25': nc[0],
                    'n26': nc[2],
                    'n27': nc[2],
                    'n28': nc[0],
                    'n29': nc[2],
                    'n30': nc[2],
                    'n31': nc[0],
                    'n32': nc[2],
                    'n33': nc[2],
                    'n34': nc[0],
                    'n35': nc[2],
                    'n36': nc[2],
                    'n37': nc[0],
                    'n38': nc[2],
                    'n39': nc[0],
                    'n40': nc[2],
                    'n41': nc[0],
                    'n42': nc[2],
                    'n43': nc[0],
                    'n44': nc[0],
                    'n45': nc[0]}

    varis = {}
    for k, v in node_capas.items():
        varis[k] = variable.Variable(name=k, B = [{0}, {1}], values = [0, v])

    edges = {'e1': ['n1', 'n3'],
            'e2': ['n2', 'n3'],
            'e3': ['n3', 'n5'],
            'e4': ['n3', 'n4'],
            'e5': ['n5', 'n6'],
            'e6': ['n6', 'n7'],
            'e7': ['n6', 'n8'],
            'e8': ['n8', 'n9'],
            'e9': ['n8', 'n10'],
            'e10': ['n10', 'n11'],
            'e11': ['n10','n12'],
            'e12': ['n5','n13'],
            'e13': ['n13','n14'],
            'e14': ['n14','n15'],
            'e15': ['n14','n16'],
            'e16': ['n16', 'n17'],
            'e17': ['n16', 'n18'],
            'e18': ['n18', 'n19'],
            'e19': ['n18', 'n20'],
            'e20': ['n13', 'n21'],
            'e21': ['n21', 'n22'],
            'e22': ['n21', 'n23'],
            'e23': ['n13', 'n24'],
            'e24': ['n24', 'n25'],
            'e25': ['n25', 'n26'],
            'e26': ['n25', 'n27'],
            'e27': ['n25', 'n28'],
            'e28': ['n28', 'n29'],
            'e29': ['n28', 'n30'],
            'e30': ['n28', 'n31'],
            'e31': ['n31', 'n32'],
            'e32': ['n31', 'n33'],
            'e33': ['n31', 'n34'],
            'e34': ['n34', 'n35'],
            'e35': ['n34', 'n36'],
            'e36': ['n24', 'n37'],
            'e37': ['n37', 'n38'],
            'e38': ['n37', 'n39'],
            'e39': ['n39', 'n40'],
            'e40': ['n39', 'n41'],
            'e41': ['n41', 'n42'],
            'e42': ['n37', 'n43'],
            'e43': ['n41', 'n44'],
            'e44': ['n43', 'n45'],
            'e45': ['n44', 'n45']}

    # Inverse direction
    edges_1d = ['e1', 'e2', 'e3', 'e12', 'e23', 'e36', 'e42', 'e43', 'e44', 'e45'] # edges in only one direction
    ek_tmp = [k for k in edges]
    e_n = len(ek_tmp)
    for e in ek_tmp:
        if e not in edges_1d:
            e_n += 1
            e_k = 'e' + str(e_n)
            edges[e_k] = edges[e][::-1]

    edge_capas = {}
    for k,v in edges.items():
        edge_capas[k] = max([node_capas[v[0]], node_capas[v[1]]])

    edges2comps = {} # mapping between edges and component events
    comps2edges = {}
    c_idx = 0
    for e, pair in edges.items():
        c_idx += 1
        x_name = 'x' + str(c_idx)
        edges2comps[e] = x_name
        if x_name in comps2edges:
            comps2edges[x_name] = comps2edges[x_name].append(e)
        else:
            comps2edges['x' + str(c_idx)] = [e]

    no_comp = c_idx

    # Numerical index of edges (in optimisation problem)
    es_idx = {}
    idx = 0
    for e in edges:
        idx += 1
        es_idx[e] = idx

    no_comp_st = 2 # Number of a comp's states
    for e, x in edges2comps.items():
        if x not in varis:
            varis[x] = variable.Variable( name=x, B = [{0}, {1}], values = [0, edge_capas[e]] )

    thres = 5 # Threhold of system failure event (minimum flow to be processed)

    # Plot the system
    G = nx.DiGraph()
    for k, x in edges.items():
        G.add_edge(x[0], x[1], label=k)

    for k, v in node_coords.items():
        G.add_node(k, pos=v, label = k)

    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = nx.get_edge_attributes(G, 'label')

    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot()
    nx.draw(G, pos, with_labels=True, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    fig.savefig(HOME.joinpath('pipe_large.png'), dpi=200)

    cpms = {}

    # Component events
    n_pf = 0.001
    for k in node_coords:
        cpms[k] = cpm.Cpm(variables=[varis[k]], no_child = 1, C = np.array([0,1]), p = [n_pf, 1-n_pf])

    e_pf = 0.0005
    for e, x in edges2comps.items():
        cpms[x] = cpm.Cpm([varis[x]], no_child = 1, C = np.array([0,1]), p = [e_pf, 1-e_pf])

    return cpms, varis, node_coords, comps2edges, edges, es_idx, edges2comps


def mcs1_pipe_comps(cpms, n_names, x_names):
    # cpms: a dictionary
    # n_names, x_names: list
    samp = {}
    for x in n_names + x_names:
        x_samp = choice([k[0] for k in cpms[x].C], 1, p=[k[0] for k in cpms[x].p])
        samp[x] = x_samp[0]

    return samp

@pytest.mark.skip('TAKESTOOLONG')
def test_inference1(setup_sys, sub_bw):

    cpms, varis, node_coords, comps2edges, edges, es_idx, edges2comps = setup_sys

    depots, sub_bw_nodes, sub_bw_edges = sub_bw

    thres = 10

    n_names = list(node_coords)
    x_names = list(comps2edges)

    cov_t = 0.05
    cov = 1
    no_fail = 0
    no_samp = 0
    t_s = time.time()

    while cov > cov_t:
        no_samp += 1
        samp1 = mcs1_pipe_comps(cpms, n_names, x_names)
        val1, sys1, _ = pipes_sys.sys_fun_pipes(samp1, thres, edges, node_coords, es_idx, edges2comps, depots, varis, sub_bw_nodes, sub_bw_edges)

        if sys1 == 'f':
            no_fail += 1

        if no_fail > 0:
            pf = no_fail / no_samp
            var = (1-pf)*pf / no_samp

        if no_fail > 10:
            cov = np.sqrt(var) / pf

        if no_samp % 2000 == 0:
            print( '[' + str(no_samp) + ' samples]: pf-' + str(pf) + ', c.o.v-' + str(cov) )
    t_e = time.time()

    print( 'Sampling took ' + str( (t_e-t_s) / 60 ) + ' minutes, with results:' )
    print( '[' + str(no_samp) + ' samples]: pf-' + str(pf) + ', c.o.v-' + str(cov) )






