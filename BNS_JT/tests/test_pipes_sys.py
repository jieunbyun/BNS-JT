#!/usr/bin/env python

import pytest
import time

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from BNS_JT import cpm, variable
from BNS_JT import pipes_sys
from BNS_JT import gen_bnb

HOME = Path(__file__).parent

@pytest.fixture()
def main_sys():

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

    depots = [['n1', 'n3'], ['n6', 'n8', 'n9'], ['n10', 'n11']] # nodes that flows must stop by

    no_node_st = 2 # Number of a node's states
    node_st_cp = [0, 2] # state index to actual capacity (e.g. state 1 stands for flow capacity 2, etc.)

    varis = {}
    for k, v in node_coords.items():
        varis[k] = variable.Variable(name=k, B=np.eye(no_node_st), values=node_st_cp)

    edges2comps = {}
    c_idx = 0
    for e, pair in edges.items():
        c_rev = [x1 for e1, x1 in edges2comps.items() if edges[e1] == pair or edges[e1]==[pair[1], pair[0]]]
        if len(c_rev) == 0:
            c_idx += 1
            edges2comps[e] = 'x' + str(c_idx)
        else:
            edges2comps[e] = c_rev[0]
    no_comp = c_idx
    es_idx = {e: idx for idx, e in enumerate(edges, 1)}

    no_comp_st = 3 # Number of a comp's states
    comp_st_fval = [0, 1, 2] # state index to actual flow capacity (e.g. state 1 stands for flow capacity 0, etc.)
    for e, x in edges2comps.items():
        if x not in varis:
            varis[x] = variable.Variable( name=k, B = np.eye( no_comp_st ), values = comp_st_fval )

    #no_sub = len(sub_bw_nodes) + 1

    # Plot the system
    G = nx.DiGraph()
    for k, x in edges.items():
        G.add_edge(x[0], x[1], label=k)

    for k, v in node_coords.items():
        G.add_node(k, pos=v, label = k)

    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = nx.get_edge_attributes(G, 'label')

    comps_st = {n: len(varis[n].B[0]) for n in node_coords}

    for c_idx in range(no_comp):
        c_name = 'x' + str(c_idx+1)
        comps_st[c_name] = len(varis[c_name].B[0])

    fig = plt.figure()
    ax = fig.add_subplot()
    nx.draw(G, pos, with_labels=True, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    fig.savefig(HOME.joinpath('graph_toy.png'), dpi=200)

    return comps_st, edges, node_coords, es_idx, edges2comps, depots, varis


@pytest.fixture()
def sub_sys():

    # subsystems information
    sub_bw_nodes = [['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9'],
                    ['n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11']] # nodes in between subsystem i and (i+1)

    sub_bw_edges = [['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8'],
                    ['e16','e17','e9','e10','e11','e12','e13','e14','e15']]

    return sub_bw_nodes, sub_bw_edges

# # System analysis

def test_run_pipes_fun(main_sys, sub_sys):

    comps_st, edges, node_coords, es_idx, edges2comps, depots, varis = main_sys

    sub_bw_nodes, sub_bw_edges = sub_sys

    ### Example 1: all in the highest state
    res = pipes_sys.run_pipes_fun(comps_st, edges, node_coords, es_idx, edges2comps, depots, varis, sub_bw_nodes, sub_bw_edges )
    print(res)

@pytest.mark.skip('FIXME')
def test_sys_fun_pipes(main_sys, sub_sys):

    comps_st, edges, node_coords, es_idx, edges2comps, depots, varis = main_sys

    sub_bw_nodes, sub_bw_edges = sub_sys

    thres = 2
    sys_val, sys_st, min_comps_st = pipes_sys.sys_fun_pipes(comps_st, thres, edges, node_coords, es_idx, edges2comps, depots, varis, sub_bw_nodes, sub_bw_edges )

    assert sys_val == 2.0
    assert sys_st == 'surv'
    #FIXME
    assert min_comps_st == {'x1': 2, 'n1': 1, 'n2': 1, 'x3': 2, 'n4': 1, 'x4': 2, 'n5': 1, 'x5': 2, 'n6': 1, 'x9': 2, 'n10': 1}


def sys_fun_wrap(thres, edges, node_coords, es_idx, edges2comps, depots, varis, sub_bw_nodes, sub_bw_edges):
    def sys_fun2(comps_st):
        return pipes_sys.sys_fun_pipes(comps_st, thres, edges, node_coords, es_idx, edges2comps, depots, varis, sub_bw_nodes, sub_bw_edges)
    return sys_fun2


@pytest.mark.skip('too long')
def test_do_gen_bnb(main_sys, sub_sys):

    # Branch and Bound
    comps_st, edges, node_coords, es_idx, edges2comps, depots, varis = main_sys

    sub_bw_nodes, sub_bw_edges = sub_sys

    thres = 2

    sys_fun = sys_fun_wrap(thres, edges, node_coords, es_idx, edges2comps, depots, varis, sub_bw_nodes, sub_bw_edges)

    comps_name = list(comps_st.keys())

    # originally max_br = 1000
    no_sf, rules, rules_st, brs, sys_res = gen_bnb.do_gen_bnb( sys_fun, varis, max_br=50)


