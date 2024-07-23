#!/usr/bin/env python

import pytest
import pdb
import copy
import time
import pickle

import networkx as nx
import graphviz as gv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from BNS_JT import cpm, config, variable, branch, pipes_sys

HOME = Path(__file__).parent


@pytest.fixture(scope='session')
def nodes_edges():

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

    return node_coords, edges, depots


@pytest.fixture(scope='session')
def main_sys(nodes_edges):
    """
    based on pipe_system_v2.ipynb
    edges2comps is different from main_sys2
    {'e1': 'x1', 'e2': 'x2', 'e3': 'x3', 'e4': 'x4', 'e5': 'x5', 'e6': 'x6', 'e7': 'x7', 'e8': 'x8', 'e9': 'x5', 'e10': 'x7', 'e11': 'x8', 'e12': 'x6', 'e13': 'x4', 'e14': 'x9', 'e15': 'x10', 'e16': 'x4', 'e17': 'x6'}
    from x1 to x10
    varis: from n1 to n11 and x1 to x10
    """
    node_coords, edges, depots = nodes_edges
    #pdb.set_trace()
    no_node_st = 2 # Number of a node's states
    node_st_cp = [0, 2] # state index to actual capacity (e.g. state 1 stands for flow capacity 2, etc.)

    varis = {}
    for k, v in node_coords.items():
        varis[k] = variable.Variable(name=k, values=node_st_cp)

    edges2comps = {}
    c_idx = 0
    for e, pair in edges.items():
        c_rev = [x1 for e1, x1 in edges2comps.items() if edges[e1] == pair or edges[e1]==[pair[1], pair[0]]]

        if len(c_rev) == 0:
            c_idx += 1
            edges2comps[e] = f'x{c_idx}'
        else:
            edges2comps[e] = c_rev[0]
    no_comp = c_idx

    es_idx = {e: idx for idx, e in enumerate(edges)}

    no_comp_st = 3 # Number of a comp's states
    comp_st_fval = [0, 1, 2] # state index to actual flow capacity (e.g. state 1 stands for flow capacity 0, etc.)
    for e, x in edges2comps.items():
        if x not in varis:
            varis[x] = variable.Variable(name=k, values = comp_st_fval)

    #no_sub = len(sub_bw_nodes) + 1
    #comps_st = {n: len(varis[n].B[0]) - 1 for _, n in edges2comps}
    comps_st = {n: len(varis[n].values) - 1 for n in node_coords}
    comps_st.update({v: len(varis[v].values) - 1 for _, v in edges2comps.items()})

    # Plot the system
    #G = nx.DiGraph()
    G = nx.MultiDiGraph()
    for k, x in edges.items():
        G.add_edge(x[0], x[1], label=k)

    for k, v in node_coords.items():
        G.add_node(k, pos=v, label = k)

    h = config.networkx_to_graphviz(G)
    outfile = HOME.joinpath('graph_pipes')
    h.render(outfile, format='png', cleanup=True)
    """
    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = nx.get_edge_attributes(G, 'label')

    fig = plt.figure()
    ax = fig.add_subplot()
    nx.draw(G, pos, with_labels=True, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    fig.savefig(HOME.joinpath('graph_pipes.png'), dpi=200)
    """
    return comps_st, edges, node_coords, es_idx, edges2comps, depots, varis


@pytest.fixture(scope='session')
def main_sys2(nodes_edges):
    """
    based on gen_bnb_to_rel_deter_ind_pipe.ipynb
    return variables consisting of n1-n11, x1-x17
    e1-e17 for edges
    varis from n1 to n11, x1 to x17
    edges2comps:
    {'e1': 'x1', 'e2': 'x2', 'e3': 'x3', 'e4': 'x4', 'e5': 'x5', 'e6': 'x6', 'e7': 'x7', 'e8': 'x8', 'e9': 'x9', 'e10': 'x10', 'e11': 'x11', 'e12': 'x12', 'e13': 'x13', 'e14': 'x14', 'e15': 'x15', 'e16': 'x16', 'e17': 'x17'}
    """

    node_coords, edges, depots = nodes_edges

    #pdb.set_trace()
    no_node_st = 2 # Number of a node's states
    node_st_cp = [0, 2] # state index to actual capacity (e.g. state 1 stands for flow capacity 2, etc.)

    varis = {}
    for k, v in node_coords.items():
        varis[k] = variable.Variable(name=k, values=node_st_cp)

    # different from the main_sys
    edges2comps = {e: f'x{i}' for i, e in enumerate(edges.keys(), 1)}
    no_comp = len(edges)

    es_idx = {e: idx for idx, e in enumerate(edges)}

    no_comp_st = 3 # Number of a comp's states
    comp_st_fval = [0, 1, 2] # state index to actual flow capacity (e.g. state 1 stands for flow capacity 0, etc.)
    for e, x in edges2comps.items():
        if x not in varis:
            varis[x] = variable.Variable(name=x, values = comp_st_fval)

    #no_sub = len(sub_bw_nodes) + 1
    #comps_st = {n: len(varis[n].B[0]) - 1 for _, n in edges2comps}
    comps_st = {n: len(varis[n].values) - 1 for n in node_coords}
    comps_st.update({v: len(varis[v].values) - 1 for _, v in edges2comps.items()})

    # Plot the system: same as main_sys
    return comps_st, edges, node_coords, es_idx, edges2comps, depots, varis


@pytest.fixture(scope='session')
def sub_sys():

    # subsystems information
    sub_bw_nodes = [['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9'],
                    ['n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11']] # nodes in between subsystem i and (i+1)

    sub_bw_edges = [['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8'],
                    ['e16','e17','e9','e10','e11','e12','e13','e14','e15']]

    return sub_bw_nodes, sub_bw_edges


def test_do_node(main_sys):
# # System analysis

    comps_st, edges, node_coords, es_idx, edges2comps, depots, varis = main_sys

    no_x = len(edges)

    orig_end = depots[0]  # (n1, n3)
    dest_end = depots[-1]  # (n10, n11)

    orig_end_inds = {n: i for i, n in enumerate(orig_end)}
    dest_end_inds = {n: i for i, n in enumerate(dest_end)}

    no_u = len(orig_end)

    no_d_vars = no_x + no_u

    A = np.empty(shape=(0, no_d_vars))
    b_up = np.empty(shape=(0,))
    b_down = np.empty(shape=(0,))

    result = pipes_sys.do_node(orig_end, orig_end_inds, es_idx, edges, A, b_up, b_down)

    expected = {}
    expected['A'] = np.array([[1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,0.],
                              [0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.]])
    expected['b_up'] = np.array([0., 0.])
    expected['b_down'] = np.array([0., 0.])

    np.testing.assert_array_almost_equal(result[0], expected['A'])
    np.testing.assert_array_almost_equal(result[1], expected['b_up'])
    np.testing.assert_array_almost_equal(result[2], expected['b_down'])


def test_do_sub(main_sys, sub_sys):

    comps_st, edges, node_coords, es_idx, edges2comps, depots, varis = main_sys

    sub_bw_nodes, sub_bw_edges = sub_sys

    no_x = len(edges)

    orig_end = depots[0]
    dest_end = depots[-1]

    orig_end_inds = {n: i for i, n in enumerate(orig_end)}
    dest_end_inds = {n: i for i, n in enumerate(dest_end)}

    no_u = len(orig_end)

    no_d_vars = no_x + no_u

    prev = {}
    prev['A'] = np.array([[1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,0.],
                              [0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.]])
    prev['b_up'] = np.array([0., 0.])
    prev['b_down'] = np.array([0., 0.])

    result = pipes_sys.do_sub(sub_bw_nodes, sub_bw_edges, edges, es_idx, depots, no_d_vars, prev['A'], prev['b_up'], prev['b_down'])

    expected = {}
    expected['A'] = np.array([[ 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0.],
                              [ 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
                              [-1.,-1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [ 0., 0.,-1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [ 0., 0., 0.,-1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [ 0., 0., 0., 0., 0.,-1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [ 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,-1., -1.],
                              [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,-1.,-1., 0., 0., 1., 1., 0., 0.],
                              [ 0., 0., 0., 0., 0., 0., 0., 0.,-1., 0., 0., 0., 1., 1., 0.,-1., 0., 0., 0.],
                              [ 0., 0., 0., 0., 0., 0., 0., 0., 0.,-1.,-1., 1., 0., 0., 1., 0.,-1., 0., 0.],
                              [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., -1., -1.]])
    expected['b_up'] =  np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    expected['b_down'] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    try:
        np.testing.assert_array_almost_equal(result[0], expected['A'])
    except AssertionError:
        unequal = np.where(result[0] != expected['A'])
        print(unequal)
        print(result[0][unequal])
        print(expected['A'][unequal])
    np.testing.assert_array_almost_equal(result[1], expected['b_up'])
    np.testing.assert_array_almost_equal(result[2], expected['b_down'])


def test_do_incoming(main_sys, sub_sys):

    comps_st, edges, node_coords, es_idx, edges2comps, depots, varis = main_sys

    sub_bw_nodes, sub_bw_edges = sub_sys

    no_x = len(edges)
    no_sub = len(sub_bw_nodes)

    orig_end = depots[0]
    dest_end = depots[-1]

    orig_end_inds = {n: i for i, n in enumerate(orig_end)}
    dest_end_inds = {n: i for i, n in enumerate(dest_end)}

    no_u = len(orig_end)

    no_d_vars = no_x + no_u

    prev = {}
    prev['A'] = np.array([[ 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0.],
                              [ 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
                              [-1.,-1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [ 0., 0.,-1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [ 0., 0., 0.,-1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [ 0., 0., 0., 0., 0.,-1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [ 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,-1., -1.],
                              [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,-1.,-1., 0., 0., 1., 1., 0., 0.],
                              [ 0., 0., 0., 0., 0., 0., 0., 0.,-1., 0., 0., 0., 1., 1., 0.,-1., 0., 0., 0.],
                              [ 0., 0., 0., 0., 0., 0., 0., 0., 0.,-1.,-1., 1., 0., 0., 1., 0.,-1., 0., 0.],
                              [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., -1., -1.]])
    prev['b_up'] =  np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    prev['b_down'] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    result = pipes_sys.do_incoming(no_sub, depots, no_d_vars, edges, es_idx, prev['A'], prev['b_up'], prev['b_down'])

    expected = {}
    expected['A'] = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0.],
                              [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
                              [-1., -1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., -1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., -1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0.,-1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., -1.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,-1.,-1., 0., 0., 1., 1., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.,-1., 0., 0., 0., 1., 1., 0.,-1., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0.,-1.,-1., 1., 0., 0., 1., 0.,-1., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., -1., -1.],
                              [0., 0., 0., 0.,-1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.,-1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0.,-1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]])

    expected['b_up'] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    expected['b_down'] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    np.testing.assert_array_almost_equal(result[0], expected['A'])
    np.testing.assert_array_almost_equal(result[1], expected['b_up'])
    np.testing.assert_array_almost_equal(result[2], expected['b_down'])


def test_do_capacity(main_sys, sub_sys):

    comps_st, edges, node_coords, es_idx, edges2comps, depots, varis = main_sys

    sub_bw_nodes, sub_bw_edges = sub_sys

    no_x = len(edges)
    no_sub = len(sub_bw_nodes)

    orig_end = depots[0]
    dest_end = depots[-1]

    orig_end_inds = {n: i for i, n in enumerate(orig_end)}
    dest_end_inds = {n: i for i, n in enumerate(dest_end)}

    no_u = len(orig_end)

    no_d_vars = no_x + no_u

    prev = {}
    prev['A'] = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0.],
                              [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
                              [-1., -1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., -1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., -1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0.,-1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., -1.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,-1.,-1., 0., 0., 1., 1., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.,-1., 0., 0., 0., 1., 1., 0.,-1., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0.,-1.,-1., 1., 0., 0., 1., 0.,-1., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., -1., -1.],
                              [0., 0., 0., 0.,-1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.,-1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0.,-1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]])

    prev['b_up'] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    prev['b_down'] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    result = pipes_sys.do_capacity(edges, edges2comps, varis, comps_st, es_idx, no_d_vars, prev['A'], prev['b_up'], prev['b_down'])

    expected = {}
    expected['A'] = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0.],
                              [ 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
                              [-1.,-1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [ 0., 0.,-1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [ 0., 0., 0.,-1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [ 0., 0., 0., 0., 0.,-1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [ 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,-1., -1.],
                              [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,-1.,-1., 0., 0., 1., 1., 0., 0.],
                              [ 0., 0., 0., 0., 0., 0., 0., 0.,-1., 0., 0., 0., 1., 1., 0., -1., 0., 0., 0.],
                              [ 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., -1., 1., 0., 0., 1., 0., -1., 0., 0.],
                              [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., -1., -1.],
                              [ 0., 0., 0., 0., -1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [ 0., 0., 0., 0., 0., 0., -1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [ 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [ 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [ 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [ 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [ 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [ 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [ 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [ 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [ 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [ 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                              [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                              [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                              [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                              [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                              [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                              [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]])
    expected['b_up'] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.])
    expected['b_down'] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    np.testing.assert_array_almost_equal(result[0], expected['A'])
    np.testing.assert_array_almost_equal(result[1], expected['b_up'])
    np.testing.assert_array_almost_equal(result[2], expected['b_down'])


def test_run_pipes_fun(main_sys, sub_sys):

    comps_st, edges, node_coords, es_idx, edges2comps, depots, varis = main_sys

    sub_bw_nodes, sub_bw_edges = sub_sys

    ### Example 1: all in the highest state
    res = pipes_sys.run_pipes_fun(comps_st, edges, node_coords, es_idx, edges2comps, depots, varis, sub_bw_nodes, sub_bw_edges )

    assert res.success == True
    assert res.status == 0
    assert res.fun == -2.0
    np.testing.assert_array_almost_equal(res.x, np.array([2.,  0.,  2.,  2.,  2.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  2.,  0., ]))


def test_sys_fun_pipes0(main_sys, sub_sys):

    comps_st, edges, node_coords, es_idx, edges2comps, depots, varis = main_sys
    #pdb.set_trace()
    sub_bw_nodes, sub_bw_edges = sub_sys

    thres = 2
    sys_val, sys_st, min_comps_st = pipes_sys.sys_fun_pipes(comps_st, thres, edges, node_coords, es_idx, edges2comps, depots, varis, sub_bw_nodes, sub_bw_edges )

    assert sys_val == 2.0
    assert sys_st == 's'
    assert min_comps_st == {'x1': 2, 'n1': 1, 'n2': 1, 'x3': 2, 'n4': 1, 'x4': 2, 'n5': 1, 'x5': 2, 'n6': 1, 'x9': 2, 'n10': 1}


def test_sys_fun_pipes1(main_sys, sub_sys):

    _, edges, node_coords, es_idx, edges2comps, depots, varis = main_sys
    #pdb.set_trace()
    sub_bw_nodes, sub_bw_edges = sub_sys

    thres = 2

    comps_st ={'n1': 1, 'n2': 1, 'n3': 1, 'n4': 1, 'n5': 1, 'n6': 0, 'n7': 1, 'n8': 1, 'n9': 1, 'n10': 1, 'n11': 1, 'x1': 2, 'x2': 2, 'x3': 2, 'x4': 2, 'x5': 2, 'x6': 2, 'x7': 2, 'x8': 2, 'x9': 2, 'x10': 2}

    sys_val, sys_st, min_comps_st = pipes_sys.sys_fun_pipes(comps_st, thres, edges, node_coords, es_idx, edges2comps, depots, varis, sub_bw_nodes, sub_bw_edges )

    expected ={'x1': 2, 'n1': 1, 'n2': 1, 'x3': 2, 'n4': 1, 'x6': 2, 'n7': 1, 'x7': 2, 'n8': 1, 'x9': 2, 'n5': 1, 'n10': 1, 'x4': 2}

    assert sys_val == 2.0
    assert sys_st == 's'
    assert min_comps_st == expected


def sys_fun_wrap(thres, edges, node_coords, es_idx, edges2comps, depots, varis, sub_bw_nodes, sub_bw_edges):
    def sys_fun2(comps_st):
        return pipes_sys.sys_fun_pipes(comps_st, thres, edges, node_coords, es_idx, edges2comps, depots, varis, sub_bw_nodes, sub_bw_edges)
    return sys_fun2


def convert_branch_old(old, names):

    assert isinstance(old, branch.Branch_old), f'{old} should be an instance of branch.Branch_old'
    down = {k: v for k, v in zip(names, old.down)}
    up = {k: v for k, v in zip(names, old.up)}
    down_state = old.down_state
    up_state = old.up_state

    return branch.Branch(down, up, down_state, up_state)


@pytest.mark.skip('TOOLONG')
@pytest.fixture(scope='session')
def setup_inference(main_sys2, sub_sys):

    comps_st, edges, node_coords, es_idx, edges2comps, depots, d_varis = main_sys2
    varis = copy.deepcopy(d_varis)

    sub_bw_nodes, sub_bw_edges = sub_sys

    # Component events
    cpms = {}
    for k in node_coords:
        cpms[k] = cpm.Cpm(variables=[varis[k]], no_child = 1, C = np.array([0, 1]), p = [0.1, 0.9])

    for e, x in edges2comps.items():
        cpms[x] = cpm.Cpm([varis[x]], no_child = 1, C = np.array([0, 1, 2]), p = [0.1, 0.2, 0.7])

    # intact state
    thres = 2

    sys_fun = sys_fun_wrap(thres, edges, node_coords, es_idx, edges2comps, depots, varis, sub_bw_nodes, sub_bw_edges)

    file_brs = Path(__file__).parent.joinpath('brs_pipes2.pk')
    if file_brs.exists():
        with open(file_brs, 'rb') as fi:
            brs = pickle.load(fi)
            print(f'{file_brs} loaded')
    else:
        output_path = Path(__file__).parent
        brs, rules, sys_res = brc.run(
            sys_fun, varis, max_br=100_000, output_path=output_path, key='pipes2', flag=True)

    #pdb.set_trace()
    st_br_to_cs = {'f': 0, 's': 1, 'u': 2}
    csys, varis = brc.get_csys(brs, varis, st_br_to_cs)

    """
    # Damage observation
    C_o = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    p_o = np.array([0.95, 0.05, 0.1, 0.9]).T
    for i, k in enumerate(node_coords, 1):
        name = f'on{i}'
	# observation that n_i = 0 or 1 ** TO DISCUSS: probably values in dictionary..?
        varis[name] = variable.Variable(name=name, B = [{0}, {1}, {0, 1}], values = [0, 1])
        cpms[name] = cpm.Cpm(variables=[varis[name], varis[k]], no_child = 1, C = C_o, p = p_o)

    C_o = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2], [2, 2]])
    p_o = np.array([0.95, 0.04, 0.01, 0.3, 0.5, 0.2, 0.01, 0.19, 0.8]).T
    for i, (_, k) in enumerate(edges2comps.items(), 1):
        name = f'ox{i}'
        varis[name] = variable.Variable(name=name, B = [{0}, {1}, {2}, {0, 1, 2}], values = [0, 1, 2]) # observation that x_i = 0, 1, or 2 ** TO DISCUSS: probably values in dictionary..?
        cpms[name] = cpm.Cpm(variables=[varis[name], varis[k]], no_child = 1, C = C_o, p = p_o)

    # add observations
    added_on = np.ones(shape=(csys.shape[0], len(node_coords)), dtype=np.int8) * 2
    added_ox = np.ones(shape=(csys.shape[0], len(edges2comps)), dtype=np.int8) * 3
    csys = np.append(csys, added_on, axis=1)
    csys = np.append(csys, added_ox, axis=1)
    """

    # add sys
    varis['sys'] = variable.Variable('sys', [{0}, {1}, {2}], ['f', 's', 'u'])
    #cpm_sys_vname = [f'n{i}' for i in range(1, len(node_coords) + 1)] + [f'x{i}' for i in range(1, len(edges2comps) + 1)] + [f'on{i}' for i in range(1, len(node_coords) + 1)] + [f'ox{i}' for i in range(1, len(edges2comps) + 1)] #observations first, components later    cpm_sys_vname = list(brs[0].up.keys())
    cpm_sys_vname = [f'n{i}' for i in range(1, len(node_coords) + 1)] + [f'x{i}' for i in range(1, len(edges2comps) + 1)]
    cpm_sys_vname.insert(0, 'sys')
    cpms['sys'] = cpm.Cpm(
        variables=[varis[k] for k in cpm_sys_vname],
        no_child = 1,
        C = csys,
        p = np.ones((len(csys), 1), int))

    #var_elim_order_name = [f'on{i}' for i in range(1, len(node_coords) + 1)] + [f'ox{i}' for i in range(1, len(edges) + 1)]
    #var_elim_order_name += list(node_coords.keys()) + list(edges2comps.values())
    var_elim_order_name = list(node_coords.keys()) + list(edges2comps.values())
    var_elim_order = [varis[k] for k in var_elim_order_name]

    return cpms, varis, var_elim_order



@pytest.fixture()
def setup_brs(main_sys, sub_sys):

    # Branch and Bound
    comps_st, edges, node_coords, es_idx, edges2comps, depots, varis = main_sys
    #pdb.set_trace()
    sub_bw_nodes, sub_bw_edges = sub_sys

    thres = 2

    sys_fun = sys_fun_wrap(thres, edges, node_coords, es_idx, edges2comps, depots, varis, sub_bw_nodes, sub_bw_edges)

    # comps_name = list(comps_st.keys())

    # originally max_br = 1000
    #_list = [k for k in node_coords] + [v for _, v in edges2comps.items()]
    #varis_comp = {k: varis[k] for k in _list}

    output_path = Path(__file__).parent
    no_sf, rules, rules_st, brs, sys_res = brc.run(sys_fun, varis, max_br=1000, output_path=output_path, key='pipes2', flag=True)

    return no_sf, rules, rules_st, brs, sys_res


@pytest.mark.skip('TAKESTOOLONG')
def test_inference_case1_pipe(setup_inference):

    cpms, varis, var_elim_order = setup_inference
    #pdb.set_trace()
    _cpms = [cpms[v] for v in varis.keys()]
    Msys = cpm.variable_elim(_cpms, var_elim_order)
    print(Msys.C)
    print(Msys.p)
    np.testing.assert_array_almost_equal(Msys.C, np.array([[0, 2]]).T)
    np.testing.assert_array_almost_equal(Msys.p, np.array([[0.1018, 0.8982]]).T)


