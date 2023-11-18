#!/usr/bin/env python

import pytest
import pdb
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
        c_rev = [x1 for e1, x1 in edges2comps.items() if edges[e1] == pair or edges[e1]==pair[::-1]]

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
            varis[x] = variable.Variable(name=k, B = np.eye(no_comp_st), values = comp_st_fval)

    #no_sub = len(sub_bw_nodes) + 1
    #comps_st = {n: len(varis[n].B[0]) - 1 for _, n in edges2comps}
    comps_st = {n: len(varis[n].B[0]) - 1 for n in node_coords}
    comps_st.update({v: len(varis[v].B[0]) - 1 for _, v in edges2comps.items()})

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
    fig.savefig(HOME.joinpath('graph_pipes.png'), dpi=200)

    return comps_st, edges, node_coords, es_idx, edges2comps, depots, varis


@pytest.fixture()
def main_sys1():
    """
    Not sure but edges2comp is different from the original.
    """

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

    edges2comps = {e: f'x{i}' for i, e in enumerate(edges, 1)}

    no_comp = len(edges2comps)

    es_idx = {e: idx for idx, e in enumerate(edges)}

    no_comp_st = 3 # Number of a comp's states
    comp_st_fval = [0, 1, 2] # state index to actual flow capacity (e.g. state 1 stands for flow capacity 0, etc.)
    for e, x in edges2comps.items():
        if x not in varis:
            varis[x] = variable.Variable(name=k, B = np.eye(no_comp_st), values = comp_st_fval )

    #no_sub = len(sub_bw_nodes) + 1

    # Plot the system
    G = nx.DiGraph()
    for k, x in edges.items():
        G.add_edge(x[0], x[1], label=k)

    for k, v in node_coords.items():
        G.add_node(k, pos=v, label = k)

    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = nx.get_edge_attributes(G, 'label')

    comps_st = {n: len(varis[n].B[0]) - 1 for _, n in edges2comps.items()}

    fig = plt.figure()
    ax = fig.add_subplot()
    nx.draw(G, pos, with_labels=True, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    fig.savefig(HOME.joinpath('graph_pipes.png'), dpi=200)

    return comps_st, edges, node_coords, es_idx, edges2comps, depots, varis


@pytest.fixture()
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
    #pdb.set_trace()
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
    assert sys_st == 'surv'
    assert min_comps_st == {'x1': 2, 'n1': 1, 'n2': 1, 'x3': 2, 'n4': 1, 'x4': 2, 'n5': 1, 'x5': 2, 'n6': 1, 'x9': 2, 'n10': 1}


def test_sys_fun_pipes1(main_sys, sub_sys):

    _, edges, node_coords, es_idx, edges2comps, depots, varis = main_sys
    #pdb.set_trace()
    sub_bw_nodes, sub_bw_edges = sub_sys

    thres = 2

    comps_st ={'n1': 0, 'n2': 0, 'n3': 0, 'n4': 0, 'n5': 0, 'n6': 0, 'n7': 0, 'n8': 0, 'n9': 0, 'n10': 0, 'n11': 0, 'x1': 0, 'x2': 0, 'x3': 0, 'x4': 0, 'x5': 0, 'x6': 0, 'x7': 0, 'x8': 0, 'x9': 0, 'x10': 0}

    sys_val, sys_st, min_comps_st = pipes_sys.sys_fun_pipes(comps_st, thres, edges, node_coords, es_idx, edges2comps, depots, varis, sub_bw_nodes, sub_bw_edges )

    #FIXME
    #assert sys_val == -0.1
    assert sys_st == 'fail'
    assert min_comps_st == None


def sys_fun_wrap(thres, edges, node_coords, es_idx, edges2comps, depots, varis, sub_bw_nodes, sub_bw_edges):
    def sys_fun2(comps_st):
        return pipes_sys.sys_fun_pipes(comps_st, thres, edges, node_coords, es_idx, edges2comps, depots, varis, sub_bw_nodes, sub_bw_edges)
    return sys_fun2

@pytest.mark.skip('FIXME')
def test_do_gen_bnb_pipe(main_sys, sub_sys):

    # Branch and Bound
    comps_st, edges, node_coords, es_idx, edges2comps, depots, varis = main_sys
    sub_bw_nodes, sub_bw_edges = sub_sys

    thres = 2

    sys_fun = sys_fun_wrap(thres, edges, node_coords, es_idx, edges2comps, depots, varis, sub_bw_nodes, sub_bw_edges)

    # originally max_br = 1000
    #_list = [k for k in node_coords] + [v for _, v in edges2comps.items()]
    #varis_comp = {k: varis[k] for k in _list}

    output_path = Path(__file__).parent
    no_sf, rules, rules_st, brs, sys_res = gen_bnb.do_gen_bnb(sys_fun, varis, max_br=1000, output_path=output_path, key='pipe', flag=False)

    assert no_sf == 314
    assert len(rules) == 100
    assert len(brs) == 115


@pytest.fixture()
def setup_brs(main_sys1, sub_sys, setup_comp_events):

    # Branch and Bound
    comps_st, edges, node_coords, es_idx, edges2comps, depots, _ = main_sys1
    #pdb.set_trace()
    sub_bw_nodes, sub_bw_edges = sub_sys

    _, varis = setup_comp_events

    thres = 2

    sys_fun = sys_fun_wrap(thres, edges, node_coords, es_idx, edges2comps, depots, varis, sub_bw_nodes, sub_bw_edges)

    # comps_name = list(comps_st.keys())

    # originally max_br = 1000
    _list = [k for k in node_coords] + [v for _, v in edges2comps.items()]
    varis_comp = {k: varis[k] for k in _list}

    output_path = Path(__file__).parent
    no_sf, rules, rules_st, brs, sys_res = gen_bnb.do_gen_bnb(sys_fun, varis_comp, max_br=1000, output_path=output_path, key='pipe')

    return no_sf, rules, rules_st, brs, sys_res


@pytest.fixture()
def setup_comp_events(main_sys1, sub_sys):

    comps_st, edges, node_coords, es_idx, edges2comps, depots, varis = main_sys1

    sub_bw_nodes, sub_bw_edges = sub_sys

    cpms = {}

    # Component events
    for k in node_coords:
        cpms[k] = cpm.Cpm(variables=[varis[k]], no_child = 1, C = np.array([0, 1]), p = [0.1, 0.9])

    for e, x in edges2comps.items():
        cpms[x] = cpm.Cpm([varis[x]], no_child = 1, C = np.array([0, 1, 2]), p = [0.1, 0.2, 0.7])

    # Damage observation
    C_o = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    p_o = np.array([0.95, 0.05, 0.1, 0.9]).T
    for i, k in enumerate(node_coords, 1):
        name = f'on{i}'
	# observation that n_i = 0 or 1 ** TO DISCUSS: probably values in dictionary..?
        varis[name] = variable.Variable(name=name, B = np.eye(2), values = [0,1])
        cpms[name] = cpm.Cpm(variables=[varis[name], varis[k]], no_child = 1, C = C_o, p = p_o)

    C_o = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2], [2, 2]])
    p_o = np.array([0.95, 0.04, 0.01, 0.3, 0.5, 0.2, 0.01, 0.19, 0.8]).T
    for i, (_, k) in enumerate(edges2comps.items(), 1):
        name = f'ox{i}'
        varis[name] = variable.Variable(name=name, B = np.eye(3), values = [0, 1, 2]) # observation that x_i = 0, 1, or 2 ** TO DISCUSS: probably values in dictionary..?
        cpms[name] = cpm.Cpm(variables=[varis[name], varis[k]], no_child = 1, C = C_o, p = p_o)

    return cpms, varis


@pytest.mark.skip('NYI')
@pytest.fixture()
def setup_inference(main_sys1, setup_comp_events, request):

    _, edges, _, _, _, _, _ = main_sys1

    cpms, varis = setup_comp_events

    file_brs = Path(__file__).parent.joinpath('brs_pipes.pk')
    if file_brs.exists():
        with open(file_brs, 'rb') as fi:
            brs = pickle.load(fi)
            print(f'{file_brs} loaded')
    else:
        _, _, _, brs, _ = request.getfixturevalue('setup_brs')

    st_br_to_cs = {'fail': 0, 'surv': 1, 'unk': 2}

    csys, varis = gen_bnb.get_csys_from_brs(brs, varis, st_br_to_cs)

    varis['sys'] = variable.Variable('sys', np.eye(3), ['fail', 'surv', 'unk'])
    cpm_sys_vname = brs[0].names[:]
    cpm_sys_vname.insert(0, 'sys')

    cpms['sys'] = cpm.Cpm(
        variables=[varis[k] for k in cpm_sys_vname],
        no_child = 1,
        C = csys,
        p = np.ones((len(csys), 1), int))

    var_elim_order_name = [f'o{i}' for i in range(1, len(arcs) + 1)] + [f'e{i}' for i in range(1, len(arcs) + 1)] # observations first, components later

    var_elim_order = [varis[k] for k in var_elim_order_name]

    return cpms, varis, var_elim_order, arcs


@pytest.mark.skip('NYI')
def test_get_cys_from_brs_pipe(setup_comp_events, request):

    _, varis = setup_comp_events

    file_brs = Path(__file__).parent.joinpath('brs_pipes.pk')
    if file_brs.exists():
        with open(file_brs, 'rb') as fi:
            brs = pickle.load(fi)
            print(f'{file_brs} loaded')
    else:
        _, _, _, brs, _ = request.getfixturevalue('setup_brs')

    st_br_to_cs = {'fail': 0, 'surv': 1, 'unk': 2}

    csys, varis = gen_bnb.get_csys_from_brs(brs, varis, st_br_to_cs)

    print(csys)
