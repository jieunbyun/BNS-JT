#!/usr/bin/env python
# coding: utf-8

import pytest

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

from BNS_JT import trans
from BNS_JT import variable
from BNS_JT import gen_bnb


HOME = Path(__file__).parent

@pytest.fixture()
def main_sys():

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

    arcs_avg_kmh = {'e1': 40,
                    'e2': 40,
                    'e3': 40,
                    'e4': 30,
                    'e5': 30,
                    'e6': 20}

    od_pair = ('n1', 'n3')

    arc_lens_km = trans.get_arcs_length(arcs, node_coords)
    arc_times_h = {k: v/arcs_avg_kmh[k] for k, v in arc_lens_km.items()}

    comps_name = [k for k in arcs] # *FIXME* this is not necessary if branch's down and up are defined by dictionary (instead of list)

    # Component events
    no_arc_st = 3 # number of component states 
    delay_rat = [10, 2, 1] # delay in travel time given each component state (ratio)
    varis = {}
    for k, v in arcs.items():
        varis[k] = variable.Variable(name=k, B = np.eye(no_arc_st), values = [arc_times_h[k]*np.float64(x) for x in delay_rat])

    # plot graph
    G = nx.Graph()
    for k, x in arcs.items():
        G.add_edge(x[0], x[1], time=arc_times_h[k], label=k)

    for k, v in node_coords.items():
        G.add_node(k, pos=v)

    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = nx.get_edge_attributes(G, 'label')

    fig = plt.figure()
    ax = fig.add_subplot()
    nx.draw(G, pos, with_labels=True, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    fig.savefig(HOME.joinpath('graph_test_gen_bnb.png'), dpi=200)

    return od_pair, arcs, varis, comps_name

# # Define system function
def get_time_and_path(comps_st, od_pair, arcs, vari):

    G = nx.Graph()
    for k, x in arcs.items():
        c_st = comps_st[k]
        G.add_edge(x[0], x[1], time=vari[k].values[c_st-1])

    path = nx.shortest_path(G, source = od_pair[0], target = od_pair[1], weight = 'time')
    time = nx.shortest_path_length(G, source = od_pair[0], target = od_pair[1], weight = 'time')

    return time, path


def sf_min_path(comps_st, od_pair, arcs, vari, thres, sys_val_itc):

    sys_val, path = get_time_and_path(comps_st, od_pair, arcs, vari)

    if sys_val > thres*sys_val_itc:
        sys_st = 'fail'
    else:
        sys_st = 'surv'

    if sys_st == 'surv': # in this case we know how to find out minimally required component state
        min_comps_st = {}
        for i in range(len(path)-1):
            nodes_i = [path[i], path[i+1]]
            nodes_i_rev = [path[i+1], path[i]] # reversed pair (arcs are bi-directional)
            arc_i = next((k for k, v in arcs.items() if v == nodes_i or v == nodes_i_rev), None)
            min_comps_st[arc_i] = comps_st[arc_i]

    else: # sys_st == 'fail'
        min_comps_st = None

    return sys_val, sys_st, min_comps_st


def test_do_gen_bnb(main_sys):

    # ## System function as an input
    # 
    # A system function needs to return (1) system function value, (2) system state, and (3) minimally required component state to fulfill the obtained system function value. If (3) is unavailable, it can be returned as None. <br>
    # It requires input being a component state in regard to which a system analysis needs to be done.
    # 
    # This branch and bound algorithm assumes that the worst state is 1, and the higher a state, the better component performance it represents. <br>
    # In addition, it presumes coherency of system event, i.e. a higher (lower) state of component vector does not worsen (improve) a system's performance.
    # 
    # Two things to note about a system function: <br>
    # (1) It is preferred if a system function returns a possible scenario of components performance that has the highest probability of occurrence. <br>
    # (2) It is preferred if one can obtain a minimum required state of component vector from a result of system analysis. In case that this is not possible, the reference component vector for decomposition becomes the input state of component vector (which work okay by the proposed branch and bound algorithm).
    # 
    # An example system function is as follows. <br>
    # Here, We define the system failure event as a travel time longer than thres*(normal time). <br>
    # It is noted that for a failure event, it returns 'None' for minimal (failure) rule since there is no efficient way to identify one.

    od_pair, arcs, varis, comps_name = main_sys

    # Intact state of component vector
    comps_st_itc = {k: len(varis[k].B[0]) for k in arcs} # intact state (i.e. the highest state)
    sys_val_itc, path_itc = get_time_and_path(comps_st_itc, od_pair, arcs, varis)

    # defines the system failure event
    thres = 2

    # Given a system function, i.e. sf_min_path, it should be represented by a function that only has "comps_st" as input.
    sys_fun = lambda comps_st : sf_min_path(comps_st, od_pair, arcs, varis, thres, sys_val_itc) # FIXME: branch needs to have states defined in dictionary instead of list.

    # # Branch and bound
    no_sf, rules, rules_st, brs, sys_res = gen_bnb.do_gen_bnb(sys_fun, varis, comps_name, max_br=1000)

    # Result
    assert no_sf == 23
    assert len(rules) == 10
    assert len(brs) == 10

    #print(rules)
    #print(rules_st)
    #print(brs)

    #print(sys_res)

