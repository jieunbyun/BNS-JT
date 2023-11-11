#!/usr/bin/env python
# coding: utf-8

import pytest
import pdb
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

from BNS_JT import trans, branch
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

    #comps_name = [k for k in arcs] # *FIXME* this is not necessary if branch's down and up are defined by dictionary (instead of list)

    # Component events
    no_arc_st = 3 # number of component states 
    delay_rat = [10, 2, 1] # delay in travel time given each component state (ratio)
    varis = {}
    for k, v in arcs.items():
        varis[k] = variable.Variable(name=k, B = np.eye(no_arc_st), values = [arc_times_h[k]*np.float64(x) for x in delay_rat])

    # plot graph
    G = nx.Graph()
    for k, x in arcs.items():
        G.add_edge(x[0], x[1], label=k)

    for k, v in node_coords.items():
        G.add_node(k, pos=v)

    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = nx.get_edge_attributes(G, 'label')

    fig = plt.figure()
    ax = fig.add_subplot()
    nx.draw(G, pos, with_labels=True, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    fig.savefig(HOME.joinpath('graph_test_gen_bnb.png'), dpi=200)

    """
    e1: 1.5, 0.3, 0.15
    e2: 0.901, 0.1803, 0.0901
    e3: 0.901, 0.1803, 0.0901
    e4: 1.054, 0.211, 0.1054
    e5: 0.943, 0.189, 0.0943
    e6: 0.707, 0.141, 0.0707
    """
    return od_pair, arcs, varis



@pytest.fixture()
def comps_st_dic():

    comps_st = {}
    expected = {}

    # 0, 1, 2 (higher, better)
    # intact state (i.e. the highest state)
    comps_st[0] = {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    expected[0] = (0.1844, 'surv', {'e2': 2, 'e5': 2}, ['n1', 'n5', 'n3'])

    comps_st[1] = {'e1': 0, 'e2': 1, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 2}
    expected[1] = (1.12308, 'fail', {}, ['n1', 'n5', 'n3'])

    comps_st[2] = {'e1': 2, 'e2': 0, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    expected[2] = (0.3344, 'surv', {'e1': 2, 'e3': 2 , 'e5': 2}, ['n1', 'n2', 'n5', 'n3'])

    comps_st[3] = {'e1': 0, 'e2': 2, 'e3': 1, 'e4': 2, 'e5': 1, 'e6': 1}
    expected[3] = (0.2787, 'surv', {'e2': 2, 'e5': 1}, ['n1', 'n5', 'n3'])

    comps_st[4] = {'e1': 2, 'e2': 2, 'e3': 0, 'e4': 1, 'e5': 1, 'e6': 2}
    expected[4] = (0.2787, 'surv', {'e2': 2, 'e5': 1}, ['n1', 'n5', 'n3'])

    comps_st[5] = {'e1':2, 'e2':1, 'e3':2, 'e4':2, 'e5':2, 'e6': 2}
    expected[5] = (0.2746, 'surv', {'e2': 1, 'e5': 2}, ['n1', 'n5', 'n3'])

    comps_st[6] = {'e1':2, 'e2':2, 'e3':2, 'e4':2, 'e5':1, 'e6': 2}
    expected[6] = (0.26626, 'surv', {'e2': 2, 'e6': 2, 'e4':2}, ['n1', 'n5', 'n4', 'n3'])

    comps_st[7] = {'e1':1, 'e2':0, 'e3':2, 'e4':2, 'e5':2, 'e6': 2}
    expected[7] = (0.4844, 'fail', {}, ['n1', 'n2', 'n5', 'n3'])

    comps_st[7] = {'e1':2, 'e2':0, 'e3':1, 'e4':2, 'e5':2, 'e6': 2}
    expected[7] = (0.4246, 'fail', {}, ['n1', 'n2', 'n5', 'n3'])


    return comps_st, expected


def test_get_time_and_path(main_sys, comps_st_dic):

    od_pair, arcs, varis = main_sys

    comps_st, expected = comps_st_dic

    for c in comps_st.keys():
        d_time, path = trans.get_time_and_path_given_comps(comps_st[c], od_pair, arcs, varis)

        assert pytest.approx(d_time, 0.001) == expected[c][0]
        assert path == expected[c][3]


def test_sf_min_path(main_sys, comps_st_dic):

    od_pair, arcs, varis = main_sys
    comps_st, expected = comps_st_dic
    thres = 2*0.1844
    for c in comps_st.keys():

        d_time, path = trans.get_time_and_path_given_comps(comps_st[c], od_pair, arcs, varis)

        result = trans.sf_min_path(comps_st[c], od_pair, arcs, varis, thres)

        assert pytest.approx(result[0], 0.001) == expected[c][0]
        assert result[1] == expected[c][1]
        assert result[2] == expected[c][2]


def test_init_brs1(main_sys):

    _, _, varis = main_sys
    rules = []
    rules_st = []

    brs = gen_bnb.init_brs(varis, rules, rules_st)

    assert len(brs) == 1
    assert brs[0].up_state == 'unk'
    assert brs[0].down_state == 'unk'
    assert brs[0].down == [0, 0, 0, 0, 0, 0]
    assert brs[0].up == [2, 2, 2, 2, 2, 2]


def test_init_brs2(main_sys):

    _, _, varis = main_sys
    rules = [{'e2': 2, 'e5': 2}]
    rules_st = ['surv']

    brs = gen_bnb.init_brs(varis, rules, rules_st)

    assert len(brs) == 1
    assert brs[0].up_state == 'surv'
    assert brs[0].down_state == 'unk'
    assert brs[0].down == [0, 0, 0, 0, 0, 0]
    assert brs[0].up == [2, 2, 2, 2, 2, 2]


def test_init_brs3(main_sys):

    _, _, varis = main_sys
    rules = [{'e2': 2, 'e5': 2}, {x: 0 for x in varis.keys()}]
    rules_st = ['surv', 'fail']

    brs = gen_bnb.init_brs(varis, rules, rules_st)

    assert len(brs) == 1
    assert brs[0].up_state == 'surv'
    assert brs[0].down_state == 'fail'
    assert brs[0].down == [0, 0, 0, 0, 0, 0]
    assert brs[0].up == [2, 2, 2, 2, 2, 2]


def test_init_brs4(main_sys):

    _, _, varis = main_sys
    rules = [{'e2': 2, 'e5': 2}, {x: 0 for x in varis.keys()}, {'e2': 2, 'e6': 2, 'e4': 2}]
    rules_st = ['surv', 'fail', 'surv']

    brs = gen_bnb.init_brs(varis, rules, rules_st)

    assert len(brs) == 1
    assert brs[0].up_state == 'surv'
    assert brs[0].down_state == 'fail'
    assert brs[0].down == [0, 0, 0, 0, 0, 0]
    assert brs[0].up == [2, 2, 2, 2, 2, 2]


def test_core1(main_sys):

    _, _, varis = main_sys
    rules = []
    rules_st = []
    cst = []
    stop_br = False
    brs = gen_bnb.init_brs(varis, rules, rules_st)

    brs, cst, stop_br = gen_bnb.core(brs, rules, rules_st, cst, stop_br)

    assert brs == []
    assert cst == [2, 2, 2, 2, 2, 2]
    assert stop_br == True


def test_core2(main_sys):

    od_pair, arcs, varis = main_sys

    cst = [2, 2, 2, 2, 2, 2]
    stop_br = True
    rules = [{'e2': 2, 'e5': 2}]
    rules_st = ['surv']
    brs = gen_bnb.init_brs(varis, rules, rules_st)

    #pdb.set_trace()
    brs, cst, stop_br = gen_bnb.core(brs, rules, rules_st, cst, stop_br)
    assert brs == []
    assert cst == [0, 0, 0, 0, 0, 0]
    assert stop_br == True


def test_core3(main_sys):

    od_pair, arcs, varis = main_sys

    cst = [0, 0, 0, 0, 0, 0]
    stop_br = True
    rules = [{'e2': 2, 'e5': 2}, {x: 0 for x in varis.keys()}]
    rules_st = ['surv', 'fail']
    brs = gen_bnb.init_brs(varis, rules, rules_st)

    #pdb.set_trace()
    brs, cst, stop_br = gen_bnb.core(brs, rules, rules_st, cst, stop_br)
    assert brs == []
    assert cst == [2, 2, 2, 2, 1, 2]
    assert stop_br == True


def test_core4(main_sys):

    od_pair, arcs, varis = main_sys

    cst = [2, 2, 2, 2, 1, 2]
    stop_br = True
    rules = [{'e2': 2, 'e5': 2}, {x: 0 for x in varis.keys()}, {'e2': 2, 'e6': 2, 'e4': 2}]
    rules_st = ['surv', 'fail', 'surv']
    brs = gen_bnb.init_brs(varis, rules, rules_st)

    #pdb.set_trace()
    brs, cst, stop_br = gen_bnb.core(brs, rules, rules_st, cst, stop_br)
    assert brs == []
    assert cst == [2, 1, 2, 2, 2, 2]
    assert stop_br == True


def test_do_gen_bnb2(main_sys):
    # no_iter: 10
    od_pair, arcs, varis = main_sys

    rules = [{'e2': 2, 'e6': 2, 'e4': 2}, {'e2': 1, 'e5': 2}, {'e1': 2, 'e3': 2, 'e5': 2}, {'e1': 0, 'e2': 1, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, {'e2': 0, 'e3': 1}, {'e1': 0, 'e2': 0, 'e4': 0, 'e5': 0, 'e6': 0}]
    rules_st = ['surv', 'surv', 'surv', 'fail', 'fail', 'fail']
    cst = [0, 0, 2, 0, 0, 0]
    thres = 2 * 0.1844

    sys_fun = trans.sys_fun_wrap(od_pair, arcs, varis, thres)

    brs = gen_bnb.init_brs(varis, rules, rules_st)
    stop_br = False
    flag=True
    #pdb.set_trace()
    while flag:
        brs, cst, stop_br = gen_bnb.core(brs, rules, rules_st, cst, stop_br)
        if stop_br:
            break
        else:
            flag = any([not b.is_complete for b in brs])

    sys_res_, rules, rules_st = gen_bnb.get_sys_rules(cst, sys_fun, rules, rules_st, varis)

    assert pytest.approx(sys_res_['sys_val'].values[0], 0.001) == 0.41626


def test_do_gen_bnb11(main_sys):
    # iteration 11
    od_pair, arcs, varis = main_sys

    rules = [{'e2': 2, 'e6': 2, 'e4': 2}, {'e2': 1, 'e5': 2}, {'e1': 2, 'e3': 2, 'e5': 2}, {'e1': 0, 'e2': 1, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, {'e2': 0, 'e3': 1}, {'e2': 0, 'e5': 1}]
    rules_st = ['surv', 'surv', 'surv', 'fail', 'fail', 'fail']
    cst = [2, 0, 2, 2, 1, 2]
    thres = 2 * 0.1844

    sys_fun = trans.sys_fun_wrap(od_pair, arcs, varis, thres)

    brs = gen_bnb.init_brs(varis, rules, rules_st)
    stop_br = False
    flag=True
    #pdb.set_trace()
    while flag:
        brs, cst, stop_br = gen_bnb.core(brs, rules, rules_st, cst, stop_br)
        if stop_br:
            break
        else:
            flag = any([not b.is_complete for b in brs])

    sys_res_, rules, rules_st = gen_bnb.get_sys_rules(cst, sys_fun, rules, rules_st, varis)

    assert pytest.approx(sys_res_['sys_val'].values[0], 0.001) == 0.9957


def test_do_gen_bnb3(main_sys):
    # iteration 21
    od_pair, arcs, varis = main_sys

    rules = [{'e1': 2, 'e3': 2, 'e5': 2}, {'e2': 0, 'e3': 1}, {'e2': 0, 'e5': 1}, {'e2': 1, 'e6': 2, 'e4': 2}, {'e2': 1, 'e5': 1}, {'e4': 1, 'e5': 0}, {'e1': 1, 'e2': 0}, {'e2': 2, 'e6': 1, 'e4': 2}, {'e5': 0, 'e6': 0}]
    rules_st = ['surv', 'fail', 'fail', 'surv', 'surv', 'fail', 'fail', 'surv', 'fail']
    cst = [2, 2, 2, 2, 0, 0]
    thres = 2 * 0.1844

    sys_fun = trans.sys_fun_wrap(od_pair, arcs, varis, thres)

    brs = gen_bnb.init_brs(varis, rules, rules_st)
    stop_br = False
    flag=True
    while flag:
        brs, cst, stop_br = gen_bnb.core(brs, rules, rules_st, cst, stop_br)
        if stop_br:
            break
        else:
            flag = any([not b.is_complete for b in brs])

    sys_res_, rules, rules_st = gen_bnb.get_sys_rules(cst, sys_fun, rules, rules_st, varis)

    assert pytest.approx(sys_res_['sys_val'].values[0], 0.001) == 0.4271


def test_do_gen_bnb1(main_sys):
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

    od_pair, arcs, varis = main_sys
    #comps_name = list(arcs.keys())

    # Intact state of component vector: zero-based index 
    comps_st_itc = {k: v.B.shape[1] - 1 for k, v in varis.items()} # intact state (i.e. the highest state)
    d_time_itc, path_itc = trans.get_time_and_path_given_comps(comps_st_itc, od_pair, arcs, varis)

    # defines the system failure event
    thres = 2

    # Given a system function, i.e. sf_min_path, it should be represented by a function that only has "comps_st" as input.
    sys_fun = trans.sys_fun_wrap(od_pair, arcs, varis, thres * d_time_itc)

    # # Branch and bound
    #pdb.set_trace()
    # break until 0.424558
    no_sf, rules, rules_st, brs, sys_res = gen_bnb.do_gen_bnb(sys_fun, varis, max_br=1000)

    # Result
    assert no_sf == 23
    assert len(rules) == 10
    assert len(brs) == 10

    expected_rules =[{'e1': 2, 'e3': 2, 'e5': 2}, {'e2': 0, 'e3': 1}, {'e2': 0, 'e5': 1}, {'e2': 1, 'e6': 2, 'e4': 2}, {'e2': 1, 'e5': 1}, {'e4': 1, 'e5': 0}, {'e1': 1, 'e2': 0}, {'e2': 2, 'e6': 1, 'e4': 2}, {'e5': 0, 'e6': 0}, {'e2': 1, 'e5': 0, 'e6': 1}]
    assert rules == expected_rules
    #expected_rules_st = ['surv', 'fail', 'fail', 'surv', 'surv', 'fail', 'fail', 'surv', 'fail', 'fail']
    #assert rules_st == expected_rules_st
    """
    expected_brs = [branch.Branch(down=[1, 1, 1, 1, 1, 1], up=[3, 1, 2, 3, 3, 3], is_complete=True, down_state=fail, up_state=fail, down_val=None, up_val=None),
                    branch.Branch(down=[1, 1, 3, 1, 1, 1], up=[3, 1, 3, 3, 2, 3], is_complete=True, down_state=fail, up_state=fail, down_val=None, up_val=None),
                    branch.Branch(down=[1, 1, 3, 1, 3, 1], up=[2, 1, 3, 3, 3, 3], is_complete=True, down_state=fail, up_state=fail, down_val=None, up_val=None),
                    branch.Branch(down=[3, 1, 3, 1, 3, 1], up=[3, 1, 3, 3, 3, 3], is_complete=True, down_state=surv, up_state=surv, down_val=None, up_val=None),
                    branch.Branch(down=[1, 2, 1, 1, 1, 1], up=[3, 3, 3, 2, 1, 3], is_complete=True, down_state=fail, up_state=fail, down_val=None, up_val=None),
                    branch.Branch(down=[1, 2, 1, 3, 1, 1], up=[3, 3, 3, 3, 1, 1], is_complete=True, down_state=fail, up_state=fail, down_val=None, up_val=None),
                    branch.Branch(down=[1, 2, 1, 3, 1, 2], up=[3, 2, 3, 3, 1, 2], is_complete=True, down_state=fail, up_state=fail, down_val=None, up_val=None),
                    branch.Branch(down=[1, 3, 1, 3, 1, 2], up=[3, 3, 3, 3, 1, 2], is_complete=True, down_state=surv, up_state=surv, down_val=None, up_val=None),
                    branch.Branch(down=[1, 2, 1, 3, 1, 3], up=[3, 3, 3, 3, 1, 3], is_complete=True, down_state=surv, up_state=surv, down_val=None, up_val=None),
                    branch.Branch(down=[1, 2, 1, 1, 2, 1], up=[3, 3, 3, 3, 3, 3], is_complete=True, down_state=surv, up_state=surv, down_val=None, up_val=None)]
    """
    expected = np.array([0.184420,
                         1.844197,
                         0.266259,
                         0.274558,
                         0.334420,
                         0.995669,
                         1.123087,
                         0.424558,
                         1.844197,
                         0.416259,
                         0.995669,
                         0.278701,
                         0.356397,
                         1.032948,
                         0.368839,
                         0.371668,
                         0.992794,
                         0.484420,
                         0.336969,
                         0.902655,
                         0.427108,
                         0.427108,
                         0.427108])

    np.testing.assert_array_almost_equal(sys_res['sys_val'].values, expected)

"""
     sys_val                                           comps_st  \
0   {'e1': 3, 'e2': 3, 'e3': 3, 'e4': 3, 'e5': 3, ...   
1   {'e1': 1, 'e2': 1, 'e3': 1, 'e4': 1, 'e5': 1, ...   
2   {'e1': 3, 'e2': 3, 'e3': 3, 'e4': 3, 'e5': 2, ...   
3   {'e1': 3, 'e2': 2, 'e3': 3, 'e4': 3, 'e5': 3, ...   
4   {'e1': 3, 'e2': 1, 'e3': 3, 'e4': 3, 'e5': 3, ...   
5   {'e1': 1, 'e2': 1, 'e3': 1, 'e4': 1, 'e5': 3, ...   
6   {'e1': 1, 'e2': 2, 'e3': 1, 'e4': 1, 'e5': 1, ...   
7   {'e1': 3, 'e2': 1, 'e3': 2, 'e4': 3, 'e5': 3, ...   
8   {'e1': 1, 'e2': 1, 'e3': 3, 'e4': 1, 'e5': 1, ...   
9   {'e1': 3, 'e2': 1, 'e3': 3, 'e4': 3, 'e5': 2, ...   
10  {'e1': 1, 'e2': 1, 'e3': 3, 'e4': 1, 'e5': 3, ...   
11  {'e1': 3, 'e2': 3, 'e3': 3, 'e4': 2, 'e5': 2, ...   
12  {'e1': 3, 'e2': 2, 'e3': 3, 'e4': 3, 'e5': 2, ...   
13  {'e1': 1, 'e2': 3, 'e3': 1, 'e4': 1, 'e5': 1, ...   
14  {'e1': 1, 'e2': 2, 'e3': 1, 'e4': 1, 'e5': 2, ...   
15  {'e1': 3, 'e2': 3, 'e3': 3, 'e4': 2, 'e5': 1, ...   
16  {'e1': 1, 'e2': 2, 'e3': 1, 'e4': 3, 'e5': 1, ...   
17  {'e1': 2, 'e2': 1, 'e3': 3, 'e4': 3, 'e5': 3, ...   
18  {'e1': 3, 'e2': 3, 'e3': 3, 'e4': 3, 'e5': 1, ...   
19  {'e1': 3, 'e2': 3, 'e3': 3, 'e4': 3, 'e5': 1, ...   
20  {'e1': 1, 'e2': 2, 'e3': 1, 'e4': 3, 'e5': 1, ...   
21  {'e1': 3, 'e2': 2, 'e3': 3, 'e4': 3, 'e5': 1, ...   
22  {'e1': 3, 'e2': 2, 'e3': 3, 'e4': 3, 'e5': 1, ...   

                   comps_st_min  
0            {'e2': 3, 'e5': 3}  
1                          None  
2   {'e2': 3, 'e6': 3, 'e4': 3}  
3            {'e2': 2, 'e5': 3}  
4   {'e1': 3, 'e3': 3, 'e5': 3}  
5                          None  
6                          None  
7                          None  
8                          None  
9                          None  
10                         None  
11           {'e2': 3, 'e5': 2}  
12  {'e2': 2, 'e6': 3, 'e4': 3}  
13                         None  
14           {'e2': 2, 'e5': 2}  
15                         None  
16                         None  
17                         None  
18  {'e2': 3, 'e6': 2, 'e4': 3}  
19                         None  
20                         None  
21                         None  
22                         None  
"""


def test_get_compat_rules0():

    rules = []
    rules_st = []

    cst = {f'e{i}': 2 for i in range(1, 7)}
    result = gen_bnb.get_compat_rules(cst, rules, rules_st)

    assert result[0] == []
    assert result[1] == 'unk'

    cst = {f'e{i}': 0 for i in range(1, 7)}
    result = gen_bnb.get_compat_rules(cst, rules, rules_st)

    assert result[0] == []
    assert result[1] == 'unk'


def test_get_compat_rules1():

    rules = [{'e2': 2, 'e5': 2}]
    rules_st = ['surv']

    cst = {f'e{i}': 2 for i in range(1, 7)}
    result = gen_bnb.get_compat_rules(cst, rules, rules_st)

    assert result[0] == [0]
    assert result[1] == 'surv'

    cst = {f'e{i}': 0 for i in range(1, 7)}
    result = gen_bnb.get_compat_rules(cst, rules, rules_st)

    assert result[0] == []
    assert result[1] == 'unk'


def test_get_compat_rules2():

    rules = [{'e2': 1, 'e5': 2}]
    rules_st = ['surv']

    cst = {f'e{i}': 2 for i in range(1, 7)}
    result = gen_bnb.get_compat_rules(cst, rules, rules_st)

    assert result[0] == [0]
    assert result[1] == 'surv'

    cst = {f'e{i}': 0 for i in range(1, 7)}
    result = gen_bnb.get_compat_rules(cst, rules, rules_st)

    assert result[0] == []
    assert result[1] == 'unk'


def test_get_compat_rules3():

    cst = {f'e{i}': 2 for i in range(1, 7)}
    rules = [{'e2': 1, 'e5': 2}, {'e1': 1, 'e2': 0}]
    rules_st = ['surv', 'fail']

    result = gen_bnb.get_compat_rules(cst, rules, rules_st)

    assert result[0] == [0]
    assert result[1] == 'surv'

    cst = {f'e{i}': 0 for i in range(1, 7)}

    result = gen_bnb.get_compat_rules(cst, rules, rules_st)

    assert result[0] == [1]
    assert result[1] == 'fail'


def test_get_compat_rules4():

    cst = {'e1': 0, 'e2': 0, 'e3': 2 , 'e4': 1, 'e5': 0, 'e6': 2}
    rules = [{'e2': 1, 'e5': 2}, {'e1': 1, 'e2': 0}]
    rules_st = ['surv', 'fail']

    result = gen_bnb.get_compat_rules(cst, rules, rules_st)

    assert result[0] == [1]
    assert result[1] == 'fail'


def test_add_rule1():
    rules = [{'e2':2, 'e5':2}]
    rules_st = ['surv']
    rule_new = {f'e{i}':0 for i in range(1, 7)}
    fail_or_surv = 'fail'

    result = gen_bnb.add_rule(rules, rules_st, rule_new, fail_or_surv)

    assert result[0] == [{'e2': 2, 'e5': 2}, {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5':0, 'e6': 0}]
    assert result[1] == ['surv', 'fail']


def test_add_rule2():
    rules = [{'e2':2, 'e5':2}, {'e2': 1, 'e5': 1}]
    rules_st = ['surv', 'fail']
    rule_new = {'e2': 1, 'e5': 2}
    fail_or_surv = 'surv'

    result = gen_bnb.add_rule(rules, rules_st, rule_new, fail_or_surv)

    assert result[0] == [{'e2': 1, 'e5': 1}, {'e2': 1, 'e5': 2}]
    assert result[1] == ['fail', 'surv']

    rule_new = {'e2': 1, 'e5': 0}
    fail_or_surv = 'fail'

    result = gen_bnb.add_rule(rules, rules_st, rule_new, fail_or_surv)

    assert result[0] == [{'e2': 1, 'e5': 1}, {'e2': 1, 'e5': 2}]
    assert result[1] == ['fail', 'surv']


def test_get_comp_st_for_next_bnb0():
    up = {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    down = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}
    rules = [{'e2': 2, 'e5': 2},
             {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}]
    rules_st = ['surv', 'fail']
    #pdb.set_trace()
    result = gen_bnb.get_comp_st_for_next_bnb(up, down, rules, rules_st)

    assert result[0] == 'e5'
    assert result[1] == 2


@pytest.mark.skip('NYI')
def test_get_comp_st_for_next_bnb1():
    up = {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    down = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}
    rules = [{'e2': 2, 'e5': 2},
             {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}]
    rules_st = ['surv', 'fail']

    result = gen_bnb.get_comp_st_for_next_bnb(up, down, rules, rules_st)

    assert result[0] == 'e5'
    assert result[1] == 2


def test_decomp_to_two_branches1():
    comps_name = ['e1', 'e2', 'e3', 'e4', 'e5' ,'e6']
    br = branch.Branch(down=[0, 0, 0, 0, 0, 0], up=[2, 2, 2, 2, 2, 2], is_complete=False, names=comps_name)
    br.down_state='fail' # FIXME
    br.up_state='surv' # FIXME
    comp_bnb = 'e5'
    st_bnb_up = 2

    result = gen_bnb.decomp_to_two_branches(br, comp_bnb, st_bnb_up)

    assert result[0] == branch.Branch(down=[0, 0, 0, 0, 0, 0], up=[2, 2, 2, 2, 1, 2], names=comps_name, is_complete=False, down_state=1, up_state=1)

    assert result[1] == branch.Branch(down=[0, 0, 0, 0, 2, 0], up=[2, 2, 2, 2, 2, 2], names=comps_name, is_complete=False, down_state=1, up_state=1)


    br.down_state='fail' # FIXME
    br.up_state='fail' # FIXME
    comp_bnb = 'e2'
    st_bnb_up = 1
    result = gen_bnb.decomp_to_two_branches(br, comp_bnb, st_bnb_up)

    assert result[0] == branch.Branch(down=[0, 0, 0, 0, 0, 0], up=[2, 0, 2, 2, 2, 2], names=comps_name, is_complete=False, down_state=1, up_state=1)

    assert result[1] == branch.Branch(down=[0, 1, 0, 0, 0, 0], up=[2, 2, 2, 2, 2, 2], names=comps_name, is_complete=False, down_state=1, up_state=1)


def test_get_sys_rules1(main_sys):

    od_pair, arcs, varis = main_sys

    thres = 2 * 0.1844
    sys_fun = trans.sys_fun_wrap(od_pair, arcs, varis, thres)

    cst = [2, 2, 2, 2, 2, 2]
    rules = []
    rules_st = []

    #pdb.set_trace()
    sys_res, rules, rules_st = gen_bnb.get_sys_rules(cst, sys_fun, rules, rules_st, varis)

    np.testing.assert_almost_equal(sys_res['sys_val'].values, np.array([0.18442]), decimal=5)
    assert sys_res['comps_st'].values == [{k: 2 for k in varis.keys()}]
    assert sys_res['comps_st_min'].values == [{'e2': 2, 'e5': 2}]
    assert rules == [{'e2': 2, 'e5': 2}]
    assert rules_st == ['surv']


def test_get_sys_rules2(main_sys):

    od_pair, arcs, varis = main_sys

    thres = 2 * 0.1844
    sys_fun = trans.sys_fun_wrap(od_pair, arcs, varis, thres)

    cst = [0, 0, 0, 0, 0, 0]
    rules = [{'e2': 2, 'e5': 2}]
    rules_st = ['surv']

    #pdb.set_trace()
    sys_res, rules, rules_st = gen_bnb.get_sys_rules(cst, sys_fun, rules, rules_st, varis)

    np.testing.assert_almost_equal(sys_res['sys_val'].values, np.array([1.8442]), decimal=4)
    assert sys_res['comps_st'].values[0] == {k: 0 for k in varis.keys()}
    assert sys_res['comps_st_min'].values == [{}]
    assert rules[0] == {'e2': 2, 'e5': 2}
    assert rules[1] == {k: 0 for k in varis.keys()}
    assert rules_st == ['surv', 'fail']


def test_get_sys_rules3(main_sys):

    od_pair, arcs, varis = main_sys

    thres = 2 * 0.1844
    sys_fun = trans.sys_fun_wrap(od_pair, arcs, varis, thres)

    cst = [2, 2, 2, 2, 1, 2]
    rules = [{'e2': 2, 'e5': 2}, {k: 0 for k in varis.keys()}]
    rules_st = ['surv', 'fail']

    #pdb.set_trace()
    sys_res, rules, rules_st = gen_bnb.get_sys_rules(cst, sys_fun, rules, rules_st, varis)

    np.testing.assert_almost_equal(sys_res['sys_val'].values, np.array([0.266]), decimal=3)
    assert sys_res['comps_st'].values[0] == {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 1, 'e6': 2}
    assert sys_res['comps_st_min'].values == [{'e2': 2, 'e6': 2, 'e4': 2}]
    assert rules[0] == {'e2': 2, 'e5': 2}
    assert rules[1] == {k: 0 for k in varis.keys()}
    assert rules[2] == {'e2': 2, 'e6': 2, 'e4': 2}
    assert rules_st == ['surv', 'fail', 'surv']
