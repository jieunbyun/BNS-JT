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


def sf_min_path(comps_st, od_pair, arcs, vari, thres):
    """
    comps_st:
    od_pair:
    arcs:
    vari:
    thres:
    """
    elapsed, path = get_time_and_path(comps_st, od_pair, arcs, vari)

    # fail, surv corresponds to 0 and 1
    min_comps_st = {}
    if elapsed > thres:
        sys_st = 'fail'
    else:
        sys_st = 'surv'
        for n0, n1 in zip(path[:-1], path[1:]):
            arc = next((k for k, v in arcs.items() if v == [n0, n1] or v == [n1, n0]), None)
            min_comps_st[arc] = comps_st[arc]

    return elapsed, sys_st, min_comps_st


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


# # Define system function
def get_time_and_path(comps_st, od_pair, arcs, vari):
    """
    comps_st: starting from 0
    od_pair:
    arcs:
    vari:
    """
    assert isinstance(comps_st, dict)
    assert all([comps_st[k] < len(v.values) for k, v in vari.items()])

    G = nx.Graph()
    for k, x in arcs.items():
        G.add_edge(x[0], x[1], time=vari[k].values[comps_st[k]])

    path = nx.shortest_path(G, source = od_pair[0], target = od_pair[1], weight = 'time')
    elapsed = nx.shortest_path_length(G, source = od_pair[0], target = od_pair[1], weight = 'time')

    return elapsed, path


def sys_fun_wrap(od_pair, arcs, varis, thres):
    def sys_fun2(comps_st):
        return sf_min_path(comps_st, od_pair, arcs, varis, thres)
    return sys_fun2


def test_get_time_and_path1(main_sys):

    od_pair, arcs, varis = main_sys

    # all surv: e2 - e5 
    comps_st = {x: 2 for x in arcs.keys()}
    expected = varis['e2'].values[2] + varis['e5'].values[2]

    elapsed, path = get_time_and_path(comps_st, od_pair, arcs, varis)

    assert elapsed == expected
    assert path == ['n1', 'n5', 'n3']


def test_get_time_and_path2(main_sys):

    od_pair, arcs, varis = main_sys

    # e2 failed; e1 - e3 - e5
    comps_st = {'e1': 2, 'e2': 0, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    expected = varis['e1'].values[2] + varis['e5'].values[2] + varis['e3'].values[2]

    elapsed, path = get_time_and_path(comps_st, od_pair, arcs, varis)

    assert pytest.approx(elapsed, 1.0e-3) == expected
    assert path == ['n1', 'n2', 'n5', 'n3']


def test_get_time_and_path3(main_sys):

    od_pair, arcs, varis = main_sys

    # e1 failed; e2-e5
    comps_st = {'e1':0, 'e2':2, 'e3':1, 'e4':2, 'e5':1, 'e6': 1}
    expected = min(varis['e2'].values[2] + varis['e6'].values[1] + varis['e4'].values[2], varis['e2'].values[2] + varis['e5'].values[1])

    elapsed, path = get_time_and_path(comps_st, od_pair, arcs, varis)

    assert pytest.approx(elapsed, 1.0e-3) == expected
    assert path == ['n1', 'n5', 'n3']


def test_sf_min_path1(main_sys):

    od_pair, arcs, varis = main_sys

    # 0, 1, 2 (higher, better)
    comps_st = {k: 2 for k, v in varis.items()} # intact state (i.e. the highest state)
    elapsed, path = get_time_and_path(comps_st, od_pair, arcs, varis)

    thres = 2 * elapsed
    result = sf_min_path(comps_st, od_pair, arcs, varis, thres)

    assert pytest.approx(result[0], 0.001) == 0.1844
    assert result[1] == 'surv'
    assert result[2] == {'e2': 2, 'e5': 2}


def test_sf_min_path2(main_sys):

    od_pair, arcs, varis = main_sys

    # 0, 1, 2 (higher, better)
    comps_st = {'e1': 2, 'e2': 0, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    thres = 2 * 0.1844
    result = sf_min_path(comps_st, od_pair, arcs, varis, thres)

    assert pytest.approx(result[0], 0.001) == 0.3344
    assert result[1] == 'surv'
    assert result[2] == {'e1': 2, 'e3': 2 , 'e5': 2}


def test_sf_min_path3(main_sys):

    od_pair, arcs, varis = main_sys
    thres = 2 * 0.1844

    # 0, 1, 2 (higher, better)
    comps_st = {'e1':0, 'e2':2, 'e3':1, 'e4':2, 'e5':1, 'e6': 1}

    result = sf_min_path(comps_st, od_pair, arcs, varis, thres)

    assert pytest.approx(result[0], 0.001) == 0.2787
    assert result[1] == 'surv'
    assert result[2] == {'e2': 2, 'e5': 1}


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

    od_pair, arcs, varis = main_sys
    #comps_name = list(arcs.keys())

    # Intact state of component vector: zero-based index 
    comps_st_itc = {k: v.B.shape[1] - 1 for k, v in varis.items()} # intact state (i.e. the highest state)
    elapsed_itc, path_itc = get_time_and_path(comps_st_itc, od_pair, arcs, varis)

    # defines the system failure event
    thres = 2

    # Given a system function, i.e. sf_min_path, it should be represented by a function that only has "comps_st" as input.
    sys_fun = sys_fun_wrap(od_pair, arcs, varis, thres * elapsed_itc)

    # # Branch and bound
    #pdb.set_trace()
    no_sf, rules, rules_st, brs, sys_res = gen_bnb.do_gen_bnb(sys_fun, varis, max_br=1000)

    # Result
    assert no_sf == 23
    assert len(rules) == 10
    assert len(brs) == 10

    #print(rules)
    #print(rules_st)
    #print(brs)

    #print(sys_res)

def test_get_compat_rules1():

    cst = {f'e{i}': 2 for i in range(1, 7)}
    rules = []
    rules_st = []

    result = gen_bnb.get_compat_rules(cst, rules, rules_st)

    assert result[0] == []
    assert result[1] == 'unk'


def test_get_compat_rules2():

    cst = {f'e{i}': 2 for i in range(1, 7)}
    rules = [{'e2': 2, 'e5': 2}]
    rules_st = ['surv']

    result = gen_bnb.get_compat_rules(cst, rules, rules_st)

    assert result[0] == [0]
    assert result[1] == 'surv'


def test_get_compat_rules3():

    cst = {f'e{i}': 0 for i in range(1, 7)}
    rules = [{'e2': 2, 'e5': 2}]
    rules_st = ['surv']

    result = gen_bnb.get_compat_rules(cst, rules, rules_st)

    assert result[0] == []
    assert result[1] == 'unk'


def test_add_rule1():
    rules = [{'e2':2, 'e5':2}]
    rules_st = ['surv']
    rule_new = {f'e{i}':0 for i in range(1, 7)}
    fail_or_surv = 'fail'

    result = gen_bnb.add_rule(rules, rules_st, rule_new, fail_or_surv)

    assert result[0] == [{'e2': 2, 'e5': 2}, {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5':0, 'e6': 0}]
    assert result[1] == ['surv', 'fail']


def test_get_comp_st_for_next_bnb():
    up = {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    down = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}
    rules = [{'e2': 2, 'e5': 2},
             {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}]
    rules_st = ['surv', 'fail']

    result = gen_bnb.get_comp_st_for_next_bnb(up, down, rules, rules_st)

    assert result[0] == 'e5'
    assert result[1] == 2


def test_decomp_to_two_branches():
    comps_name = ['e1', 'e2', 'e3', 'e4', 'e5' ,'e6']
    br = branch.Branch(down=[0, 0, 0, 0, 0, 0], up=[2, 2, 2, 2, 2, 2], is_complete=False, names=comps_name)
    br.down_state='fail' # FIXME
    br.up_state='surv' # FIXME
    comp_bnb = 'e5'
    st_bnb_up = 2

    result = gen_bnb.decomp_to_two_branches(br, comp_bnb, st_bnb_up)

    assert result[0] == branch.Branch(down=[0, 0, 0, 0, 0, 0], up=[2, 2, 2, 2, 1, 2], names=comps_name, is_complete=False, down_state=1, up_state=1)

    assert result[1] == branch.Branch(down=[0, 0, 0, 0, 2, 0], up=[2, 2, 2, 2, 2, 2], names=comps_name, is_complete=False, down_state=1, up_state=1)


def test_get_sys_rules1(main_sys):

    od_pair, arcs, varis = main_sys

    thres = 2 * 0.1844
    sys_fun = sys_fun_wrap(od_pair, arcs, varis, thres)

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
    sys_fun = sys_fun_wrap(od_pair, arcs, varis, thres)

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
    sys_fun = sys_fun_wrap(od_pair, arcs, varis, thres)

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
