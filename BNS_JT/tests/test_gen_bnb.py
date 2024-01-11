# coding: utf-8

import pytest
import pdb
import pickle
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

from BNS_JT import trans, branch, variable, cpm, gen_bnb


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
def setup_comp_events(main_sys):

    _, arcs, varis = main_sys

    no_arc_st = 3

    cpms = {}

    # Component events
    for k in arcs:
        cpms[k] = cpm.Cpm(variables=[varis[k]],
                          no_child = 1,
                          C = np.array([0, 1, 2]),
                          p = [0.1, 0.2, 0.7])

    # Damage observation
    C_o = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2], [2, 2]])
    p_o = np.array([0.95, 0.04, 0.01, 0.3, 0.5, 0.2, 0.01, 0.19, 0.8]).T
    for i, k in enumerate(arcs, 1):
        name = f'o{i}'
        varis[name] = variable.Variable(name=name,
            B=np.eye(no_arc_st), values = [0, 1, 2]) # observation that e_i = 0, 1, or 2 ** TO DISCUSS: probably values in dictionary..?
        cpms[name] = cpm.Cpm(variables=[varis[name], varis[k]], no_child=1, C=C_o, p=p_o)

    return cpms, varis


@pytest.fixture()
def setup_brs(main_sys):

    od_pair, arcs, varis = main_sys

    comps_name = list(arcs.keys())

    # Intact state of component vector: zero-based index 
    comps_st_itc = {k: v.B.shape[1] - 1 for k, v in varis.items()} # intact state (i.e. the highest state)
    d_time_itc, path_itc = trans.get_time_and_path_given_comps(comps_st_itc, od_pair, arcs, varis)

    # defines the system failure event
    thres = 2

    # Given a system function, i.e. sf_min_path, it should be represented by a function that only has "comps_st" as input.
    sys_fun = trans.sys_fun_wrap(od_pair, arcs, varis, thres * d_time_itc)

    # Branch and bound
    output_path = Path(__file__).parent
    #pdb.set_trace()
    no_sf, rules, rules_st, brs, sys_res = gen_bnb.do_gen_bnb(sys_fun, varis, max_br=1000,
                                                              output_path=output_path, key='bridge', flag=True)

    return no_sf, rules, rules_st, brs, sys_res


@pytest.fixture()
def comps_st_dic():

    comps_st = {}
    expected = {}

    # 0, 1, 2 (higher, better)
    # intact state (i.e. the highest state)
    comps_st[0] = {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    expected[0] = (0.1844, 's', {'e2': 2, 'e5': 2}, ['n1', 'n5', 'n3'])

    comps_st[1] = {'e1': 0, 'e2': 1, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 2}
    expected[1] = (1.12308, 'f', {}, ['n1', 'n5', 'n3'])

    comps_st[2] = {'e1': 2, 'e2': 0, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    expected[2] = (0.3344, 's', {'e1': 2, 'e3': 2 , 'e5': 2}, ['n1', 'n2', 'n5', 'n3'])

    comps_st[3] = {'e1': 0, 'e2': 2, 'e3': 1, 'e4': 2, 'e5': 1, 'e6': 1}
    expected[3] = (0.2787, 's', {'e2': 2, 'e5': 1}, ['n1', 'n5', 'n3'])

    comps_st[4] = {'e1': 2, 'e2': 2, 'e3': 0, 'e4': 1, 'e5': 1, 'e6': 2}
    expected[4] = (0.2787, 's', {'e2': 2, 'e5': 1}, ['n1', 'n5', 'n3'])

    comps_st[5] = {'e1':2, 'e2':1, 'e3':2, 'e4':2, 'e5':2, 'e6': 2}
    expected[5] = (0.2746, 's', {'e2': 1, 'e5': 2}, ['n1', 'n5', 'n3'])

    comps_st[6] = {'e1':2, 'e2':2, 'e3':2, 'e4':2, 'e5':1, 'e6': 2}
    expected[6] = (0.26626, 's', {'e2': 2, 'e6': 2, 'e4':2}, ['n1', 'n5', 'n4', 'n3'])

    comps_st[7] = {'e1':1, 'e2':0, 'e3':2, 'e4':2, 'e5':2, 'e6': 2}
    expected[7] = (0.4844, 'f', {}, ['n1', 'n2', 'n5', 'n3'])

    comps_st[7] = {'e1':2, 'e2':0, 'e3':1, 'e4':2, 'e5':2, 'e6': 2}
    expected[7] = (0.4246, 'f', {}, ['n1', 'n2', 'n5', 'n3'])


    return comps_st, expected


def test_get_time_and_path(main_sys, comps_st_dic):

    od_pair, arcs, varis = main_sys

    comps_st, expected = comps_st_dic

    for c in comps_st.keys():
        d_time, path = trans.get_time_and_path_given_comps(comps_st[c], od_pair, arcs, varis)

        assert pytest.approx(d_time, 0.001) == expected[c][0]
        assert path == expected[c][3]


def test_init_branch_old1(main_sys):

    _, _, varis = main_sys
    rules = []
    rules_st = []

    brs = gen_bnb.init_branch_old(varis, rules, rules_st)

    assert len(brs) == 1
    assert brs[0].up_state == 'u'
    assert brs[0].down_state == 'u'
    assert brs[0].down == [0 for i in range(1, 7)]
    assert brs[0].up == [2 for i in range(1, 7)]


def test_init_branch_old2(main_sys):

    _, _, varis = main_sys
    rules = [{'e2': 2, 'e5': 2}]
    rules_st = ['s']

    brs = gen_bnb.init_branch_old(varis, rules, rules_st)
    assert len(brs) == 1
    assert brs[0].up_state == 's'
    assert brs[0].down_state == 'u'
    assert brs[0].down == [0 for i in range(1, 7)]
    assert brs[0].up == [2 for i in range(1, 7)]


def test_init_branch_old3(main_sys):

    _, _, varis = main_sys
    rules = [{'e2': 2, 'e5': 2}, {x: 0 for x in varis.keys()}]
    rules_st = ['s', 'f']
    brs = gen_bnb.init_branch_old(varis, rules, rules_st)

    assert len(brs) == 1
    assert brs[0].up_state == 's'
    assert brs[0].down_state == 'f'
    assert brs[0].down == [0 for i in range(1, 7)]
    assert brs[0].up == [2 for i in range(1, 7)]


def test_init_branch_old4(main_sys):

    _, _, varis = main_sys
    rules = [{'e2': 2, 'e5': 2}, {x: 0 for x in varis.keys()}, {'e2': 2, 'e6': 2, 'e4': 2}]
    rules_st = ['s', 'f', 's']
    brs = gen_bnb.init_branch_old(varis, rules, rules_st)

    assert len(brs) == 1
    assert brs[0].up_state == 's'
    assert brs[0].down_state == 'f'
    assert brs[0].down == [0 for i in range(1, 7)]
    assert brs[0].up == [2 for i in range(1, 7)]


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


def test_init_branch1(main_sys):

    _, _, varis = main_sys
    rules = []

    brs = gen_bnb.init_branch(varis, rules)

    assert len(brs) == 1
    assert brs[0].up_state == 'u'
    assert brs[0].down_state == 'u'
    assert brs[0].down == {f'e{i}': 0 for i in range(1, 7)}
    assert brs[0].up == {f'e{i}': 2 for i in range(1, 7)}


def test_init_branch2(main_sys):

    _, _, varis = main_sys
    rules = [({'e2': 2, 'e5': 2}, 's')]

    brs = gen_bnb.init_branch(varis, rules)

    assert len(brs) == 1
    assert brs[0].up_state == 's'
    assert brs[0].down_state == 'u'
    assert brs[0].down == {f'e{i}': 0 for i in range(1, 7)}
    assert brs[0].up == {f'e{i}': 2 for i in range(1, 7)}


def test_init_branch3(main_sys):

    _, _, varis = main_sys
    rules = [({'e2': 2, 'e5': 2}, 's'), ({x: 0 for x in varis.keys()}, 'f')]

    brs = gen_bnb.init_branch(varis, rules)

    assert len(brs) == 1
    assert brs[0].up_state == 's'
    assert brs[0].down_state == 'f'
    assert brs[0].down == {f'e{i}': 0 for i in range(1, 7)}
    assert brs[0].up == {f'e{i}': 2 for i in range(1, 7)}


def test_init_branch4(main_sys):

    _, _, varis = main_sys
    rules = [({'e2': 2, 'e5': 2}, 's'), ({x: 0 for x in varis.keys()}, 'f'), ({'e2': 2, 'e6': 2, 'e4': 2}, 's')]

    brs = gen_bnb.init_branch(varis, rules)

    assert len(brs) == 1
    assert brs[0].up_state == 's'
    assert brs[0].down_state == 'f'
    assert brs[0].down == {f'e{i}': 0 for i in range(1, 7)}
    assert brs[0].up == {f'e{i}': 2 for i in range(1, 7)}


def test_core1(main_sys):

    _, _, varis = main_sys
    rules = []
    rules_st = []
    cst = []
    stop_br = False
    brs = gen_bnb.init_branch_old(varis, rules, rules_st)

    brs, cst, stop_br = gen_bnb.core(brs, rules, rules_st, cst, stop_br)

    assert len(brs) == 1
    assert brs[0].up == [2] * 6
    assert brs[0].down == [0] * 6
    assert cst == [2, 2, 2, 2, 2, 2]
    assert stop_br == True


def test_core2(main_sys):

    od_pair, arcs, varis = main_sys

    cst = [2, 2, 2, 2, 2, 2]
    stop_br = True
    rules = [{'e2': 2, 'e5': 2}]
    rules_st = ['s']
    brs = gen_bnb.init_branch_old(varis, rules, rules_st)

    #pdb.set_trace()
    brs, cst, stop_br = gen_bnb.core(brs, rules, rules_st, cst, stop_br)
    assert len(brs) == 1
    assert brs[0].up == [2] * 6
    assert brs[0].down == [0] * 6
    assert cst == [0, 0, 0, 0, 0, 0]
    assert stop_br == True


def test_core3(main_sys):

    od_pair, arcs, varis = main_sys

    cst = [0, 0, 0, 0, 0, 0]
    stop_br = True
    rules = [{'e2': 2, 'e5': 2}, {x: 0 for x in varis.keys()}]
    rules_st = ['s', 'f']
    brs = gen_bnb.init_branch_old(varis, rules, rules_st)

    #pdb.set_trace()
    brs, cst, stop_br = gen_bnb.core(brs, rules, rules_st, cst, stop_br)
    assert len(brs) == 1
    assert brs[0].up == [2] * 6
    assert brs[0].down == [0] * 6
    assert cst == [2, 2, 2, 2, 1, 2]
    assert stop_br == True


def test_core4(main_sys):

    od_pair, arcs, varis = main_sys

    cst = [2, 2, 2, 2, 1, 2]
    stop_br = True
    rules = [{'e2': 2, 'e5': 2}, {x: 0 for x in varis.keys()}, {'e2': 2, 'e6': 2, 'e4': 2}]
    rules_st = ['s', 'f', 's']
    brs = gen_bnb.init_branch_old(varis, rules, rules_st)

    #pdb.set_trace()
    brs, cst, stop_br = gen_bnb.core(brs, rules, rules_st, cst, stop_br)
    assert len(brs) == 1
    assert brs[0].up == [2] * 6
    assert brs[0].down == [0] * 6
    assert cst == [2, 1, 2, 2, 2, 2]
    assert stop_br == True


def test_do_gen_bnb2(main_sys):
    # no_iter: 10
    od_pair, arcs, varis = main_sys

    rules = [{'e2': 2, 'e6': 2, 'e4': 2}, {'e2': 1, 'e5': 2}, {'e1': 2, 'e3': 2, 'e5': 2}, {'e1': 0, 'e2': 1, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, {'e2': 0, 'e3': 1}, {'e1': 0, 'e2': 0, 'e4': 0, 'e5': 0, 'e6': 0}]
    rules_st = ['s', 's', 's', 'f', 'f', 'f']
    cst = [0, 0, 2, 0, 0, 0]
    thres = 2 * 0.1844

    sys_fun = trans.sys_fun_wrap(od_pair, arcs, varis, thres)

    brs = gen_bnb.init_branch_old(varis, rules, rules_st)
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


def test_do_gen_bnb1(main_sys):
    # iteration 11
    od_pair, arcs, varis = main_sys

    rules = [{'e2': 2, 'e6': 2, 'e4': 2}, {'e2': 1, 'e5': 2}, {'e1': 2, 'e3': 2, 'e5': 2}, {'e1': 0, 'e2': 1, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, {'e2': 0, 'e3': 1}, {'e2': 0, 'e5': 1}]
    rules_st = ['s', 's', 's', 'f', 'f', 'f']
    cst = [2, 0, 2, 2, 1, 2]
    thres = 2 * 0.1844

    sys_fun = trans.sys_fun_wrap(od_pair, arcs, varis, thres)

    brs = gen_bnb.init_branch_old(varis, rules, rules_st)
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
    rules_st = ['s', 'f', 'f', 's', 's', 'f', 'f', 's', 'f']
    cst = [2, 2, 2, 2, 0, 0]
    thres = 2 * 0.1844

    sys_fun = trans.sys_fun_wrap(od_pair, arcs, varis, thres)

    brs = gen_bnb.init_branch_old(varis, rules, rules_st)
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


def test_proposed_branch_and_bound(main_sys):
    # Branch and bound

    od_pair, arcs, varis = main_sys

    comps_name = list(arcs.keys())

    # Intact state of component vector: zero-based index 
    comps_st_itc = {k: v.B.shape[1] - 1 for k, v in varis.items()} # intact state (i.e. the highest state)
    d_time_itc, path_itc = trans.get_time_and_path_given_comps(comps_st_itc, od_pair, arcs, varis)

    # defines the system failure event
    thres = 2

    # Given a system function, i.e. sf_min_path, it should be represented by a function that only has "comps_st" as input.
    sys_fun = trans.sys_fun_wrap(od_pair, arcs, varis, thres * d_time_itc)

    # Branch and bound
    output_path = Path(__file__).parent
    #pdb.set_trace()
    brs, rules, sys_res = gen_bnb.proposed_branch_and_bound(sys_fun, varis, max_br=1000,
                                                              output_path=output_path, key='bridge', flag=True)

    #print(brs)
    # Result
    #assert no_sf == 23
    assert len(rules) == 10
    expected_rules =[{'e1': 2, 'e3': 2, 'e5': 2}, {'e2': 0, 'e3': 1}, {'e2': 0, 'e5': 1}, {'e2': 1, 'e6': 2, 'e4': 2}, {'e2': 1, 'e5': 1}, {'e4': 1, 'e5': 0}, {'e1': 1, 'e2': 0}, {'e2': 2, 'e6': 1, 'e4': 2}, {'e5': 0, 'e6': 0}, {'e2': 1, 'e5': 0, 'e6': 1}]
    expected_st = ['s', 'f', 'f', 's', 's', 'f', 'f', 's', 'f', 'f']
    assert all([item in rules for item in zip(expected_rules, expected_st)])

    assert len(brs) == 10
    expected_brs = [branch.Branch_old(down=[1, 1, 1, 1, 1, 1], up=[3, 1, 2, 3, 3, 3], is_complete=True, down_state='f', up_state='f', down_val=None, up_val=None, names=comps_name),
                    branch.Branch_old(down=[1, 1, 3, 1, 1, 1], up=[3, 1, 3, 3, 2, 3], is_complete=True, down_state='f', up_state='f', down_val=None, up_val=None, names=comps_name),
                    branch.Branch_old(down=[1, 1, 3, 1, 3, 1], up=[2, 1, 3, 3, 3, 3], is_complete=True, down_state='f', up_state='f', down_val=None, up_val=None, names=comps_name),
                    branch.Branch_old(down=[3, 1, 3, 1, 3, 1], up=[3, 1, 3, 3, 3, 3], is_complete=True, down_state='s', up_state='s', down_val=None, up_val=None, names=comps_name),
                    branch.Branch_old(down=[1, 2, 1, 1, 1, 1], up=[3, 3, 3, 2, 1, 3], is_complete=True, down_state='f', up_state='f', down_val=None, up_val=None, names=comps_name),
                    branch.Branch_old(down=[1, 2, 1, 3, 1, 1], up=[3, 3, 3, 3, 1, 1], is_complete=True, down_state='f', up_state='f', down_val=None, up_val=None, names=comps_name),
                    branch.Branch_old(down=[1, 2, 1, 3, 1, 2], up=[3, 2, 3, 3, 1, 2], is_complete=True, down_state='f', up_state='f', down_val=None, up_val=None, names=comps_name),
                    branch.Branch_old(down=[1, 3, 1, 3, 1, 2], up=[3, 3, 3, 3, 1, 2], is_complete=True, down_state='s', up_state='s', down_val=None, up_val=None, names=comps_name),
                    branch.Branch_old(down=[1, 2, 1, 3, 1, 3], up=[3, 3, 3, 3, 1, 3], is_complete=True, down_state='s', up_state='s', down_val=None, up_val=None, names=comps_name),
                    branch.Branch_old(down=[1, 2, 1, 1, 2, 1], up=[3, 3, 3, 3, 3, 3], is_complete=True, down_state='s', up_state='s', down_val=None, up_val=None, names=comps_name)]

    i = 0
    for br in expected_brs:

        br_ = branch.Branch({k: v-1 for k, v in zip(br.names, br.down)},
                            {k: v-1 for k, v in zip(br.names, br.up)},
                            br.down_state,
                            br.up_state)
        if br_ in brs:
            brs.remove(br_)
            print(f'removed: {i}')
            i += 1
        else:
            print(f'not found: {br_}')

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

    #np.testing.assert_array_almost_equal(sys_res['sys_val'].values, expected)
    expected_comps_min = [{'e2': 2, 'e5': 2},
                          {},
                          {'e2': 2, 'e6': 2, 'e4': 2},
                          {'e2': 1, 'e5': 2},
                          {'e1': 2, 'e3': 2, 'e5': 2},
                          {},
                          {},
                          {},
                          {},
                          {},
                          {},
                          {'e2': 2, 'e5': 1},
                          {'e2': 1, 'e6': 2, 'e4': 2},
                          {},
                          {'e2': 1, 'e5': 1},
                          {},
                          {},
                          {},
                          {'e2': 2, 'e6': 1, 'e4': 2},
                          {},
                          {},
                          {},
                          {}]

    #assert all([x == y for x, y in zip(expected_comps_min, sys_res['comps_st_min'].values)])


def test_do_gen_bnb0(setup_brs):
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

    # # Branch and bound
    no_sf, rules, rules_st, brs, sys_res = setup_brs

    # Result
    assert no_sf == 23
    assert len(rules) == 10
    assert len(brs) == 10

    expected_rules =[{'e1': 2, 'e3': 2, 'e5': 2}, {'e2': 0, 'e3': 1}, {'e2': 0, 'e5': 1}, {'e2': 1, 'e6': 2, 'e4': 2}, {'e2': 1, 'e5': 1}, {'e4': 1, 'e5': 0}, {'e1': 1, 'e2': 0}, {'e2': 2, 'e6': 1, 'e4': 2}, {'e5': 0, 'e6': 0}, {'e2': 1, 'e5': 0, 'e6': 1}]
    assert rules == expected_rules

    expected_rules_st = ['s', 'f', 'f', 's', 's', 'f', 'f', 's', 'f', 'f']
    assert rules_st == expected_rules_st
    """
    expected_brs = [branch.Branch(down=[1, 1, 1, 1, 1, 1], up=[3, 1, 2, 3, 3, 3], is_complete=True, down_state='f', up_state='f', down_val=None, up_val=None, names=comps_name),
                    branch.Branch(down=[1, 1, 3, 1, 1, 1], up=[3, 1, 3, 3, 2, 3], is_complete=True, down_state='f', up_state='f', down_val=None, up_val=None, names=comps_name),
                    branch.Branch(down=[1, 1, 3, 1, 3, 1], up=[2, 1, 3, 3, 3, 3], is_complete=True, down_state='f', up_state='f', down_val=None, up_val=None, names=comps_name),
                    branch.Branch(down=[3, 1, 3, 1, 3, 1], up=[3, 1, 3, 3, 3, 3], is_complete=True, down_state='s', up_state='s', down_val=None, up_val=None, names=comps_name),
                    branch.Branch(down=[1, 2, 1, 1, 1, 1], up=[3, 3, 3, 2, 1, 3], is_complete=True, down_state='f', up_state='f', down_val=None, up_val=None, names=comps_name),
                    branch.Branch(down=[1, 2, 1, 3, 1, 1], up=[3, 3, 3, 3, 1, 1], is_complete=True, down_state='f', up_state='f', down_val=None, up_val=None, names=comps_name),
                    branch.Branch(down=[1, 2, 1, 3, 1, 2], up=[3, 2, 3, 3, 1, 2], is_complete=True, down_state='f', up_state='f', down_val=None, up_val=None, names=comps_name),
                    branch.Branch(down=[1, 3, 1, 3, 1, 2], up=[3, 3, 3, 3, 1, 2], is_complete=True, down_state='s', up_state='s', down_val=None, up_val=None, names=comps_name),
                    branch.Branch(down=[1, 2, 1, 3, 1, 3], up=[3, 3, 3, 3, 1, 3], is_complete=True, down_state='s', up_state='s', down_val=None, up_val=None, names=comps_name),
                    branch.Branch(down=[1, 2, 1, 1, 2, 1], up=[3, 3, 3, 3, 3, 3], is_complete=True, down_state='s', up_state='s', down_val=None, up_val=None, names=comps_name)]
    assert all([x == y for x, y in zip(expected_brs, brs)])
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

    expected_comps_min = [{'e2': 2, 'e5': 2},
                          {},
                          {'e2': 2, 'e6': 2, 'e4': 2},
                          {'e2': 1, 'e5': 2},
                          {'e1': 2, 'e3': 2, 'e5': 2},
                          {},
                          {},
                          {},
                          {},
                          {},
                          {},
                          {'e2': 2, 'e5': 1},
                          {'e2': 1, 'e6': 2, 'e4': 2},
                          {},
                          {'e2': 1, 'e5': 1},
                          {},
                          {},
                          {},
                          {'e2': 2, 'e6': 1, 'e4': 2},
                          {},
                          {},
                          {},
                          {}]

    assert all([x == y for x, y in zip(expected_comps_min, sys_res['comps_st_min'].values)])


def test_get_state0():

    rules = []

    cst = {f'e{i}': 2 for i in range(1, 7)}
    result = gen_bnb.get_state(cst, rules)

    assert result == 'u'

    cst = {f'e{i}': 0 for i in range(1, 7)}
    result = gen_bnb.get_state(cst, rules)

    assert result == 'u'


def test_get_compat_rules2_0():

    upper = {f'e{i}': 2 for i in range(1, 7)}
    lower = {f'e{i}': 0 for i in range(1, 7)}

    rules = []
    result = gen_bnb.get_compat_rules2(lower, upper, rules)
    assert result == []

    rules = [({'e2': 2, 'e5': 2}, 's')]
    result = gen_bnb.get_compat_rules2(lower, upper, rules)
    assert result == [({'e2': 2, 'e5': 2}, 's')]

    rules = [({'e2': 1, 'e5': 2}, 's'), ({f'e{i}': 0 for i in range(1, 7)}, 'f')]
    result = gen_bnb.get_compat_rules2(lower, upper, rules)

    assert result[0] == rules[0]
    assert result[1] == rules[1]

    upper = {'e1': 2, 'e2': 0, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    lower = {f'e{i}': 0 for i in range(1, 7)}
    #pdb.set_trace()
    result = gen_bnb.get_compat_rules2(lower, upper, rules)
    assert len(result) == 1
    # FIXME: note that 'e2' was removed from rules
    assert result[0] == ({'e1': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, 'f')
    """
    upper = {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    lower = {'e1': 2, 'e2': 0, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    rules = [({'e2': 1, 'e5': 2}, 's'),
             ({f'e{i}': 0 for i in range(1, 7)}, 'f'),
             ({'e1': 2, 'e3': 2, 'e5': 2}, 's'),
             ({'e2': 2, 'e4': 2, 'e6': 2}, 's'),
            ]
    result = gen_bnb.get_compat_rules2(lower, upper, rules)
    print(result)
    assert result[0] == rules[0]
    assert result[1] == rules[2]
    assert result[2] == rules[3]
    """

def test_get_compat_rules0():

    rules = []

    upper = {f'e{i}': 2 for i in range(1, 7)}
    lower = {f'e{i}': 0 for i in range(1, 7)}

    result = gen_bnb.get_compat_rules(lower, upper, rules)
    assert result == []


def test_get_compat_rules_old0():

    rules = []
    rules_st = []

    cst = {f'e{i}': 2 for i in range(1, 7)}
    result = gen_bnb.get_compat_rules_old(cst, rules, rules_st)
    assert result == ([], 'u')

    cst = {f'e{i}': 0 for i in range(1, 7)}
    result = gen_bnb.get_compat_rules_old(cst, rules, rules_st)
    assert result == ([], 'u')


def test_get_state1():

    rules = [({'e2': 2, 'e5': 2}, 's')]

    cst = {f'e{i}': 2 for i in range(1, 7)}
    result = gen_bnb.get_state(cst, rules)

    assert result == 's'

    cst = {f'e{i}': 0 for i in range(1, 7)}
    result = gen_bnb.get_state(cst, rules)

    assert result == 'u'




def test_get_compat_rules1():

    rules = [({'e2': 2, 'e5': 2}, 's')]

    upper = {f'e{i}': 2 for i in range(1, 7)}
    lower = {f'e{i}': 0 for i in range(1, 7)}
    result = gen_bnb.get_compat_rules(lower, upper, rules)

    assert result == [({'e2': 2, 'e5': 2}, 's')]


def test_get_compat_rules_old1():

    rules = [{'e2': 2, 'e5': 2}]
    rules_st = ['s']

    cst = {f'e{i}': 2 for i in range(1, 7)}
    result = gen_bnb.get_compat_rules_old(cst, rules, rules_st)
    assert result == ([0], 's')

    cst = {f'e{i}': 0 for i in range(1, 7)}
    result = gen_bnb.get_compat_rules_old(cst, rules, rules_st)
    assert result == ([], 'u')


def test_get_state2():

    rules = [({'e2': 1, 'e5': 2}, 's')]
    rules_st = ['s']

    cst = {f'e{i}': 2 for i in range(1, 7)}
    result = gen_bnb.get_state(cst, rules)

    assert result == 's'

    cst = {f'e{i}': 0 for i in range(1, 7)}
    result = gen_bnb.get_state(cst, rules)

    assert result == 'u'


def test_get_compat_rules2():

    rules = [({'e2': 1, 'e5': 2}, 's')]

    upper = {f'e{i}': 2 for i in range(1, 7)}
    lower = {f'e{i}': 0 for i in range(1, 7)}
    result = gen_bnb.get_compat_rules(lower, upper, rules)

    assert result ==  [({'e2': 1, 'e5': 2}, 's')]


def test_get_compat_rules_old2():

    rules = [{'e2': 1, 'e5': 2}]
    rules_st = ['s']

    cst = {f'e{i}': 2 for i in range(1, 7)}
    result = gen_bnb.get_compat_rules_old(cst, rules, rules_st)

    assert result == ([0], 's')

    cst = {f'e{i}': 0 for i in range(1, 7)}
    result = gen_bnb.get_compat_rules_old(cst, rules, rules_st)

    assert result == ([], 'u')


def test_get_state3():

    cst = {f'e{i}': 2 for i in range(1, 7)}
    rules = [({'e2': 1, 'e5': 2}, 's'), ({'e1': 1, 'e2': 0}, 'f')]

    result = gen_bnb.get_state(cst, rules)

    assert result == 's'

    cst = {f'e{i}': 0 for i in range(1, 7)}

    result = gen_bnb.get_state(cst, rules)

    assert result == 'f'


def test_get_compat_rules3():

    upper = {f'e{i}': 2 for i in range(1, 7)}
    lower = {f'e{i}': 0 for i in range(1, 7)}
    rules = [({'e2': 1, 'e5': 2}, 's'), ({'e1': 1, 'e2': 0}, 'f')]

    result = gen_bnb.get_compat_rules(lower, upper, rules)

    assert result[0] == rules[0]
    assert result[1] == rules[1]


def test_get_compat_rules_old3():

    rules = [{'e2': 1, 'e5': 2}, {'e1': 1, 'e2': 0}]
    rules_st = ['s', 'f']

    cst = {f'e{i}': 2 for i in range(1, 7)}
    result = gen_bnb.get_compat_rules_old(cst, rules, rules_st)
    assert result == ([0], 's')

    cst = {f'e{i}': 0 for i in range(1, 7)}
    result = gen_bnb.get_compat_rules_old(cst, rules, rules_st)
    assert result == ([1], 'f')


def test_get_state4():

    cst = {'e1': 0, 'e2': 0, 'e3': 2 , 'e4': 1, 'e5': 0, 'e6': 2}
    rules = [({'e2': 1, 'e5': 2}, 's'), ({'e1': 1, 'e2': 0}, 'f')]

    result = gen_bnb.get_state(cst, rules)

    assert result == 'f'


def test_get_compat_rules4():

    lower = {'e1': 0, 'e2': 0, 'e3': 2 , 'e4': 1, 'e5': 0, 'e6': 2}
    upper = {'e1': 0, 'e2': 0, 'e3': 2 , 'e4': 1, 'e5': 0, 'e6': 2}
    rules = [({'e2': 1, 'e5': 2}, 's'), ({'e1': 1, 'e2': 0}, 'f')]

    result = gen_bnb.get_compat_rules(lower, upper, rules)

    assert result == [({'e1': 1, 'e2': 0}, 'f')]


def test_get_compat_rules_old4():

    cst = {'e1': 0, 'e2': 0, 'e3': 2 , 'e4': 1, 'e5': 0, 'e6': 2}
    rules = [{'e2': 1, 'e5': 2}, {'e1': 1, 'e2': 0}]
    rules_st = ['s', 'f']

    result = gen_bnb.get_compat_rules_old(cst, rules, rules_st)
    assert result == ([1], 'f')


def test_update_rule_set0():

    rules = []
    rule_new = ({'e2':2, 'e5':2}, 's')

    result = gen_bnb.update_rule_set(rules, rule_new)
    assert result == [({'e2': 2, 'e5': 2}, 's')]


def test_update_rule_set1():
    rules = [({'e2':2, 'e5':2}, 's')]
    rule_new = {f'e{i}':0 for i in range(1, 7)}, 'f'

    result = gen_bnb.update_rule_set(rules, rule_new)

    assert result == [({'e2': 2, 'e5': 2}, 's'), ({'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5':0, 'e6': 0}, 'f')]



def test_add_rule1():
    rules = [{'e2':2, 'e5':2}]
    rules_st = ['s']
    rule_new = {f'e{i}':0 for i in range(1, 7)}
    fail_or_surv = 'f'

    result = gen_bnb.add_rule(rules, rules_st, rule_new, fail_or_surv)

    assert result[0] == [{'e2': 2, 'e5': 2}, {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5':0, 'e6': 0}]
    assert result[1] == ['s', 'f']


def test_update_rule_set2():
    rules = [({'e2':2, 'e5':2}, 's'), ({'e2': 1, 'e5': 1}, 'f')]
    rule_new = {'e2': 1, 'e5': 2}, 's'

    result = gen_bnb.update_rule_set(rules, rule_new)

    assert result == [({'e2': 1, 'e5': 1}, 'f'), ({'e2': 1, 'e5': 2}, 's')]

    rule_new = {'e2': 1, 'e5': 0}, 'f'

    result = gen_bnb.update_rule_set(rules, rule_new)

    assert result == [({'e2': 1, 'e5': 1}, 'f'), ({'e2': 1, 'e5': 2}, 's')]



def test_add_rule2():
    rules = [{'e2':2, 'e5':2}, {'e2': 1, 'e5': 1}]
    rules_st = ['s', 'f']
    rule_new = {'e2': 1, 'e5': 2}
    fail_or_surv = 's'

    result = gen_bnb.add_rule(rules, rules_st, rule_new, fail_or_surv)

    assert result[0] == [{'e2': 1, 'e5': 1}, {'e2': 1, 'e5': 2}]
    assert result[1] == ['f', 's']

    rule_new = {'e2': 1, 'e5': 0}
    fail_or_surv = 'f'

    result = gen_bnb.add_rule(rules, rules_st, rule_new, fail_or_surv)

    assert result[0] == [{'e2': 1, 'e5': 1}, {'e2': 1, 'e5': 2}]
    assert result[1] == ['f', 's']


def test_get_decomp_comp2_0():

    rules = [({'e2': 2, 'e5': 2}, 's'),
             ({'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, 'f')]
    upper = {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    lower = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}
    result = gen_bnb.get_decomp_comp2(lower, upper, rules)

    assert result == ('e2', 2)

    rules = [({'e2': 2, 'e5': 2}, 's'),
             ({'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, 'f'),
             ({'e1': 2, 'e3': 2, 'e5': 2}, 's')]
    upper = {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    lower = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}
    result = gen_bnb.get_decomp_comp2(lower, upper, rules)

    assert result == ('e5', 2)

    rules = [({'e2': 2, 'e5': 2}, 's'),
             ({'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, 'f'),
             ({'e1': 2, 'e3': 2, 'e5': 2}, 's'),
             ({'e2': 2, 'e4': 2, 'e6': 2}, 's')]
    upper = {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    lower = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}
    result = gen_bnb.get_decomp_comp2(lower, upper, rules)

    assert result == ('e2', 2)

    rules = [({'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, 'f'),
             ({'e1': 2, 'e3': 2, 'e5': 2}, 's')]
    upper = {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    lower = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}
    result = gen_bnb.get_decomp_comp2(lower, upper, rules)

    assert result == ('e1', 2)


def test_get_decomp_comp0():

    rules = [({'e2': 2, 'e5': 2}, 's'),
             ({'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, 'f')]
    upper = {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    lower = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}
    result = gen_bnb.get_decomp_comp(lower, upper, rules)

    assert result == ('e2', 1)


def test_get_decomp_comp1():

    rules = [({'e2': 2, 'e5': 2}, 's'),
             ({'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, 'f'),
             ({'e1': 2, 'e3': 2, 'e5': 2}, 's')]
    upper = {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    lower = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}
    result = gen_bnb.get_decomp_comp(lower, upper, rules)

    assert result == ('e5', 2)


def test_get_comp_st_for_next_bnb0():
    up = {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    down = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}
    rules = [{'e2': 2, 'e5': 2},
             {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}]
    rules_st = ['s', 'f']
    #pdb.set_trace()
    result = gen_bnb.get_comp_st_for_next_bnb(up, down, rules, rules_st)

    assert result[0] == 'e5'
    assert result[1] == 2


def test_get_decomp_comp2():
    up = {'e1': 2, 'e2': 0, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    down = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}

    rules = [({'e2': 2, 'e6': 2, 'e4': 2}, 's'),
             ({'e2': 1, 'e5': 2}, 's'),
             ({'e1': 2, 'e3': 2, 'e5': 2}, 's'),
             ({'e1': 0, 'e2': 1, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, 'f'),
             ({'e2': 0, 'e3': 1}, 'f'),
             ({'e2': 0, 'e5': 1}, 'f')]

    result = gen_bnb.get_decomp_comp(down, up, rules)

    assert result == ('e3', 2) or result == ('e5', 2)


#@pytest.mark.skip('NYI')
def test_get_comp_st_for_next_bnb2():
    up = {'e1': 2, 'e2': 0, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    down = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}
    rules = [{'e2': 2, 'e6': 2, 'e4': 2}, {'e2': 1, 'e5': 2}, {'e1': 2, 'e3': 2, 'e5': 2}, {'e1': 0, 'e2': 1, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, {'e2': 0, 'e3': 1}, {'e2': 0, 'e5': 1}]
    rules_st = ['s', 's', 's', 'f', 'f', 'f']

    #pdb.set_trace()
    result = gen_bnb.get_comp_st_for_next_bnb(up, down, rules, rules_st)

    assert result[0] == 'e3'
    assert result[1] == 2


def test_decomp_to_two_branches1():
    comps_name = ['e1', 'e2', 'e3', 'e4', 'e5' ,'e6']
    br = branch.Branch_old(down=[0, 0, 0, 0, 0, 0], up=[2, 2, 2, 2, 2, 2], is_complete=False, names=comps_name)
    br.down_state='f' # FIXME
    br.up_state='s' # FIXME
    comp_bnb = 'e5'
    st_bnb_up = 2

    result = gen_bnb.decomp_to_two_branches(br, comp_bnb, st_bnb_up)

    assert result[0] == branch.Branch_old(down=[0, 0, 0, 0, 0, 0], up=[2, 2, 2, 2, 1, 2], names=comps_name, is_complete=False, down_state=1, up_state=1)

    assert result[1] == branch.Branch_old(down=[0, 0, 0, 0, 2, 0], up=[2, 2, 2, 2, 2, 2], names=comps_name, is_complete=False, down_state=1, up_state=1)


    br.down_state='f' # FIXME
    br.up_state='f' # FIXME
    comp_bnb = 'e2'
    st_bnb_up = 1
    result = gen_bnb.decomp_to_two_branches(br, comp_bnb, st_bnb_up)

    assert result[0] == branch.Branch_old(down=[0, 0, 0, 0, 0, 0], up=[2, 0, 2, 2, 2, 2], names=comps_name, is_complete=False, down_state=1, up_state=1)

    assert result[1] == branch.Branch_old(down=[0, 1, 0, 0, 0, 0], up=[2, 2, 2, 2, 2, 2], names=comps_name, is_complete=False, down_state=1, up_state=1)


def test_decomp_to_two_branches2():
    comps_name = ['e1', 'e2', 'e3', 'e4', 'e5' ,'e6']
    br = branch.Branch_old(down=[0, 0, 0, 0, 0, 0], up=[2, 2, 2, 2, 2, 2], is_complete=False, names=comps_name)
    br.down_state='f' # FIXME
    br.up_state='s' # FIXME
    comp_bnb = 'e2'
    st_bnb_up = 1

    result = gen_bnb.decomp_to_two_branches(br, comp_bnb, st_bnb_up)

    assert result[0] == branch.Branch_old(down=[0, 0, 0, 0, 0, 0], up=[2, 0, 2, 2, 2, 2], names=comps_name, is_complete=False, down_state=1, up_state=1)

    assert result[1] == branch.Branch_old(down=[0, 1, 0, 0, 0, 0], up=[2, 2, 2, 2, 2, 2], names=comps_name, is_complete=False, down_state=1, up_state=1)


def test_run_sys_fn1(main_sys):

    od_pair, arcs, varis = main_sys

    thres = 2 * 0.1844
    sys_fun = trans.sys_fun_wrap(od_pair, arcs, varis, thres)

    cst = {f'e{i}': 2 for i in range(1, 7)}
    rules = []

    #pdb.set_trace()
    sys_res, rule = gen_bnb.run_sys_fn(cst, sys_fun, rules, varis)
    np.testing.assert_almost_equal(sys_res['sys_val'].values, np.array([0.18442]), decimal=5)
    assert sys_res['comp_st'].values == [{k: 2 for k in varis.keys()}]
    assert sys_res['comp_st_min'].values == [{'e2': 2, 'e5': 2}]
    assert rule == ({'e2': 2, 'e5': 2}, 's')


def test_run_sys_fn2(main_sys):

    od_pair, arcs, varis = main_sys

    thres = 2 * 0.1844
    sys_fun = trans.sys_fun_wrap(od_pair, arcs, varis, thres)

    cst = {f'e{i}': 0 for i in range(1, 7)}
    rules = [({'e2': 2, 'e5': 2}, 's')]

    #pdb.set_trace()
    sys_res, rule = gen_bnb.run_sys_fn(cst, sys_fun, rules, varis)

    np.testing.assert_almost_equal(sys_res['sys_val'].values, np.array([1.8442]), decimal=4)
    assert sys_res['comp_st'].values[0] == {k: 0 for k in varis.keys()}
    assert sys_res['comp_st_min'].values == [{}]
    assert rule == ({k: 0 for k in varis.keys()}, 'f')


def test_run_sys_fn3(main_sys):

    od_pair, arcs, varis = main_sys

    thres = 2 * 0.1844
    sys_fun = trans.sys_fun_wrap(od_pair, arcs, varis, thres)

    cst = {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 1, 'e6': 2}
    rules = [({'e2': 2, 'e5': 2}, 's'), ({k: 0 for k in varis.keys()}, 'f')]

    #pdb.set_trace()
    sys_res, rule = gen_bnb.run_sys_fn(cst, sys_fun, rules, varis)

    np.testing.assert_almost_equal(sys_res['sys_val'].values, np.array([0.266]), decimal=3)
    assert sys_res['comp_st'].values[0] == {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 1, 'e6': 2}
    assert sys_res['comp_st_min'].values == [{'e2': 2, 'e6': 2, 'e4': 2}]
    assert rule == ({'e2': 2, 'e6': 2, 'e4': 2}, 's')


def test_get_composite_state1(main_sys):

    od_pair, arcs, varis = main_sys

    states = [1, 2]
    result = variable.get_composite_state(varis['e1'], states)

    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]])
    np.testing.assert_array_equal(result[0].B, expected)

    assert result[1] == 3


def test_get_composite_state2(main_sys):

    #od_pair, arcs, varis = main_sys
    varis = {}
    varis['e1'] = variable.Variable(name='e1', B=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]), values=[1.5, 0.3, 0.15])

    states = [1, 2]
    result = variable.get_composite_state(varis['e1'], states)

    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]])
    np.testing.assert_array_equal(result[0].B, expected)

    assert result[1] == 3


def test_get_composite_state3(main_sys):

    od_pair, arcs, varis = main_sys

    states = [0, 2]
    result = variable.get_composite_state(varis['e1'], states)

    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]])
    np.testing.assert_array_equal(result[0].B, expected)

    assert result[1] == 3


def test_get_csys_from_brs2(main_sys):

    od_pair, arcs, varis = main_sys

    comps_name = list(arcs.keys())

    # Intact state of component vector: zero-based index 
    comps_st_itc = {k: v.B.shape[1] - 1 for k, v in varis.items()} # intact state (i.e. the highest state)
    d_time_itc, path_itc = trans.get_time_and_path_given_comps(comps_st_itc, od_pair, arcs, varis)

    # defines the system failure event
    thres = 2

    # Given a system function, i.e. sf_min_path, it should be represented by a function that only has "comps_st" as input.
    sys_fun = trans.sys_fun_wrap(od_pair, arcs, varis, thres * d_time_itc)

    # Branch and bound
    output_path = Path(__file__).parent
    #pdb.set_trace()
    brs, rules, sys_res = gen_bnb.proposed_branch_and_bound(sys_fun, varis, max_br=1000,
                                                              output_path=output_path, key='bridge', flag=True)


    st_br_to_cs = {'f': 0, 's': 1, 'u': 2}
    #pdb.set_trace()
    result = gen_bnb.get_csys_from_brs2(brs, varis, st_br_to_cs)
    #cmat = result[0]
    cmat = result[0][np.argsort(result[0][:, 0])]
    print(cmat)
    expected = np.array([[0, 4, 0, 4, 3, 3, 3],
                         [0, 4, 0, 2, 3, 4, 3],
                         [0, 3, 0, 2, 3, 2, 3],
                         [0, 4, 3, 3, 4, 0, 3],
                         [0, 4, 3, 3, 2, 0, 0],
                         [0, 4, 1, 3, 2, 0, 1],
                         [1, 2, 0, 2, 3, 2, 3],
                         [1, 4, 2, 3, 2, 0, 1],
                         [1, 4, 3, 3, 2, 0, 2],
                         [1, 4, 3, 3, 3, 5, 3]])
    # FIXME: not exactly matching
    #np.testing.assert_array_equal(cmat, expected)

    np.testing.assert_array_equal(result[1]['e1'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]]))
    np.testing.assert_array_equal(result[1]['e2'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]))
    np.testing.assert_array_equal(result[1]['e3'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0]]))
    np.testing.assert_array_equal(result[1]['e4'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0]]))
    np.testing.assert_array_equal(result[1]['e5'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1]]))
    np.testing.assert_array_equal(result[1]['e6'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0]]))


def test_get_cmat_from_br1(main_sys):

    od_pair, arcs, varis = main_sys

    st_br_to_cs = {'f': 0, 's': 1, 'u': 2}

    br = branch.Branch({'e1': 1, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0},
                       {'e1': 2, 'e2': 0, 'e3': 1, 'e4': 2, 'e5': 2, 'e6': 2}, 'f', 'f')
    #pdb.set_trace()
    expected = gen_bnb.get_cmat_from_br(br, varis, st_br_to_cs)

    np.testing.assert_array_equal(expected[1], [0, 3, 0, 3, 3, 3, 3])
    np.testing.assert_array_equal(expected[0]['e1'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]))
    np.testing.assert_array_equal(expected[0]['e2'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    np.testing.assert_array_equal(expected[0]['e3'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]))
    np.testing.assert_array_equal(expected[0]['e4'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]))
    np.testing.assert_array_equal(expected[0]['e5'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]))
    np.testing.assert_array_equal(expected[0]['e6'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]))

@pytest.mark.skip('NYI')
def test_get_cmat_from_br2(main_sys):

    od_pair, arcs, varis = main_sys

    st_br_to_cs = {'f': 0, 's': 1, 'u': 2}

    # using the previous output as an input
    br = branch.Branch_old(down=[0, 0, 2, 0, 0, 0], up=[2, 0, 2, 2, 1, 2], is_complete=True, names=['e1', 'e2', 'e3', 'e4', 'e5', 'e6'])
    br.down_state = 'f'
    br.up_state='f'
    expected = gen_bnb.get_cmat_from_br(br, expected[0], st_br_to_cs)

    np.testing.assert_array_equal(expected[1], [0, 3, 0, 2, 3, 4, 3])
    np.testing.assert_array_equal(expected[0]['e1'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]))
    np.testing.assert_array_equal(expected[0]['e2'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    np.testing.assert_array_equal(expected[0]['e3'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]))
    np.testing.assert_array_equal(expected[0]['e4'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]))
    np.testing.assert_array_equal(expected[0]['e5'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0]]))
    np.testing.assert_array_equal(expected[0]['e6'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]))

@pytest.mark.skip('FIXME')
def test_get_c_from_br1(main_sys):

    od_pair, arcs, varis = main_sys

    st_br_to_cs = {'f': 0, 's': 1, 'u': 2}

    br = branch.Branch_old(down=[1, 0, 0, 0, 0, 0], up=[2, 0, 1, 2, 2, 2], is_complete=True, names=['e1', 'e2', 'e3', 'e4', 'e5', 'e6'])
    br.down_state = 'f'
    br.up_state='f'
    expected = gen_bnb.get_c_from_br(br, varis, st_br_to_cs)

    np.testing.assert_array_equal(expected[1], [0, 3, 0, 3, 3, 3, 3])
    np.testing.assert_array_equal(expected[0]['e1'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]))
    np.testing.assert_array_equal(expected[0]['e2'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    np.testing.assert_array_equal(expected[0]['e3'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]))
    np.testing.assert_array_equal(expected[0]['e4'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]))
    np.testing.assert_array_equal(expected[0]['e5'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]))
    np.testing.assert_array_equal(expected[0]['e6'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]))

    # using the previous output as an input
    br = branch.Branch_old(down=[0, 0, 2, 0, 0, 0], up=[2, 0, 2, 2, 1, 2], is_complete=True, names=['e1', 'e2', 'e3', 'e4', 'e5', 'e6'])
    br.down_state = 'f'
    br.up_state='f'
    expected = gen_bnb.get_c_from_br(br, expected[0], st_br_to_cs)

    np.testing.assert_array_equal(expected[1], [0, 3, 0, 2, 3, 4, 3])
    np.testing.assert_array_equal(expected[0]['e1'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]))
    np.testing.assert_array_equal(expected[0]['e2'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    np.testing.assert_array_equal(expected[0]['e3'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]))
    np.testing.assert_array_equal(expected[0]['e4'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]))
    np.testing.assert_array_equal(expected[0]['e5'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0]]))
    np.testing.assert_array_equal(expected[0]['e6'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]))


def test_get_csys_from_brs(main_sys, request):

    od_pair, arcs, varis = main_sys

    comps_name = list(arcs.keys())

    # Intact state of component vector: zero-based index 
    comps_st_itc = {k: v.B.shape[1] - 1 for k, v in varis.items()} # intact state (i.e. the highest state)
    d_time_itc, path_itc = trans.get_time_and_path_given_comps(comps_st_itc, od_pair, arcs, varis)

    # defines the system failure event
    thres = 2

    # Given a system function, i.e. sf_min_path, it should be represented by a function that only has "comps_st" as input.
    sys_fun = trans.sys_fun_wrap(od_pair, arcs, varis, thres * d_time_itc)

    # # Branch and bound
    file_brs = Path(__file__).parent.joinpath('brs_bridge.pk')
    if file_brs.exists():
        with open(file_brs, 'rb') as fi:
            brs = pickle.load(fi)
            print(f'{file_brs} loaded')
    else:
        _, _, _, brs, _ = request.getfixturevalue('setup_brs')

    st_br_to_cs = {'f': 0, 's': 1, 'u': 2}

    result = gen_bnb.get_csys_from_brs(brs, varis, st_br_to_cs)

    expected = np.array([[0, 3, 0, 3, 3, 3, 3],
                         [0, 3, 0, 2, 3, 4, 3],
                         [0, 4, 0, 2, 3, 2, 3],
                         [1, 2, 0, 2, 3, 2, 3],
                         [0, 3, 3, 4, 4, 0, 3],
                         [0, 3, 3, 4, 2, 0, 0],
                         [0, 3, 1, 4, 2, 0, 1],
                         [1, 3, 2, 4, 2, 0, 1],
                         [1, 3, 3, 4, 2, 0, 2],
                         [1, 3, 3, 4, 3, 5, 3]])
    np.testing.assert_array_equal(result[0], expected)

    np.testing.assert_array_equal(result[1]['e1'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0]]))
    np.testing.assert_array_equal(result[1]['e2'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]))
    np.testing.assert_array_equal(result[1]['e3'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]]))
    np.testing.assert_array_equal(result[1]['e4'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0]]))
    np.testing.assert_array_equal(result[1]['e5'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1]]))
    np.testing.assert_array_equal(result[1]['e6'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]))


@pytest.fixture()
def setup_inference(main_sys, setup_comp_events, request):

    _, arcs, _ = main_sys

    cpms, varis = setup_comp_events

    file_brs = Path(__file__).parent.joinpath('brs_bridge.pk')
    if file_brs.exists():
        with open(file_brs, 'rb') as fi:
            brs = pickle.load(fi)
            print(f'{file_brs} loaded')
    else:
        _, _, _, brs, _ = request.getfixturevalue('setup_brs')

    st_br_to_cs = {'f': 0, 's': 1, 'u': 2}

    csys, varis = gen_bnb.get_csys_from_brs(brs, varis, st_br_to_cs)

    varis['sys'] = variable.Variable('sys', np.eye(3), ['f', 's', 'u'])
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


def test_inference_case1(setup_inference):

    cpms, varis, var_elim_order, _ = setup_inference

    Msys = cpm.variable_elim([cpms[v] for v in varis.keys()], var_elim_order )
    np.testing.assert_array_almost_equal(Msys.C, np.array([[0, 1]]).T)
    np.testing.assert_array_almost_equal(Msys.p, np.array([[0.1018, 0.8982]]).T)


def test_inference_case2(setup_inference):

    cpms, varis, var_elim_order, arcs = setup_inference

    cnd_vars = [f'o{i}' for i in range(1, len(arcs) + 1)]
    cnd_states = [1, 1, 0, 1, 0, 1]

    Mobs = cpm.condition([cpms[v] for v in varis.keys()], cnd_vars, cnd_states)
    Msys_obs = cpm.variable_elim(Mobs, var_elim_order)

    np.testing.assert_array_almost_equal(Msys_obs.C, np.array([[0, 1]]).T)
    np.testing.assert_array_almost_equal(Msys_obs.p, np.array([[2.765e-5, 5.515e-5]]).T)


