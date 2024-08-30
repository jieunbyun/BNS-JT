# coding: utf-8

import pytest
import copy
import time
import pdb
import pickle
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

from BNS_JT import trans, branch, variable, cpm, brc, operation


HOME = Path(__file__).parent


def compare_list_of_sets(a, b):

    return set([tuple(x) for x in a]) == set([tuple(x) for x in b])


@pytest.fixture(scope='package')
def main_sys(data_bridge):

    node_coords = data_bridge['node_coords']
    arcs = data_bridge['arcs']
    arcs_avg_kmh = data_bridge['arcs_avg_kmh']

    # od_pair is different from setup_bridge in test_trans.py
    od_pair = ('n1', 'n3')

    arc_lens_km = trans.get_arcs_length(arcs, node_coords)
    arc_times_h = {k: v/arcs_avg_kmh[k] for k, v in arc_lens_km.items()}

    G = nx.Graph()
    for k, v in arcs.items():
        G.add_edge(v[0], v[1], label=k, key=k, weight=arc_times_h[k])

    [G.add_node(f'n{k}') for k in range(1, 6)]

    # add weight
    for n1, n2, e in G.edges(data=True):
        G[n1][n2]['weight'] = arc_times_h[e['label']]

    # Component events
    #no_arc_st = 3 # number of component states 
    varis = {}
    delay_rat = [10, 2, 1] # delay in travel time given each component state (ratio)
    for k, v in arcs.items():
        varis[k] = variable.Variable(name=k, values = [arc_times_h[k]*np.float64(x) for x in delay_rat])

    return G, od_pair, arcs, varis


@pytest.fixture(scope='package')
def setup_brs(main_sys):

    G, od_pair, arcs, d_varis = main_sys

    varis = copy.deepcopy(d_varis)

    # Intact state of component vector: zero-based index 
    comps_st_itc = {k: len(v.values) - 1 for k, v in varis.items()} # intact state (i.e. the highest state)
    d_time_itc, path_n, path_itc = trans.get_time_and_path_given_comps(comps_st_itc, G, od_pair, varis)

    # defines the system failure event
    thres = 2
    # Given a system function, i.e. sf_min_path, it should be represented by a function that only has "comps_st" as input.
    sys_fun = trans.sys_fun_wrap(G, od_pair, varis, thres * d_time_itc)

    p = [1/3, 1/3, 1/3]
    probs = {k: p for k in varis.keys()}

    file_brs = Path(__file__).parent.joinpath('brs_bridge.pk')
    if file_brs.exists():
        with open(file_brs, 'rb') as fi:
            brs = pickle.load(fi)
            print(f'{file_brs} loaded')
    else:
        # Branch and bound
        output_path = Path(__file__).parent
        #pdb.set_trace()
        brs, rules, sys_res, monitor = brc.run(varis, probs, sys_fun, max_sf=1000, max_nb=100)

    return varis, brs


@pytest.mark.skip('NYI')
@pytest.fixture()
def main_sys_bridge(data_bridge):

    # Network
    _, arcs, arc_times_h = info_bridge

    # od_pair is different from setup_bridge in test_trans.py
    od_pair = ('n5', 'n1')

    #arc_lens_km = trans.get_arcs_length(arcs, node_coords)
    #arc_times_h = {k: v/arcs_avg_kmh[k] for k, v in arc_lens_km.items()}

    #comps_name = [k for k in arcs] # *FIXME* this is not necessary if branch's down and up are defined by dictionary (instead of list)

    # Component events
    #no_arc_st = 2 # number of component states 
    #delay_rat = [10, 2, 1] # delay in travel time given each component state (ratio)
    varis = {}
    for k, v in arcs.items():
        #varis[k] = variable.Variable(name=k, B=[{0}, {1}, {2}] , values = [arc_times_h[k]*np.float64(x) for x in delay_rat])
        varis[k] = variable.Variable(name=k, values = [10.0*arc_times_h[k], arc_times_h[k]])

    return od_pair, arcs, varis



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

    G, od_pair, arcs, varis = main_sys

    comps_st, expected = comps_st_dic

    for c in comps_st.keys():

        d_time, path, path_e = trans.get_time_and_path_given_comps(comps_st[c], G, od_pair, varis)

        assert pytest.approx(d_time, 0.001) == expected[c][0]
        assert path == expected[c][3]


def test_sf_min_path(main_sys, comps_st_dic):

    G, od_pair, arcs, varis = main_sys
    comps_st, expected = comps_st_dic
    thres = 2*0.1844

    for c in comps_st.keys():

        d_time, path_n, path_e = trans.get_time_and_path_given_comps(comps_st[c], G, od_pair, varis)

        result = trans.sf_min_path(comps_st[c], G, od_pair, varis, thres)

        assert pytest.approx(result[0], 0.001) == expected[c][0]
        assert result[1] == expected[c][1]
        assert result[2] == expected[c][2]


def test_init_branch1():

    rules = {'s':[], 'f': [], 'u': []}
    #worst = {f'e{i}': 0 for i in range(1, 7)}
    #best = {f'e{i}': 2 for i in range(1, 7)}

    varis = {f'e{i}': variable.Variable(name=f'e{i}', values=['1', '2', '3']) for i in range(1, 7)}
    brs = brc.init_branch(varis, rules)

    assert len(brs) == 1
    assert brs[0].up_state == 'u'
    assert brs[0].down_state == 'u'
    assert brs[0].down == {f'e{i}': 0 for i in range(1, 7)}
    assert brs[0].up == {f'e{i}': 2 for i in range(1, 7)}

    rules = {'s': [{'e2': 2, 'e5': 2}], 'f': [], 'u': []}
    brs = brc.init_branch(varis, rules)

    assert len(brs) == 1
    assert brs[0].up_state == 's'
    assert brs[0].down_state == 'u'
    assert brs[0].down == {f'e{i}': 0 for i in range(1, 7)}
    assert brs[0].up == {f'e{i}': 2 for i in range(1, 7)}

    rules = {'s': [{'e2': 2, 'e5': 2}], 'f': [{f'e{x}': 0 for x in range(1, 7)}], 'u': []}
    brs = brc.init_branch(varis, rules)

    assert len(brs) == 1
    assert brs[0].up_state == 's'
    assert brs[0].down_state == 'f'
    assert brs[0].down == {f'e{i}': 0 for i in range(1, 7)}
    assert brs[0].up == {f'e{i}': 2 for i in range(1, 7)}

    rules = {'s': [{'e2': 2, 'e5': 2}, {'e2': 2, 'e6': 2, 'e4': 2}], 'f': [{f'e{x}': 0 for x in range(1, 7)}], 'u': []}
    brs = brc.init_branch(varis, rules)

    assert len(brs) == 1
    assert brs[0].up_state == 's'
    assert brs[0].down_state == 'f'
    assert brs[0].down == {f'e{i}': 0 for i in range(1, 7)}
    assert brs[0].up == {f'e{i}': 2 for i in range(1, 7)}




def test_get_state0():

    rules = {'s': [], 'f': [], 'u': []}

    cst = {f'e{i}': 2 for i in range(1, 7)}
    result = brc.get_state(cst, rules)
    assert result == 'u'

    cst = {f'e{i}': 0 for i in range(1, 7)}
    result = brc.get_state(cst, rules)
    assert result == 'u'



def test_get_state1():

    rules = {'s': [{'e2': 2, 'e5': 2}], 'f': [], 'u': []}

    cst = {f'e{i}': 2 for i in range(1, 7)}
    result = brc.get_state(cst, rules)
    assert result == 's'

    cst = {f'e{i}': 0 for i in range(1, 7)}
    result = brc.get_state(cst, rules)
    assert result == 'u'


def test_get_state2():

    rules = {'s': [{'e2': 2, 'e5': 2}], 'f': [{'e1': 1, 'e2': 0}], 'u': []}

    cst = {f'e{i}': 2 for i in range(1, 7)}
    result = brc.get_state(cst, rules)
    assert result == 's'

    cst = {f'e{i}': 0 for i in range(1, 7)}
    result = brc.get_state(cst, rules)
    assert result == 'f'


def test_get_state4():

    rules = {'s': [{'e2': 1, 'e5': 2}], 'f': [{'e1': 1, 'e2': 0}], 'u': []}
    cst = {'e1': 0, 'e2': 0, 'e3': 2 , 'e4': 1, 'e5': 0, 'e6': 2}
    result = brc.get_state(cst, rules)
    assert result == 'f'


def test_update_rule_set0():

    rules = {'s': [], 'f': [], 'u': []}
    rule_new = ({'e2':2, 'e5':2}, 's')

    result = brc.update_rule_set(rules, rule_new)
    assert result == {'s': [{'e2': 2, 'e5': 2}], 'f': [], 'u': []}


def test_update_rule_set1():

    rules = {'s': [{'e2': 2, 'e5': 2}], 'f': [], 'u': []}
    rule_new = {f'e{i}':0 for i in range(1, 7)}, 'f'

    result = brc.update_rule_set(rules, rule_new)
    expected = {'s': [{'e2': 2, 'e5': 2}], 'f': [{'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5':0, 'e6': 0}], 'u': []}
    assert result == expected


def test_update_rule_set2():

    rules = {'s': [{'e2':2, 'e5':2}], 'u': [], 'f':[{'e2': 1, 'e5': 1}]}
    rule_new = {'e2': 1, 'e5': 2}, 's'

    result = brc.update_rule_set(rules, rule_new)
    expected = {'s': [{'e2':1, 'e5':2}], 'u': [], 'f':[{'e2': 1, 'e5': 1}]}
    assert result == expected

    rule_new = {'e2': 1, 'e5': 0}, 'f'

    result = brc.update_rule_set(rules, rule_new)
    expected = {'s': [{'e2':1, 'e5':2}], 'u': [], 'f':[{'e2': 1, 'e5': 1}]}
    assert result == expected


def test_update_rule_set3():

    rules = {'s': [{'e2':2, 'e5':2}, {'e2':3, 'e5':2}], 'u': [], 'f':[{'e2': 1, 'e5': 1}, {'e2': 0, 'e5': 1}]}
    rule_new = {'e2': 1, 'e5': 2}, 's'
    result = brc.update_rule_set(rules, rule_new)
    expected = {'s': [{'e2':1, 'e5':2}], 'u': [], 'f':[{'e2': 1, 'e5': 1}, {'e2': 0, 'e5': 1}]}
    assert result == expected

    rule_new = {'e2': 1, 'e5': 0}, 'f'

    result = brc.update_rule_set(rules, rule_new)
    expected = {'s': [{'e2':1, 'e5':2}], 'u': [], 'f':[{'e2': 1, 'e5': 1}, {'e2': 0, 'e5': 1}]}
    assert result == expected



def test_run_sys_fn1(main_sys):

    G, od_pair, arcs, varis = main_sys

    thres = 2 * 0.1844
    sys_fun = trans.sys_fun_wrap(G, od_pair, varis, thres)

    cst = {f'e{i}': 2 for i in range(1, 7)}
    rules = []

    rule, sys_res = brc.run_sys_fn(cst, sys_fun, varis)
    np.testing.assert_almost_equal(sys_res['sys_val'].values, np.array([0.18442]), decimal=5)
    assert sys_res['comp_st'].values == [{k: 2 for k in varis.keys()}]
    assert sys_res['comp_st_min'].values == [{'e2': 2, 'e5': 2}]
    assert rule == ({'e2': 2, 'e5': 2}, 's')


def test_run_sys_fn2(main_sys):

    G, od_pair, arcs, varis = main_sys

    thres = 2 * 0.1844
    sys_fun = trans.sys_fun_wrap(G, od_pair, varis, thres)

    cst = {f'e{i}': 0 for i in range(1, 7)}
    rules = [({'e2': 2, 'e5': 2}, 's')]

    rule, sys_res = brc.run_sys_fn(cst, sys_fun, varis)

    np.testing.assert_almost_equal(sys_res['sys_val'].values, np.array([1.8442]), decimal=4)
    assert sys_res['comp_st'].values[0] == {k: 0 for k in varis.keys()}
    assert sys_res['comp_st_min'].values == [{}]
    assert rule == ({k: 0 for k in varis.keys()}, 'f')


def test_run_sys_fn3(main_sys):

    G, od_pair, arcs, varis = main_sys

    thres = 2 * 0.1844
    sys_fun = trans.sys_fun_wrap(G, od_pair, varis, thres)

    cst = {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 1, 'e6': 2}
    rules = [({'e2': 2, 'e5': 2}, 's'), ({k: 0 for k in varis.keys()}, 'f')]

    rule, sys_res = brc.run_sys_fn(cst, sys_fun, varis)

    np.testing.assert_almost_equal(sys_res['sys_val'].values, np.array([0.266]), decimal=3)
    assert sys_res['comp_st'].values[0] == {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 1, 'e6': 2}
    assert sys_res['comp_st_min'].values == [{'e2': 2, 'e6': 2, 'e4': 2}]
    assert rule == ({'e2': 2, 'e6': 2, 'e4': 2}, 's')


def test_get_composite_state1():

    varis = {}
    varis['e1'] = variable.Variable(name='e1', values=[1.5, 0.3, 0.15])

    states = [1, 2]
    result = variable.get_composite_state(varis['e1'], states)
    expected = [{0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 2}]
    assert compare_list_of_sets(result[0].B, expected)
    assert result[1] == 5


def test_get_composite_state2():

    #od_pair, arcs, varis = main_sys
    varis = {}
    varis['e1'] = variable.Variable(name='e1', values=[1.5, 0.3, 0.15])

    states = [1, 2]
    result = variable.get_composite_state(varis['e1'], states)

    expected = [{0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 2}]
    #expected = [{0}, {1}, {2}, {1, 2}]
    assert compare_list_of_sets(result[0].B, expected)
    assert result[1] == 5


def test_get_composite_state3():

    #od_pair, arcs, varis = main_sys
    varis = {}
    varis['e1'] = variable.Variable(name='e1', values=[1.5, 0.3, 0.15])
    states = [0, 2]
    result = variable.get_composite_state(varis['e1'], states)

    expected = [{0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 2}]
    #expected = [{0}, {1}, {2}, {0, 2}]
    assert compare_list_of_sets(result[0].B, expected)
    assert result[1] == 4



def test_inference1(setup_inference):

    # case 1: no observatio
    cpms, varis, var_elim_order, arcs = setup_inference

    Msys = operation.variable_elim([cpms[v] for v in varis.keys()], var_elim_order )
    np.testing.assert_array_almost_equal(Msys.C, np.array([[0, 1]]).T)
    np.testing.assert_array_almost_equal(Msys.p, np.array([[0.1018, 0.8982]]).T)

    pf_sys = Msys.p[0]
    assert pf_sys == pytest.approx(0.1018, rel=1.0e-3)


def test_inference2(setup_inference):

    # case 2: observation
    cpms, varis, var_elim_order, arcs = setup_inference

    cnd_vars = [f'o{i}' for i in range(1, len(arcs) + 1)]
    cnd_states = [1, 1, 0, 1, 0, 1]  # observing e3, e5 failure

    Mobs = operation.condition([cpms[v] for v in varis.keys()], cnd_vars, cnd_states)
    # P(sys, obs)
    Msys_obs = operation.variable_elim(Mobs, var_elim_order)

    np.testing.assert_array_almost_equal(Msys_obs.C, np.array([[0, 1]]).T)
    np.testing.assert_array_almost_equal(Msys_obs.p, np.array([[2.765e-5, 5.515e-5]]).T)
    # P(sys=0|obs) = P(sys=0,obs) / P(obs) 
    pf_sys = Msys_obs.p[0] / np.sum(Msys_obs.p)
    assert pf_sys == pytest.approx(0.334, rel=1.0e-3)


@pytest.mark.skip('TODO')
def test_run_brc():
    """
    varis
    probs
    sys_fun
    max_sf
    max_nb

    run_brc(varis, probs, sys_fun, max_sf, max_nb)
    """

    pass

@pytest.mark.skip('TODO')
def test_get_decomp_depth_first():

    max_nb = 1000
    probs = {'e1': 0.1, 'e2': 0.2, 'e3': 0.3}
    varis = {}

    for i in range(1, 4):
        varis[f'e{i}'] = variable.Variable(name=f'e{i}', values=['Fail', 'Survive'])

    varis['od1'] = variable.Variable(name='od1', values=['Fail', 'Survive'])

    rules = {'s': [], 'f': []}

    G = nx.MultiGraph()
    G.add_edge('n1', 'n2', capacity=1)
    G.add_edge('n2', 'n3', 1)
    G.add_edge('n2', 'n3', 1)


def test_get_comp_st1():

    probs = {'e1': {0: 0.1, 1: 0.9},
             'e2': {0: 0.2, 1: 0.8},
             'e3': {0: 0.3, 1: 0.7}}

    varis = {}
    for i in range(1, 4):
        varis[f'e{i}'] = variable.Variable(name=f'e{i}', values=['Fail', 'Survive'])

    brs = [branch.Branch(down={f'e{i}': 0 for i in range(1, 4)},
                         up={f'e{i}': 1 for i in range(1, 4)},
                                down_state = 'u',
                         up_state = 'u',
                         p = 1.0)]
    st = brc.get_comp_st(brs)
    assert st == {'e1': 1, 'e2': 1, 'e3': 1}

    # surv_first = False
    st = brc.get_comp_st(brs, surv_first=False, varis=varis, probs=probs)
    assert st == {'e1': 1, 'e2': 1, 'e3': 1}


def test_get_comp_st2():

    probs = {'e1': {0: 0.1, 1: 0.9},
             'e2': {0: 0.2, 1: 0.8},
             'e3': {0: 0.3, 1: 0.7}}

    brs = [branch.Branch(down={'e1': 1, 'e2': 0, 'e3': 0},
                         up={'e1': 1, 'e2': 0, 'e3': 1},
                         down_state = 'u',
                         up_state = 'u',
                         p = 0.18),
           branch.Branch(down={'e1': 1, 'e2': 1, 'e3': 0},
                         up={'e1': 1, 'e2': 1, 'e3': 1},
                         down_state = 's',
                         up_state = 's',
                         p = 0.72),
           branch.Branch(down={'e1': 0, 'e2': 0, 'e3': 0},
                         up={'e1': 0, 'e2': 1, 'e3': 1},
                         down_state = 'u',
                         up_state = 'u',
                         p = 0.1),
           ]

    st = brc.get_comp_st(brs)
    assert st == {'e1': 1, 'e2': 0, 'e3': 1}


def test_get_connectivity_given_comps4():

    x_star = {'e1': 1, 'e2': 0, 'e3': 1}
    varis = {}
    for i in range(1, 4):
        varis[f'e{i}'] = variable.Variable(name=f'e{i}', values=['Fail', 'Survive'])

    od_pair = ('n1', 'n3')
    edges = {
         'e1': {'origin': 'n1', 'destination': 'n2', 'link_capacity': None, 'weight': 1.0},
         'e2': {'origin': 'n2', 'destination': 'n3', 'link_capacity': None, 'weight': 1.0},
         'e3': {'origin': 'n2', 'destination': 'n3', 'link_capacity': None, 'weight': 1.0},
         }

    G = nx.MultiDiGraph()
    G.add_edge('n1', 'n2', label='e1', key='e1', weight=1)
    G.add_edge('n2', 'n3', label='e2', key='e2', weight=1)
    G.add_edge('n2', 'n3', label='e3', key='e3', weight=1)
    G.add_node('n1', key='n1', label='n1')
    G.add_node('n2', key='n2', label='n2')
    G.add_node('n3', key='n3', label='n3')

    s_path_edges, s_path_nodes = trans.get_connectivity_given_comps(x_star, G, od_pair)

    assert s_path_edges == ['e1', 'e3']
    assert s_path_nodes == ['n1', 'n2', 'n3']

    #sys_fun = trans.sys_fun_wrap(od_pair, edges, varis)

    #sys_val, sys_st, comp_st_min = sys_fun(x_star)

    #assert sys_st == 's'
    #assert sys_val == None
    #assert comp_st_min == {'e1': 1, 'e3': 1}


@pytest.mark.skip('NYI')
def test_get_csys2(main_sys_bridge):

    od_pair, arcs, varis = main_sys_bridge
    comps_name = list(arcs.keys())

    # Intact state of component vector: zero-based index 
    comps_st_itc = {k: len(v.values) - 1 for k, v in varis.items()} # intact state (i.e. the highest state)
    d_time_itc, path_itc = trans.get_time_and_path_given_comps(comps_st_itc, od_pair, arcs, varis)

    # defines the system failure event
    thres = 2

    # Given a system function, i.e. sf_min_path, it should be represented by a function that only has "comps_st" as input.
    # can we define multi state for system here??
    sys_fun = trans.sys_fun_wrap(od_pair, arcs, varis, thres * d_time_itc)

    # Branch and bound
    output_path = Path(__file__).parent
    brs, rules, _, _ = brc.run(varis, probs, sys_fun, max_sf=1000, max_nb=1000),


    st_br_to_cs = {'f': 0, 's': 1, 'u': 2}
    #pdb.set_trace()
    result = brc.get_csys(brs, varis, st_br_to_cs)
    #cmat = result[0]
    cmat = result[0][np.argsort(result[0][:, 0])]
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
    #print(varis['e1'])
    #print(varis['e3'])
    # FIXME: not exactly matching
    #np.testing.assert_array_equal(cmat, expected)
    """
    np.testing.assert_array_equal(result[1]['e1'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]]))
    np.testing.assert_array_equal(result[1]['e2'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]))
    np.testing.assert_array_equal(result[1]['e3'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0]]))
    np.testing.assert_array_equal(result[1]['e4'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0]]))
    np.testing.assert_array_equal(result[1]['e5'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1]]))
    np.testing.assert_array_equal(result[1]['e6'].B, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0]]))
    """


@pytest.mark.skip('NYI')
@pytest.fixture()
def main_sys_bridge(data_bridge):

    # Network
    _, arcs, arc_times_h = info_bridge

    # od_pair is different from setup_bridge in test_trans.py
    od_pair = ('n5', 'n1')

    #arc_lens_km = trans.get_arcs_length(arcs, node_coords)
    #arc_times_h = {k: v/arcs_avg_kmh[k] for k, v in arc_lens_km.items()}

    #comps_name = [k for k in arcs] # *FIXME* this is not necessary if branch's down and up are defined by dictionary (instead of list)

    # Component events
    #no_arc_st = 2 # number of component states 
    #delay_rat = [10, 2, 1] # delay in travel time given each component state (ratio)
    varis = {}
    for k, v in arcs.items():
        #varis[k] = variable.Variable(name=k, B=[{0}, {1}, {2}] , values = [arc_times_h[k]*np.float64(x) for x in delay_rat])
        varis[k] = variable.Variable(name=k, values = [10.0*arc_times_h[k], arc_times_h[k]])

    return od_pair, arcs, varis


@pytest.fixture(scope='package')
def setup_inference(main_sys, setup_brs):

    G, _, arcs, _ = main_sys

    d_varis, brs = setup_brs
    varis = copy.deepcopy(d_varis)

    # Component events
    cpms = {}
    for k, v in arcs.items():
        cpms[k] = cpm.Cpm(variables=[varis[k]],
                          no_child = 1,
                          C = np.array([0, 1, 2]),
                          p = [0.1, 0.2, 0.7])

    st_br_to_cs = {'f': 0, 's': 1, 'u': 2}
    csys, varis = brc.get_csys(brs, varis, st_br_to_cs)

    # Damage observation
    C_o = np.array([[0, 0], [1, 0], [2, 0],
                    [0, 1], [1, 1], [2, 1],
                    [0, 2], [1, 2], [2, 2]])

    p_o = np.array([0.95, 0.04, 0.01,
                    0.3, 0.5, 0.2,
                    0.01, 0.19, 0.8]).T

    for i, k in enumerate(arcs, 1):
        name = f'o{i}'
        varis[name] = variable.Variable(name=name, values = [0, 1, 2])
        cpms[name] = cpm.Cpm(variables=[varis[name], varis[k]], no_child=1, C=C_o, p=p_o)

    # add observations
    added = np.ones(shape=(csys.shape[0], len(arcs)), dtype=np.int8) * 6
    csys = np.append(csys, added, axis=1)

    # add sys
    varis['sys'] = variable.Variable('sys', ['f', 's', 'u'])
    cpm_sys_vname = [f'e{i}' for i in range(1, len(arcs) + 1)] + [f'o{i}' for i in range(1, len(arcs) + 1)] # observations first, components later
    cpm_sys_vname.insert(0, 'sys')
    cpms['sys'] = cpm.Cpm(
        variables=[varis[k] for k in cpm_sys_vname],
        no_child = 1,
        C = csys,
        p = np.ones((len(csys), 1), int))

    var_elim_order_name = [f'o{i}' for i in range(1, len(arcs) + 1)] + [f'e{i}' for i in range(1, len(arcs) + 1)] # observations first, components later
    var_elim_order = [varis[k] for k in var_elim_order_name]

    return cpms, varis, var_elim_order, arcs


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


def test_proposed_branch_and_bound_using_probs(main_sys):
    # Branch and bound
    G, od_pair, arcs, d_varis = main_sys

    varis = copy.deepcopy(d_varis)

    # Intact state of component vector: zero-based index 
    comps_st_itc = {k: len(v.values) - 1 for k, v in varis.items()} # intact state (i.e. the highest state)
    d_time_itc, path_n, path_itc = trans.get_time_and_path_given_comps(comps_st_itc, G, od_pair, varis)

    # defines the system failure event
    thres = 2

    # Given a system function, i.e. sf_min_path, it should be represented by a function that only has "comps_st" as input.
    sys_fun = trans.sys_fun_wrap(G, od_pair, varis, thres * d_time_itc)

    p = [1/3, 1/3, 1/3]
    probs = {'e1': p, 'e2': p, 'e3': p,
             'e4': p, 'e5': p, 'e6': p}

    # Branch and bound
    output_path = Path(__file__).parent
    #t1 = time.perf_counter()
    brs, rules, _, monitor = brc.run(varis, probs, sys_fun, max_sf=100, max_nb=100)

    st_br_to_cs = {'f': 0, 's': 1, 'u': 2}
    csys, varis = brc.get_csys(brs, varis, st_br_to_cs)

    varis['sys'] = variable.Variable('sys', ['f', 's', 'u'])
    cpm_sys_vname = list(brs[0].up.keys())
    cpm_sys_vname.insert(0, 'sys')

    # Component events
    cpms = {}
    for k, v in arcs.items():
        cpms[k] = cpm.Cpm(variables=[varis[k]],
                          no_child = 1,
                          C = np.array([0, 1, 2]),
                          p = [0.1, 0.2, 0.7])

    cpms['sys'] = cpm.Cpm(
        variables=[varis[k] for k in cpm_sys_vname],
        no_child = 1,
        C = csys,
        p = np.ones((len(csys), 1), int))

    var_elim_order = [varis[k] for k in arcs.keys()]
    Msys = operation.variable_elim([cpms[v] for v in varis.keys()], var_elim_order )
    np.testing.assert_array_almost_equal(Msys.C, np.array([[0, 1]]).T)
    np.testing.assert_array_almost_equal(Msys.p, np.array([[0.1018, 0.8982]]).T)


def test_get_csys(setup_brs):

    varis, brs = setup_brs

    st_br_to_cs = {'f': 0, 's': 1, 'u': 2}
    csys, varis = brc.get_csys(brs, varis, st_br_to_cs)

    for i in range(1, 7):
        assert compare_list_of_sets(varis[f'e{i}'].B, [{0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 2}])
    #assert compare_list_of_sets(varis['e1'].B, [{0}, {1}, {2}, {0, 1}, {0, 1, 2}])
    #assert compare_list_of_sets(varis['e2'].B, [{0}, {1}, {2}, {1, 2}])
    #assert compare_list_of_sets(varis['e3'].B, [{0}, {1}, {2}, {0, 1}, {0, 1, 2}])
    #assert compare_list_of_sets(varis['e4'].B, [{0}, {1}, {2}, {0, 1, 2}, {0, 1}])
    #assert compare_list_of_sets(varis['e5'].B, [{0}, {1}, {2}, {0, 1, 2}, {0, 1}, {1, 2}])
    #assert compare_list_of_sets(varis['e6'].B, [{0}, {1}, {2}, {0, 1, 2}, {0, 1}])

    expected = np.array([[0, 3, 0, 6, 6, 6, 6],
                         [0, 2, 0, 3, 6, 6, 6],
                         [0, 2, 0, 2, 6, 3, 6],
                         [0, 6, 5, 6, 3, 0, 6],
                         [0, 6, 1, 6, 2, 0, 3],
                         [0, 6, 2, 6, 2, 0, 0],
                         [1, 2, 0, 2, 6, 2, 6],
                         [1, 6, 2, 6, 2, 0, 1],
                         [1, 6, 5, 6, 2, 0, 2],
                         [1, 6, 5, 6, 6, 5, 6]])

    assert compare_list_of_sets(csys, expected)


def test_get_csys3(main_sys):

    G, od_pair, arcs, d_varis = main_sys

    varis = copy.deepcopy(d_varis)

    p1 = {0: 0.05, 1: 0.15, 2: 0.80}
    p2 = {0: 0.01, 1: 0.09, 2: 0.90}

    probs = {'e1': p1, 'e2': p1, 'e3': p1,
             'e4': p2, 'e5': p2, 'e6': p2}

    #comps_name = list(arcs.keys())

    # Intact state of component vector: zero-based index 
    comps_st_itc = {k: len(v.values) - 1 for k, v in varis.items()} # intact state (i.e. the highest state)
    d_time_itc, path_n, path_itc = trans.get_time_and_path_given_comps(comps_st_itc, G, od_pair, varis)

    # defines the system failure event
    thres = 2

    # Given a system function, i.e. sf_min_path, it should be represented by a function that only has "comps_st" as input.
    sys_fun = trans.sys_fun_wrap(G, od_pair, varis, thres * d_time_itc)

    # Branch and bound
    output_path = Path(__file__).parent
    #pdb.set_trace()
    brs, rules, _, monitor = brc.run(varis, probs, sys_fun, max_sf=100, max_nb=100)

    st_br_to_cs = {'f': 0, 's': 1, 'u': 2}
    #pdb.set_trace()
    csys, varis = brc.get_csys(brs, varis, st_br_to_cs)

    varis['sys'] = variable.Variable('sys', ['f', 's', 'u'])
    cpm_sys_vname = list(brs[0].up.keys())
    cpm_sys_vname.insert(0, 'sys')

    # Component events
    cpms = {}
    for k, v in arcs.items():
        cpms[k] = cpm.Cpm(variables=[varis[k]],
                          no_child = 1,
                          C = np.array([0, 1, 2]),
                          p = [0.1, 0.2, 0.7])

    cpms['sys'] = cpm.Cpm(
        variables=[varis[k] for k in cpm_sys_vname],
        no_child = 1,
        C = csys,
        p = np.ones((len(csys), 1), int))

    var_elim_order = [varis[k] for k in arcs.keys()]
    Msys = operation.variable_elim([cpms[v] for v in varis.keys()], var_elim_order )
    np.testing.assert_array_almost_equal(Msys.C, np.array([[0, 1]]).T)
    np.testing.assert_array_almost_equal(Msys.p, np.array([[0.1018, 0.8982]]).T)


def test_inference1(setup_inference):

    # case 1: no observatio
    cpms, varis, var_elim_order, arcs = setup_inference

    Msys = operation.variable_elim([cpms[v] for v in varis.keys()], var_elim_order )
    np.testing.assert_array_almost_equal(Msys.C, np.array([[0, 1]]).T)
    np.testing.assert_array_almost_equal(Msys.p, np.array([[0.1018, 0.8982]]).T)

    pf_sys = Msys.p[0]
    assert pf_sys == pytest.approx(0.1018, rel=1.0e-3)


def test_inference2(setup_inference):

    # case 2: observation
    cpms, varis, var_elim_order, arcs = setup_inference

    cnd_vars = [f'o{i}' for i in range(1, len(arcs) + 1)]
    cnd_states = [1, 1, 0, 1, 0, 1]  # observing e3, e5 failure

    Mobs = operation.condition([cpms[v] for v in varis.keys()], cnd_vars, cnd_states)
    # P(sys, obs)
    Msys_obs = operation.variable_elim(Mobs, var_elim_order)

    np.testing.assert_array_almost_equal(Msys_obs.C, np.array([[0, 1]]).T)
    np.testing.assert_array_almost_equal(Msys_obs.p, np.array([[2.765e-5, 5.515e-5]]).T)
    # P(sys=0|obs) = P(sys=0,obs) / P(obs) 
    pf_sys = Msys_obs.p[0] / np.sum(Msys_obs.p)
    assert pf_sys == pytest.approx(0.334, rel=1.0e-3)


def test_get_connectivity_given_comps4():

    x_star = {'e1': 1, 'e2': 0, 'e3': 1}
    varis = {}
    for i in range(1, 4):
        varis[f'e{i}'] = variable.Variable(name=f'e{i}', values=['Fail', 'Survive'])

    od_pair = ('n1', 'n3')
    edges = {
         'e1': {'origin': 'n1', 'destination': 'n2', 'link_capacity': None, 'weight': 1.0},
         'e2': {'origin': 'n2', 'destination': 'n3', 'link_capacity': None, 'weight': 1.0},
         'e3': {'origin': 'n2', 'destination': 'n3', 'link_capacity': None, 'weight': 1.0},
         }

    G = nx.MultiDiGraph()
    G.add_edge('n1', 'n2', label='e1', key='e1', weight=1)
    G.add_edge('n2', 'n3', label='e2', key='e2', weight=1)
    G.add_edge('n2', 'n3', label='e3', key='e3', weight=1)
    G.add_node('n1', key='n1', label='n1')
    G.add_node('n2', key='n2', label='n2')
    G.add_node('n3', key='n3', label='n3')

    s_path_edges, s_path_nodes = trans.get_connectivity_given_comps(x_star, G, od_pair)

    assert s_path_edges == ['e1', 'e3']
    assert s_path_nodes == ['n1', 'n2', 'n3']

    #sys_fun = trans.sys_fun_wrap(od_pair, edges, varis)

    #sys_val, sys_st, comp_st_min = sys_fun(x_star)

    #assert sys_st == 's'
    #assert sys_val == None
    #assert comp_st_min == {'e1': 1, 'e3': 1}

