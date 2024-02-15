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

from BNS_JT import trans, branch, variable, cpm, gen_bnb


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

    # Component events
    #no_arc_st = 3 # number of component states 
    varis = {}
    delay_rat = [10, 2, 1] # delay in travel time given each component state (ratio)
    for k, v in arcs.items():
        varis[k] = variable.Variable(name=k, B=[{0}, {1}, {2}] , values = [arc_times_h[k]*np.float64(x) for x in delay_rat])

    return od_pair, arcs, varis


@pytest.mark.skip('NYI')
def test_get_csys_from_brs2(main_sys_bridge):

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
    #pdb.set_trace()
    brs, rules, _ = gen_bnb.proposed_branch_and_bound(sys_fun, varis, max_br=1000,
                                                              output_path=output_path, key='bridge', flag=True)


    st_br_to_cs = {'f': 0, 's': 1, 'u': 2}
    #pdb.set_trace()
    result = gen_bnb.get_csys_from_brs(brs, varis, st_br_to_cs)
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
        varis[k] = variable.Variable(name=k, B=[{0}, {1}, {0, 1}] , values = [10.0*arc_times_h[k], arc_times_h[k]])

    return od_pair, arcs, varis


@pytest.fixture(scope='package')
def setup_brs(main_sys):

    od_pair, arcs, d_varis = main_sys

    varis = copy.deepcopy(d_varis)

    # Intact state of component vector: zero-based index 
    comps_st_itc = {k: len(v.values) - 1 for k, v in varis.items()} # intact state (i.e. the highest state)
    d_time_itc, path_itc = trans.get_time_and_path_given_comps(comps_st_itc, od_pair, arcs, varis)

    # defines the system failure event
    thres = 2
    # Given a system function, i.e. sf_min_path, it should be represented by a function that only has "comps_st" as input.
    sys_fun = trans.sys_fun_wrap(od_pair, arcs, varis, thres * d_time_itc)

    file_brs = Path(__file__).parent.joinpath('brs_bridge.pk')
    if file_brs.exists():
        with open(file_brs, 'rb') as fi:
            brs = pickle.load(fi)
            print(f'{file_brs} loaded')
    else:
        # Branch and bound
        output_path = Path(__file__).parent
        #pdb.set_trace()
        brs, rules, sys_res = gen_bnb.proposed_branch_and_bound(
            sys_fun, varis, max_br=1000,
            output_path=output_path, key='bridge', flag=True)

    return varis, brs


@pytest.fixture(scope='package')
def setup_inference(main_sys, setup_brs):

    _, arcs, _ = main_sys

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
    csys, varis = gen_bnb.get_csys_from_brs(brs, varis, st_br_to_cs)

    # Damage observation
    C_o = np.array([[0, 0], [1, 0], [2, 0],
                    [0, 1], [1, 1], [2, 1],
                    [0, 2], [1, 2], [2, 2]])

    p_o = np.array([0.95, 0.04, 0.01,
                    0.3, 0.5, 0.2,
                    0.01, 0.19, 0.8]).T

    for i, k in enumerate(arcs, 1):
        name = f'o{i}'
        varis[name] = variable.Variable(name=name, B=[{0}, {1}, {2}, {0, 1, 2}], values = [0, 1, 2])
        cpms[name] = cpm.Cpm(variables=[varis[name], varis[k]], no_child=1, C=C_o, p=p_o)

    # add observations
    added = np.ones(shape=(csys.shape[0], len(arcs)), dtype=np.int8) * 3
    csys = np.append(csys, added, axis=1)

    # add sys
    varis['sys'] = variable.Variable('sys', [{0}, {1}, {2}], ['f', 's', 'u'])
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


@pytest.mark.skip('notused')
#@pytest.fixture(scope='session')
def setup_inference_not_used(main_sys):

    od_pair, arcs, d_varis = main_sys

    varis = copy.deepcopy(d_varis)

    # Component events
    cpms = {}
    for k, v in arcs.items():
        cpms[k] = cpm.Cpm(variables=[varis[k]],
                          no_child = 1,
                          C = np.array([0, 1, 2]),
                          p = [0.1, 0.2, 0.7])
    """
    varis values
    e1: 1.5, 0.3, 0.15
    e2: 0.901, 0.1803, 0.0901
    e3: 0.901, 0.1803, 0.0901
    e4: 1.054, 0.211, 0.1054
    e5: 0.943, 0.189, 0.0943
    e6: 0.707, 0.141, 0.0707
    """
    # Damage observation
    C_o = np.array([[0, 0], [1, 0], [2, 0],
                    [0, 1], [1, 1], [2, 1],
                    [0, 2], [1, 2], [2, 2]])

    p_o = np.array([0.95, 0.04, 0.01,
                    0.3, 0.5, 0.2,
                    0.01, 0.19, 0.8]).T

    for i, k in enumerate(arcs, 1):
        name = f'o{i}'
        varis[name] = variable.Variable(name=name, B=[{0}, {1}, {2}], values = [0, 1, 2])
        cpms[name] = cpm.Cpm(variables=[varis[name], varis[k]], no_child=1, C=C_o, p=p_o)

    # Intact state of component vector: zero-based index 
    comps_st_itc = {k: len(v.values) - 1 for k, v in varis.items()} # intact state (i.e. the highest state)
    d_time_itc, path_itc = trans.get_time_and_path_given_comps(comps_st_itc, od_pair, arcs, varis)
    # intact state: 0.1844 over n1 - (e2) - n5 - (e5) - n3
    # defines the system failure event
    thres = 2
    # Given a system function, i.e. sf_min_path, it should be represented by a function that only has "comps_st" as input.
    sys_fun = trans.sys_fun_wrap(od_pair, arcs, varis, thres * d_time_itc)

    # Branch and bound
    output_path = Path(__file__).parent
    #pdb.set_trace()
    brs, rules, sys_res = gen_bnb.proposed_branch_and_bound(
            sys_fun, varis, max_br=1000, output_path=output_path, key='bridge', flag=False)

    st_br_to_cs = {'f': 0, 's': 1, 'u': 2}
    csys, varis = gen_bnb.get_csys_from_brs(brs, varis, st_br_to_cs)
    varis['sys'] = variable.Variable('sys', [{0}, {1}, {2}], ['f', 's', 'u'])
    cpm_sys_vname = list(brs[0].up.keys())
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


def test_init_branch1():

    rules = {'s':[], 'f': [], 'u': []}
    worst = {f'e{i}': 0 for i in range(1, 7)}
    best = {f'e{i}': 2 for i in range(1, 7)}

    brs = gen_bnb.init_branch(worst, best, rules)

    assert len(brs) == 1
    assert brs[0].up_state == 'u'
    assert brs[0].down_state == 'u'
    assert brs[0].down == {f'e{i}': 0 for i in range(1, 7)}
    assert brs[0].up == {f'e{i}': 2 for i in range(1, 7)}

    rules = {'s': [{'e2': 2, 'e5': 2}], 'f': [], 'u': []}
    brs = gen_bnb.init_branch(worst, best, rules)

    assert len(brs) == 1
    assert brs[0].up_state == 's'
    assert brs[0].down_state == 'u'
    assert brs[0].down == {f'e{i}': 0 for i in range(1, 7)}
    assert brs[0].up == {f'e{i}': 2 for i in range(1, 7)}

    rules = {'s': [{'e2': 2, 'e5': 2}], 'f': [{f'e{x}': 0 for x in range(1, 7)}], 'u': []}
    brs = gen_bnb.init_branch(worst, best, rules)

    assert len(brs) == 1
    assert brs[0].up_state == 's'
    assert brs[0].down_state == 'f'
    assert brs[0].down == {f'e{i}': 0 for i in range(1, 7)}
    assert brs[0].up == {f'e{i}': 2 for i in range(1, 7)}

    rules = {'s': [{'e2': 2, 'e5': 2}, {'e2': 2, 'e6': 2, 'e4': 2}], 'f': [{f'e{x}': 0 for x in range(1, 7)}], 'u': []}
    brs = gen_bnb.init_branch(worst, best, rules)

    assert len(brs) == 1
    assert brs[0].up_state == 's'
    assert brs[0].down_state == 'f'
    assert brs[0].down == {f'e{i}': 0 for i in range(1, 7)}
    assert brs[0].up == {f'e{i}': 2 for i in range(1, 7)}

@pytest.mark.skip('removed')
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


@pytest.mark.skip('removed')
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


@pytest.mark.skip('removed')
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


@pytest.mark.skip('removed')
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


@pytest.mark.skip('removed')
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


@pytest.mark.skip('removed')
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


@pytest.mark.skip('removed')
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


def test_approx_branch_prob():

    d = {f'e{i}': 0 for i in range(1, 7)}
    u = {f'e{i}': 2 for i in range(1, 7)}

    p1 = {0: 0.05, 1: 0.15, 2: 0.80}
    p2 = {0: 0.01, 1: 0.09, 2: 0.90}

    p = {'e1': p1, 'e2': p1, 'e3': p1,
         'e4': p2, 'e5': p2, 'e6': p2}

    result = gen_bnb.approx_branch_prob(d, u, p)
    assert result == 1.0

    d = {'e1': 2, 'e2': 0, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    u = {f'e{i}': 2 for i in range(1, 7)}

    result = gen_bnb.approx_branch_prob(d, u, p)
    assert pytest.approx(result) == 0.80**2*0.90**3

    d = {'e1': 0, 'e2': 1, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}
    u = {f'e{i}': 2 for i in range(1, 7)}

    result = gen_bnb.approx_branch_prob(d, u, p)
    assert pytest.approx(result) == 0.95

    d = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}
    u = {'e1': 2, 'e2': 0, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}

    result = gen_bnb.approx_branch_prob(d, u, p)
    assert pytest.approx(result) == 0.05


def test_approx_joint_prob_compat_rules():

    p1 = {0: 0.05, 1: 0.15, 2: 0.80}
    p2 = {0: 0.01, 1: 0.09, 2: 0.90}

    p = {'e1': p1, 'e2': p1, 'e3': p1,
         'e4': p2, 'e5': p2, 'e6': p2}

    d = {f'e{i}': 0 for i in range(1, 7)}
    u = {f'e{i}': 2 for i in range(1, 7)}
    rule = {'e2': 2, 'e5': 2}
    rule_st = 's'

    result = gen_bnb.approx_joint_prob_compat_rule(d, u, rule, rule_st, p)
    assert pytest.approx(result) == 0.8*0.9

    rule = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}
    rule_st = 'f'

    result = gen_bnb.approx_joint_prob_compat_rule(d, u, rule, rule_st, p)
    assert pytest.approx(result) == 0.05**3*0.01**3


def test_proposed_branch_and_bound_using_probs(main_sys):
    # Branch and bound

    od_pair, arcs, d_varis = main_sys

    varis = copy.deepcopy(d_varis)

    # Intact state of component vector: zero-based index 
    comps_st_itc = {k: len(v.values) - 1 for k, v in varis.items()} # intact state (i.e. the highest state)
    d_time_itc, path_itc = trans.get_time_and_path_given_comps(comps_st_itc, od_pair, arcs, varis)

    # defines the system failure event
    thres = 2

    # Given a system function, i.e. sf_min_path, it should be represented by a function that only has "comps_st" as input.
    sys_fun = trans.sys_fun_wrap(od_pair, arcs, varis, thres * d_time_itc)

    p = {0: 1/3, 1: 1/3, 2: 1/3}
    probs = {'e1': p, 'e2': p, 'e3': p,
             'e4': p, 'e5': p, 'e6': p}

    # Branch and bound
    output_path = Path(__file__).parent
    #t1 = time.perf_counter()
    brs, rules, _ = gen_bnb.proposed_branch_and_bound_using_probs(
            sys_fun, varis, probs, max_br=100,
            output_path=output_path, key='bridge', flag=False)

    st_br_to_cs = {'f': 0, 's': 1, 'u': 2}
    csys, varis = gen_bnb.get_csys_from_brs(brs, varis, st_br_to_cs)

    varis['sys'] = variable.Variable('sys', [{0}, {1}, {2}], ['f', 's', 'u'])
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
    Msys = cpm.variable_elim([cpms[v] for v in varis.keys()], var_elim_order )
    np.testing.assert_array_almost_equal(Msys.C, np.array([[0, 1]]).T)
    np.testing.assert_array_almost_equal(Msys.p, np.array([[0.1018, 0.8982]]).T)


def test_get_state0():

    rules = {'s': [], 'f': [], 'u': []}

    cst = {f'e{i}': 2 for i in range(1, 7)}
    result = gen_bnb.get_state(cst, rules)
    assert result == 'u'

    cst = {f'e{i}': 0 for i in range(1, 7)}
    result = gen_bnb.get_state(cst, rules)
    assert result == 'u'


def test_get_compat_rules():

    upper = {f'e{i}': 2 for i in range(1, 7)}
    lower = {f'e{i}': 0 for i in range(1, 7)}
    rules = {'s': [], 'f': []}
    result = gen_bnb.get_compat_rules(lower, upper, rules)
    assert result == rules

    rules = {'s': [{'e2': 2, 'e5': 2}], 'f': []}
    result = gen_bnb.get_compat_rules(lower, upper, rules)
    assert result == {'s': [{'e2': 2, 'e5': 2}], 'f': []}

    rules = {'s': [{'e2': 1, 'e5': 2}], 'f': [{f'e{i}': 0 for i in range(1, 7)}]}
    result = gen_bnb.get_compat_rules(lower, upper, rules)
    assert result['s'] == rules['s']
    assert result['f'] == rules['f']

    upper = {'e1': 2, 'e2': 0, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    lower = {f'e{i}': 0 for i in range(1, 7)}
    rules = {'s': [{'e2': 1, 'e5': 2}],
             'f': [{f'e{i}': 0 for i in range(1, 7)}]}
    result = gen_bnb.get_compat_rules(lower, upper, rules)
    assert result['s'] == []
    assert result['f'] == [{'e1': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}]

    upper = {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    lower = {'e1': 2, 'e2': 0, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    rules = {'s': [{'e2': 1, 'e5': 2}, {'e1': 2, 'e3': 2, 'e5': 2},
                   {'e2': 2, 'e4': 2, 'e6': 2}],
             'f': [{f'e{i}': 0 for i in range(1, 7)}]}

    result = gen_bnb.get_compat_rules(lower, upper, rules)
    assert result['s'] == [{'e2': 1}, {'e2': 2}]
    assert result['f'] == []


def test_get_compat_rules_list():

    upper = {f'e{i}': 2 for i in range(1, 7)}
    lower = {f'e{i}': 0 for i in range(1, 7)}

    rules = []
    result = gen_bnb.get_compat_rules_list(lower, upper, rules)
    assert result == []

    rules = [({'e2': 2, 'e5': 2}, 's')]
    result = gen_bnb.get_compat_rules_list(lower, upper, rules)
    assert result == [({'e2': 2, 'e5': 2}, 's')]

    rules = [({'e2': 1, 'e5': 2}, 's')]
    result = gen_bnb.get_compat_rules_list(lower, upper, rules)
    assert result ==  [({'e2': 1, 'e5': 2}, 's')]

    rules = [({'e2': 1, 'e5': 2}, 's'), ({'e1': 1, 'e2': 0}, 'f')]
    result = gen_bnb.get_compat_rules_list(lower, upper, rules)
    assert result[0] == rules[0]
    assert result[1] == rules[1]

    lower = {'e1': 0, 'e2': 0, 'e3': 2 , 'e4': 1, 'e5': 0, 'e6': 2}
    upper = {'e1': 0, 'e2': 0, 'e3': 2 , 'e4': 1, 'e5': 0, 'e6': 2}
    rules = [({'e2': 1, 'e5': 2}, 's'), ({'e1': 1, 'e2': 0}, 'f')]
    result = gen_bnb.get_compat_rules_list(lower, upper, rules)
    assert result == [({'e1': 1, 'e2': 0}, 'f')]


def test_get_state1():

    rules = {'s': [{'e2': 2, 'e5': 2}], 'f': [], 'u': []}

    cst = {f'e{i}': 2 for i in range(1, 7)}
    result = gen_bnb.get_state(cst, rules)
    assert result == 's'

    cst = {f'e{i}': 0 for i in range(1, 7)}
    result = gen_bnb.get_state(cst, rules)
    assert result == 'u'


def test_get_state2():

    rules = {'s': [{'e2': 2, 'e5': 2}], 'f': [{'e1': 1, 'e2': 0}], 'u': []}

    cst = {f'e{i}': 2 for i in range(1, 7)}
    result = gen_bnb.get_state(cst, rules)
    assert result == 's'

    cst = {f'e{i}': 0 for i in range(1, 7)}
    result = gen_bnb.get_state(cst, rules)
    assert result == 'f'


def test_get_state4():

    rules = {'s': [{'e2': 1, 'e5': 2}], 'f': [{'e1': 1, 'e2': 0}], 'u': []}
    cst = {'e1': 0, 'e2': 0, 'e3': 2 , 'e4': 1, 'e5': 0, 'e6': 2}
    result = gen_bnb.get_state(cst, rules)
    assert result == 'f'


def test_update_rule_set0():

    rules = {'s': [], 'f': [], 'u': []}
    rule_new = ({'e2':2, 'e5':2}, 's')

    result = gen_bnb.update_rule_set(rules, rule_new)
    assert result == {'s': [{'e2': 2, 'e5': 2}], 'f': [], 'u': []}


def test_update_rule_set1():

    rules = {'s': [{'e2': 2, 'e5': 2}], 'f': [], 'u': []}
    rule_new = {f'e{i}':0 for i in range(1, 7)}, 'f'

    result = gen_bnb.update_rule_set(rules, rule_new)
    expected = {'s': [{'e2': 2, 'e5': 2}], 'f': [{'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5':0, 'e6': 0}], 'u': []}
    assert result == expected


def test_update_rule_set2():

    rules = {'s': [{'e2':2, 'e5':2}], 'u': [], 'f':[{'e2': 1, 'e5': 1}]}
    rule_new = {'e2': 1, 'e5': 2}, 's'

    result = gen_bnb.update_rule_set(rules, rule_new)
    expected = {'s': [{'e2':1, 'e5':2}], 'u': [], 'f':[{'e2': 1, 'e5': 1}]}
    assert result == expected

    rule_new = {'e2': 1, 'e5': 0}, 'f'

    result = gen_bnb.update_rule_set(rules, rule_new)
    expected = {'s': [{'e2':1, 'e5':2}], 'u': [], 'f':[{'e2': 1, 'e5': 1}]}
    assert result == expected


@pytest.mark.skip('removed')
def test_add_rule():
    rules = [{'e2':2, 'e5':2}]
    rules_st = ['s']
    rule_new = {f'e{i}':0 for i in range(1, 7)}
    fail_or_surv = 'f'

    result = gen_bnb.add_rule(rules, rules_st, rule_new, fail_or_surv)

    assert result[0] == [{'e2': 2, 'e5': 2}, {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5':0, 'e6': 0}]
    assert result[1] == ['s', 'f']

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


def test_get_decomp_comp_using_probs_0():

    rules = {'s': [{'e2': 2, 'e5': 2}],
            'f': [{'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}]}
    upper = {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    lower = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}

    p1 = {0: 0.05, 1: 0.15, 2: 0.80}
    p2 = {0: 0.01, 1: 0.09, 2: 0.90}

    probs = {'e1': p1, 'e2': p1, 'e3': p1,
             'e4': p2, 'e5': p2, 'e6': p2}

    result = gen_bnb.get_decomp_comp_using_probs(lower, upper, rules, probs)

    assert result == ('e2', 2)

    rules = {'s': [{'e2': 2, 'e5': 2}, {'e1': 2, 'e3': 2, 'e5': 2}],
            'f': [{'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}],
            }
    upper = {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    lower = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}
    result = gen_bnb.get_decomp_comp_using_probs(lower, upper, rules, probs)

    assert result == ('e5', 2)

    rules = {'s': [{'e2': 2, 'e5': 2}, {'e1': 2, 'e3': 2, 'e5': 2}, {'e2': 2, 'e4': 2, 'e6': 2}],
            'f': [{'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}],
            }
    upper = {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    lower = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}
    result = gen_bnb.get_decomp_comp_using_probs(lower, upper, rules, probs)

    assert result == ('e2', 2)

    rules = {'s': [{'e1': 2, 'e3': 2, 'e5': 2}],
            'f': [{'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}],
            }
    upper = {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    lower = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}
    result = gen_bnb.get_decomp_comp_using_probs(lower, upper, rules, probs)

    assert result == ('e1', 2)


def test_get_decomp_comp_0():

    rules = {'s': [{'e2': 2, 'e5': 2}], 'u': [],
            'f': [{'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}]}
    upper = {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    lower = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}
    result = gen_bnb.get_decomp_comp(lower, upper, rules)

    assert result == ('e2', 2)

    rules = {'s': [{'e2': 2, 'e5': 2}, {'e1': 2, 'e3': 2, 'e5': 2}],
            'f': [{'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}],
            'u': []}
    upper = {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    lower = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}
    result = gen_bnb.get_decomp_comp(lower, upper, rules)

    assert result == ('e5', 2)

    rules = {'s': [{'e2': 2, 'e5': 2}, {'e1': 2, 'e3': 2, 'e5': 2}, {'e2': 2, 'e4': 2, 'e6': 2}],
            'f': [{'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}],
            'u': []}
    upper = {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    lower = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}
    result = gen_bnb.get_decomp_comp(lower, upper, rules)

    assert result == ('e2', 2)

    rules = {'s': [{'e1': 2, 'e3': 2, 'e5': 2}],
            'f': [{'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}],
            'u': []}
    upper = {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    lower = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}
    result = gen_bnb.get_decomp_comp(lower, upper, rules)

    assert result == ('e1', 2)

    rules = {'s': [{'e2': 2, 'e6': 2, 'e4': 2},
                   {'e2': 1, 'e5': 2},
                   {'e1': 2, 'e3': 2, 'e5': 2}],
             'f': [{'e1': 0, 'e2': 1, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0},
                   {'e2': 0, 'e3': 1}, {'e2': 0, 'e5': 1}]}

    up = {'e1': 2, 'e2': 0, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    down = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}

    result = gen_bnb.get_decomp_comp(down, up, rules)

    assert result == ('e3', 2) or result == ('e5', 2)


@pytest.mark.skip('removed')
def test_get_comp_st_for_next_bnb0():
    up = {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    down = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}
    rules = [{'e2': 2, 'e5': 2},
             {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}]
    rules_st = ['s', 'f']
    result = gen_bnb.get_comp_st_for_next_bnb(up, down, rules, rules_st)

    assert result[0] == 'e5'
    assert result[1] == 2

    up = {'e1': 2, 'e2': 0, 'e3': 2, 'e4': 2, 'e5': 2, 'e6': 2}
    down = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}
    rules = [{'e2': 2, 'e6': 2, 'e4': 2}, {'e2': 1, 'e5': 2}, {'e1': 2, 'e3': 2, 'e5': 2}, {'e1': 0, 'e2': 1, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}, {'e2': 0, 'e3': 1}, {'e2': 0, 'e5': 1}]
    rules_st = ['s', 's', 's', 'f', 'f', 'f']
    result = gen_bnb.get_comp_st_for_next_bnb(up, down, rules, rules_st)

    assert result[0] == 'e3'
    assert result[1] == 2


@pytest.mark.skip('removed')
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

    rule, sys_res = gen_bnb.run_sys_fn(cst, sys_fun, varis)
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

    rule, sys_res = gen_bnb.run_sys_fn(cst, sys_fun, varis)

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

    rule, sys_res = gen_bnb.run_sys_fn(cst, sys_fun, varis)

    np.testing.assert_almost_equal(sys_res['sys_val'].values, np.array([0.266]), decimal=3)
    assert sys_res['comp_st'].values[0] == {'e1': 2, 'e2': 2, 'e3': 2, 'e4': 2, 'e5': 1, 'e6': 2}
    assert sys_res['comp_st_min'].values == [{'e2': 2, 'e6': 2, 'e4': 2}]
    assert rule == ({'e2': 2, 'e6': 2, 'e4': 2}, 's')


def test_get_composite_state1():

    varis = {}
    varis['e1'] = variable.Variable(name='e1', B=[{0}, {1}, {2}], values=[1.5, 0.3, 0.15])

    states = [1, 2]
    result = variable.get_composite_state(varis['e1'], states)
    expected = [{0}, {1}, {2}, {1, 2}]
    assert compare_list_of_sets(result[0].B, expected)
    assert result[1] == 3


def test_get_composite_state2(main_sys):

    #od_pair, arcs, varis = main_sys
    varis = {}
    varis['e1'] = variable.Variable(name='e1', B=[{0}, {1}, {2}, {1, 2}], values=[1.5, 0.3, 0.15])

    states = [1, 2]
    result = variable.get_composite_state(varis['e1'], states)

    expected = [{0}, {1}, {2}, {1, 2}]
    assert compare_list_of_sets(result[0].B, expected)
    assert result[1] == 3


def test_get_composite_state3(main_sys):

    #od_pair, arcs, varis = main_sys
    varis = {}
    varis['e1'] = variable.Variable(name='e1', B=[{0}, {1}, {2}], values=[1.5, 0.3, 0.15])
    states = [0, 2]
    result = variable.get_composite_state(varis['e1'], states)

    expected = [{0}, {1}, {2}, {0, 2}]
    assert compare_list_of_sets(result[0].B, expected)
    assert result[1] == 3


def test_get_csys_from_brs3(main_sys):

    od_pair, arcs, d_varis = main_sys

    varis = copy.deepcopy(d_varis)

    p1 = {0: 0.05, 1: 0.15, 2: 0.80}
    p2 = {0: 0.01, 1: 0.09, 2: 0.90}

    probs = {'e1': p1, 'e2': p1, 'e3': p1,
             'e4': p2, 'e5': p2, 'e6': p2}

    #comps_name = list(arcs.keys())

    # Intact state of component vector: zero-based index 
    comps_st_itc = {k: len(v.values) - 1 for k, v in varis.items()} # intact state (i.e. the highest state)
    d_time_itc, path_itc = trans.get_time_and_path_given_comps(comps_st_itc, od_pair, arcs, varis)

    # defines the system failure event
    thres = 2

    # Given a system function, i.e. sf_min_path, it should be represented by a function that only has "comps_st" as input.
    sys_fun = trans.sys_fun_wrap(od_pair, arcs, varis, thres * d_time_itc)

    # Branch and bound
    output_path = Path(__file__).parent
    #pdb.set_trace()
    brs, rules, _ = gen_bnb.proposed_branch_and_bound_using_probs(
            sys_fun, varis, probs, max_br=100,
            output_path=output_path, key='bridge', flag=False)

    st_br_to_cs = {'f': 0, 's': 1, 'u': 2}
    #pdb.set_trace()
    csys, varis = gen_bnb.get_csys_from_brs(brs, varis, st_br_to_cs)

    varis['sys'] = variable.Variable('sys', [{0}, {1}, {2}], ['f', 's', 'u'])
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
    Msys = cpm.variable_elim([cpms[v] for v in varis.keys()], var_elim_order )
    np.testing.assert_array_almost_equal(Msys.C, np.array([[0, 1]]).T)
    np.testing.assert_array_almost_equal(Msys.p, np.array([[0.1018, 0.8982]]).T)


def test_get_c_from_br(main_sys):

    _, _, d_varis = main_sys

    varis = copy.deepcopy(d_varis)

    st_br_to_cs = {'f': 0, 's': 1, 'u': 2}

    # test1
    br = branch.Branch({'e1': 1, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0},
                       {'e1': 2, 'e2': 0, 'e3': 1, 'e4': 2, 'e5': 2, 'e6': 2}, 'f', 'f')

    varis, cst = gen_bnb.get_c_from_br(br, varis, st_br_to_cs)

    assert cst.tolist() == [0, 3, 0, 3, 3, 3, 3]
    assert compare_list_of_sets(varis['e1'].B, [{0}, {1}, {2}, {1, 2}])
    assert compare_list_of_sets(varis['e2'].B, [{0}, {1}, {2}])
    assert compare_list_of_sets(varis['e3'].B, [{0}, {1}, {2}, {0, 1}])
    assert compare_list_of_sets(varis['e4'].B, [{0}, {1}, {2}, {0, 1, 2}])
    assert compare_list_of_sets(varis['e5'].B, [{0}, {1}, {2}, {0, 1, 2}])
    assert compare_list_of_sets(varis['e6'].B, [{0}, {1}, {2}, {0, 1, 2}])

    # test2
    # using the previous output as an input
    br = branch.Branch({'e1': 0, 'e2': 0, 'e3': 2, 'e4': 0, 'e5': 0, 'e6': 0},
                       {'e1': 2, 'e2': 0, 'e3': 2, 'e4': 2, 'e5': 1, 'e6': 2}, 'f', 'f')

    varis, cst = gen_bnb.get_c_from_br(br, varis, st_br_to_cs)
    assert cst.tolist() == [0, 4, 0, 2, 3, 4, 3]
    assert compare_list_of_sets(varis['e1'].B, [{0}, {1}, {2}, {1, 2}, {0, 1, 2}])
    assert compare_list_of_sets(varis['e2'].B, [{0}, {1}, {2}])
    assert compare_list_of_sets(varis['e3'].B, [{0}, {1}, {2}, {0, 1}])
    assert compare_list_of_sets(varis['e4'].B, [{0}, {1}, {2}, {0, 1, 2}])
    assert compare_list_of_sets(varis['e5'].B, [{0}, {1}, {2}, {0, 1, 2}, {0, 1}])
    assert compare_list_of_sets(varis['e6'].B, [{0}, {1}, {2}, {0, 1, 2}])


def test_get_csys_from_brs(setup_brs):

    varis, brs = setup_brs

    st_br_to_cs = {'f': 0, 's': 1, 'u': 2}
    csys, varis = gen_bnb.get_csys_from_brs(brs, varis, st_br_to_cs)

    assert compare_list_of_sets(varis['e1'].B, [{0}, {1}, {2}, {0, 1}, {0, 1, 2}])
    assert compare_list_of_sets(varis['e2'].B, [{0}, {1}, {2}, {1, 2}])
    assert compare_list_of_sets(varis['e3'].B, [{0}, {1}, {2}, {0, 1}, {0, 1, 2}])
    assert compare_list_of_sets(varis['e4'].B, [{0}, {1}, {2}, {0, 1, 2}, {0, 1}])
    assert compare_list_of_sets(varis['e5'].B, [{0}, {1}, {2}, {0, 1, 2}, {0, 1}, {1, 2}])
    assert compare_list_of_sets(varis['e6'].B, [{0}, {1}, {2}, {0, 1, 2}, {0, 1}])

    expected = np.array([[0, 3, 0, 3, 3, 3, 3],
                         [0, 2, 0, 4, 3, 3, 3],
                         [0, 2, 0, 2, 3, 4, 3],
                         [0, 4, 3, 3, 4, 0, 3],
                         [0, 4, 1, 3, 2, 0, 4],
                         [0, 4, 2, 3, 2, 0, 0],
                         [1, 2, 0, 2, 3, 2, 3],
                         [1, 4, 2, 3, 2, 0, 1],
                         [1, 4, 3, 3, 2, 0, 2],
                         [1, 4, 3, 3, 3, 5, 3]])

    assert compare_list_of_sets(csys, expected)


def test_inference1(setup_inference):

    # case 1: no observation
    cpms, varis, var_elim_order, arcs = setup_inference

    Msys = cpm.variable_elim([cpms[v] for v in varis.keys()], var_elim_order )
    np.testing.assert_array_almost_equal(Msys.C, np.array([[0, 1]]).T)
    np.testing.assert_array_almost_equal(Msys.p, np.array([[0.1018, 0.8982]]).T)

    pf_sys = Msys.p[0]
    assert pf_sys == pytest.approx(0.1018, rel=1.0e-3)


def test_inference2(setup_inference):

    # case 2: observation
    cpms, varis, var_elim_order, arcs = setup_inference

    cnd_vars = [f'o{i}' for i in range(1, len(arcs) + 1)]
    cnd_states = [1, 1, 0, 1, 0, 1]  # observing e3, e5 failure

    Mobs = cpm.condition([cpms[v] for v in varis.keys()], cnd_vars, cnd_states)
    Msys_obs = cpm.variable_elim(Mobs, var_elim_order)

    np.testing.assert_array_almost_equal(Msys_obs.C, np.array([[0, 1]]).T)
    np.testing.assert_array_almost_equal(Msys_obs.p, np.array([[2.765e-5, 5.515e-5]]).T)

    pf_sys = Msys_obs.p[0] / np.sum(Msys_obs.p)
    assert pf_sys == pytest.approx(0.334, rel=1.0e-3)

