#!/usr/bin/env python
import networkx as nx
import numpy as np
from pathlib import Path
import copy

from BNS_JT import trans, branch, variable, cpm, gen_bnb, config

HOME = Path(__file__).parent

def setup():
    # # Illustrative example: Routine
    # Network
    node_coords = {'n1': (0, 0),
                   'n2': (1, 1),
                   'n3': (1, -1),
                   'n4': (2, 0)}

    arcs = {'e1': ['n1', 'n2'],
            'e2': ['n1', 'n3'],
            'e3': ['n2', 'n3'],
            'e4': ['n2', 'n4'],
            'e5': ['n3', 'n4']}
    n_arc = len(arcs)

    probs = {'e1': {0: 0.01, 1:0.99}, 'e2': {0:0.02, 1:0.98}, 'e3': {0:0.03, 1:0.97}, 'e4': {0:0.04, 1:0.96}, 'e5': {0:0.05, 1:0.95}}

    od_pair=('n1','n4')

    ODs = {'od1': od_pair}

    outfile = HOME.joinpath('./model.json')
    dic_model = trans.create_model_json_for_graph_network(arcs, node_coords, ODs, outfile)


def plot():

    cfg = config.Config(HOME.joinpath('./config.json'))

    trans.plot_graph(cfg.infra['G'], HOME.joinpath('routine.png'))



def conn(comps_st, od_pair, arcs): # connectivity analysis
    G = nx.Graph()
    for k,x in comps_st.items():
        if x > 0:
            G.add_edge(arcs[k][0], arcs[k][1], capacity=1)


    if od_pair[0] in G.nodes and od_pair[1] in G.nodes:
        f_val, _ = nx.maximum_flow(G,od_pair[0],od_pair[1])
    else:
        f_val = 0

    if f_val > 0:
        sys_st = 's'

        p = nx.shortest_path( G, od_pair[0], od_pair[1] )

        min_comps_st = {}
        for i in range(len(p)-1):
            pair = [p[i], p[i+1]]
            if pair in arcs.values():
                a = list(arcs.keys())[list(arcs.values()).index(pair)]
            else:
                a = list(arcs.keys())[list(arcs.values()).index([pair[1], pair[0]])]
            min_comps_st[a] = 1

    else:
        sys_st = 'f'
        min_comps_st = None

    return f_val, sys_st, min_comps_st


def main(max_br):

    cfg = config.Config(HOME.joinpath('./config.json'))

    st_br_to_cs = {'f':0, 's':1, 'u': 2}

    od_pair = cfg.infra['ODs']['od1']

    probs = {'e1': {0: 0.01, 1:0.99}, 'e2': {0:0.02, 1:0.98}, 'e3': {0:0.03, 1:0.97}, 'e4': {0:0.04, 1:0.96}, 'e5': {0:0.05, 1:0.95}}

    varis = {}
    cpms = {}
    for k in cfg.infra['edges'].keys():
        varis[k] = variable.Variable(name=k, values=['f', 's'])

        cpms[k] = cpm.Cpm(variables = [varis[k]], no_child=1,
                          C = np.array([0, 1]).T, p = [probs[k][0], probs[k][1]])

    #sys_fun = lambda comps_st : conn(comps_st, od_pair, arcs)
    sys_fun = trans.sys_fun_wrap(od_pair, cfg.infra['edges'], varis)

    brs, rules, sys_res, monitor = gen_bnb.proposed_branch_and_bound_using_probs(
            sys_fun, varis, probs, max_br=max_br, output_path=HOME, key='routine')


    gen_bnb.plot_monitoring(monitor, HOME.joinpath('./monitor.png'))

    csys_by_od, varis_by_od = gen_bnb.get_csys_from_brs(brs, varis, st_br_to_cs)

    varis['sys'] = variable.Variable(name='sys', values=['f', 's', 'u'])
    cpms['sys'] = cpm.Cpm(variables = [varis[k] for k in ['sys'] + list(cfg.infra['edges'].keys())],
                          no_child = 1,
                          C = csys_by_od.copy(),
                          p = np.ones(csys_by_od.shape[0]))

    cpms_comps = {k: cpms[k] for k in cfg.infra['edges'].keys()}

    cpms_new = cpm.prod_cpm_sys_and_comps(cpms['sys'], cpms_comps, varis)

    p_f = cpm.get_prob(cpms_new, ['sys'], [0])
    p_s = cpm.get_prob(cpms_new, ['sys'], [1])

    print(f'failure prob: {p_f:.5f}, survival prob: {p_s:.5f}')

    return cpms, varis

