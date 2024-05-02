from pathlib import Path
import pandas as pd
import json
import networkx as nx
import matplotlib.pyplot as plt
import pdb
import numpy as np
import pickle
import typer

from BNS_JT import variable, gen_bnb
import config_mf

HOME = Path(__file__).parent
output_path = HOME.joinpath('./output')
output_path.mkdir(parents=True, exist_ok=True)


app = typer.Typer()
def get_max_flow(comps_st, od_pair, edges, varis, target_flow, is_bi_dir):

    if is_bi_dir:
        G = nx.Graph()
    else: 
        G = nx.DiGraph()

    first = next(iter(edges.items()))[1]
    if isinstance(first, (list, tuple)):
        for k, v in edges.items():
            G.add_edge(v[0], v[1], capacity=varis[k].values[comps_st[k]])
            
            
    elif isinstance(first, dict):
        for k, v in edges.items():
            G.add_edge(v['origin'], v['destination'], capacity=varis[k].values[comps_st[k]])

    G.add_edge( od_pair[1], 't_new', capacity=target_flow)
    mf, flow_dict = nx.maximum_flow(G, od_pair[0], 't_new')
    return mf, flow_dict

def max_flow_fun(comps_st, od_pair, edges, varis, target_flow, is_bi_dir=False):

    """
    # test
    comps_st = {k: 2 for k in varis.keys()}
    comps_st['e17'] = 0
    mf, sys_st, min_comps_st = max_flow_fun(comps_st, od_pair, edges, varis, target_flow, is_bi_dir)
    print(mf, sys_st, min_comps_st)
    """

    mf, flow_dict = get_max_flow(comps_st, od_pair, edges, varis, target_flow, is_bi_dir)
    if mf < target_flow:
        sys_st = 'f'
        min_comps_st = {}
    else:
        sys_st = 's'
        min_comps_st = {}
        for k, v in edges.items():
            if is_bi_dir:
                flow = max([flow_dict[v[0]][v[1]], flow_dict[v[1]][v[0]]])
            else:
                flow = flow_dict[v[0]][v[1]]
            if flow > 0:
                k_st = next((i for i, x in enumerate(varis[k].values) if x >= flow), None)
                assert k_st is not None, f'flow {flow} is not in {varis[k].values}'
                min_comps_st[k] = k_st
    
    return mf, sys_st, min_comps_st

def init_prob( cfg_fname, od_name ):

    cfg = config_mf.Config_mf(HOME.joinpath('./input/'+cfg_fname))

    # variables
    varis = {}
    for k, v in cfg.infra['edges'].items():
        varis[k] = variable.Variable(name=k, values = v['link_capacity'])

    edges = {}
    for k, v in cfg.infra['edges'].items():
        edges[k] = (v['origin'], v['destination'])

    od_pair = (cfg.infra['ODs'][od_name]['pair'][0], cfg.infra['ODs'][od_name]['pair'][1])
    target_flow = cfg.infra['ODs'][od_name]['target_flow']
    is_bi_dir = cfg.infra['ODs'][od_name]['is_bi_dir']

    probs = {}
    for k, v in cfg.infra['edges'].items():
        prob_ = {}
        for st, v2 in enumerate(v['prob']):
            prob_[st] = v2
        probs[k] = prob_

    # system function
    sys_fun = lambda x: max_flow_fun(x, od_pair, edges, varis, target_flow, is_bi_dir)

    return cfg, varis, edges, od_pair, target_flow, is_bi_dir, probs, sys_fun


@app.command()
def run_MCS(cfg_fname, od_name):
        
        cfg, _, _, _, _, _, probs, sys_fun = init_prob( cfg_fname, od_name )

        key = cfg.infra['ODs'][od_name]['key']
        pf, cov, nsamp = gen_bnb.run_MCS_indep_comps(probs, sys_fun, cov_t=0.01)
        print(f'pf: {pf:.4e}, cov: {cov:.4e}, nsamp: {nsamp:d}')

        with open(output_path.joinpath(f'{key}_mcs.txt'), 'w') as fout:
            fout.write(f"pf: {pf:.4e} \n")
            fout.write(f"cov: {cov:.4e} \n")
            fout.write(f"no_samples: {nsamp:d}")


@app.command()
def main(cfg_fname, od_name):

    #cfg_fname = 'net1_config.json'
    #od_name = 'od1'

    cfg, varis, _, _, _, _, probs, sys_fun = init_prob( cfg_fname, od_name )

    # run BRC
    brs, rules, sys_res, monitor = gen_bnb.run_brc(varis, probs, sys_fun, max_sf = cfg.max_sys_fun, max_nb = 0.01*cfg.max_branches, pf_bnd_wr = cfg.sys_bnd_wr, surv_first=False, rules=None)

    if monitor['pf_low'][-1] * cfg.sys_bnd_wr < monitor['pf_up'][-1] - monitor['pf_low'][-1]:
        brs, rules, sys_res2, monitor2 = gen_bnb.run_brc(varis, probs, sys_fun, max_sf = cfg.max_sys_fun, max_nb = cfg.max_branches, pf_bnd_wr = cfg.sys_bnd_wr, surv_first=True, rules=rules)
        sys_res = pd.concat([sys_res, sys_res2], ignore_index=True)
        for k, v in monitor.items():
            monitor[k] += monitor2[k]

    # Store result
    ### Data Store ###
    key = cfg.infra['ODs'][od_name]['key']
    fout_br = output_path.joinpath(f'{key}_brs.pk')
    with open(fout_br, 'wb') as fout:
        pickle.dump(brs, fout)

    fout_varis = output_path.joinpath(f'{key}_varis.pk')
    with open(fout_varis, 'wb') as fout:
        pickle.dump(varis, fout)

    fout_rules = output_path.joinpath(f'{key}_rules.pk')
    with open(fout_rules, 'wb') as fout:
        pickle.dump(rules, fout)

    fout_monitor = output_path.joinpath(f'{key}_monitor.pk')
    with open(fout_monitor, 'wb') as fout:
        pickle.dump(monitor, fout)

    print(f"{key} done. Output files saved")

#if __name__=='__main__':
#    app()


cfg_fname = 'config.json'
od_name = 'od1'

"""
cfg, varis, edges, od_pair, target_flow, is_bi_dir, probs, sys_fun = init_prob( cfg_fname, od_name )

# case 1
comps_st = {k: 0 for k in varis.keys()}
for e in ['e1', 'e10', 'e15', 'e17', 'e20']:
    comps_st[e] = 1
mf, sys_st, min_comps_st = max_flow_fun(comps_st, od_pair, edges, varis, target_flow, is_bi_dir)
print(mf, sys_st, min_comps_st)
"""


for i in range(5):
    od_name = f'od{i+1}'
    main(cfg_fname, od_name)
    run_MCS(cfg_fname, od_name)
