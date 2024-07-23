import networkx as nx
import numpy as np
from pathlib import Path
import copy
import pandas as pd
import pickle
import random
import matplotlib.pyplot as plt
import typer


from BNS_JT import config
from BNS_JT import trans, branch, variable, cpm

HOME = Path(__file__).parent
output_path = HOME.joinpath('./output')
output_path.mkdir(parents=True, exist_ok=True)

app = typer.Typer()

@app.command()
def main():

    random.seed(1)

    ### Problem definition ###
    cfg = config.Config(HOME.joinpath('./config.json'))
    st_br_to_cs = {'f':0, 's':1, 'u': 2}
    #od_pair, key = ('n1', 'n14'), 'ema1'
    od_pair, key = ('n1','n53'), 'ema2'

    n_edge = len(cfg.infra['edges'])

    integers = list(range(n_edge))
    random.shuffle(integers)
    group_size = len(integers) // 3

    # Randomly assign probabilities
    prob_groups = [sorted(integers[:group_size]), sorted(integers[group_size:2*group_size]), sorted(integers[2*group_size:])]
    probs_setting = [{0:0.01, 1: 0.04, 2: 0.95}, {0:0.03, 1: 0.12, 2: 0.85}, {0:0.06, 1: 0.24, 2: 0.70}]

    probs = {}
    for i in range(n_edge):
        g_idx = next(index for index, group in enumerate(prob_groups) if i in group)
        probs['e'+str(i+1)] = probs_setting[g_idx]


    varis = {}
    cpms = {}
    for k in cfg.infra['edges'].keys():
        varis[k] = variable.Variable(name=k, values = cfg.scenarios['scenarios']['s1'][k])
        cpms[k] = cpm.Cpm(variables = [varis[k]], no_child=1,
                            C = np.array([0, 1, 2]).T, p = [probs[k][0], probs[k][1], probs[k][2]])


    sys_fun = trans.sys_fun_wrap(od_pair, cfg.infra['edges'], varis)

    thres = 2
    st_br_to_cs = {'f': 0, 's': 1, 'u': 2}

    #csys_by_od, varis_by_od = {}, {}
    # branches by od_pair
    #for k, od_pair in cfg.infra['ODs'].items():
    comps_st_itc = {k: len(v.values) - 1 for k, v in varis.items()}
    d_time_itc, _, path_itc = trans.get_time_and_path_given_comps(comps_st_itc, cfg.infra['G'], cfg.infra['ODs']['od2'], varis)

    # system function
    sys_fun = trans.sys_fun_wrap(cfg.infra['G'], cfg.infra['ODs']['od2'], varis, thres * d_time_itc)


    ### BRC algorithm ###
    brs1, rules1, sys_res1, monitor1 = brc.run(varis, probs, sys_fun, max_sf=100, max_nb=10000, surv_first=True)
    brs2, rules2, sys_res2, monitor2 = brc.run(varis, probs, sys_fun, max_sf=100, max_nb=50000, surv_first=False, rules=rules1)


    csys, varis = brc.get_csys(brs2, varis, st_br_to_cs)

    varis['sys'] = variable.Variable( name='sys', values=['f', 's', 'u'] )
    cpms['sys'] = cpm.Cpm(variables = [varis['sys']]+[varis['e'+str(i+1)] for i in range(n_edge)], no_child=1, C=csys, p=np.ones((len(csys),1), dtype=np.float32))

    ### Data Store ###
    fout_br = output_path.joinpath(f'brs_{key}.pk')
    with open(fout_br, 'wb') as fout:
        pickle.dump(brs2, fout)


    fout_cpm = output_path.joinpath(f'cpms_{key}.pk')
    with open(fout_cpm, 'wb') as fout:
        pickle.dump(cpms, fout)

    fout_varis = output_path.joinpath(f'varis_{key}.pk')
    with open(fout_varis, 'wb') as fout:
        pickle.dump(varis, fout)

    fout_rules = output_path.joinpath(f'rules_{key}.pk')
    with open(fout_rules, 'wb') as fout:
        pickle.dump(rules2, fout)

    monitor = {}
    for k in monitor2.keys():
        monitor[k] = monitor1[k] + monitor2[k]

    fout_monitor = output_path.joinpath(f'monitor_{key}.pk')
    with open(fout_monitor, 'wb') as fout:
        pickle.dump(monitor, fout)

if __name__=='__main__':

    app()
