import networkx as nx
import numpy as np
from pathlib import Path
import copy
import pandas as pd
import pickle
import random
import matplotlib.pyplot as plt
from BNS_JT import config

from BNS_JT import trans, branch, variable, cpm

od_pair, key = ('n1','n53'), 'ema2'

HOME = Path(__file__).parent
output_path = HOME.joinpath('./output')

cfg = config.Config(HOME.joinpath('./config.json'))
st_br_to_cs = {'f':0, 's':1, 'u': 2}
thres = 2

n_edge = len(cfg.infra['edges'])

### Load data ###
fout_cpm = output_path.joinpath(f'cpms_{key}.pk')
with open(fout_cpm, 'rb') as file:
    cpms = pickle.load(file)

fout_varis = output_path.joinpath(f'varis_{key}.pk')
with open(fout_varis, 'rb') as file:
    varis = pickle.load(file)

fout_br = output_path.joinpath(f'brs_{key}.pk')
with open(fout_br, 'rb') as file:
    brs = pickle.load(file)

comps_st_itc = {k: len(v.values) - 1 for k, v in varis.items()}
d_time_itc, path_itc = trans.get_time_and_path_given_comps(comps_st_itc, od_pair, cfg.infra['edges'], varis)


### Rejection sampling ###
known_prob = sum(b.p for b in brs if b.down_state == b.up_state and b.down_state != 'u' )
prob_pf = sum(b.p for b in brs if b.up_state == 'f' )

cpms2 = copy.deepcopy(cpms)
cpms2['sys'] = cpm.condition([cpms2['sys']], ['sys'], [varis['sys'].B.index({0,1})])[0]

# system function for rejection sampling
def sf_min_path_rs(comps_st, od_pair, arcs, vari, st_br_to_cs, thres=None):
    d_time, sys_st, _ = trans.sf_min_path(comps_st, od_pair, arcs, vari, thres)
    return d_time, st_br_to_cs[sys_st]

sys_fun_rs = lambda x: sf_min_path_rs(x, od_pair, cfg.infra['edges'], {'e'+str(e+1): varis['e'+str(e+1)] for e in range(n_edge)}, st_br_to_cs, thres * d_time_itc) 

cpms_rs, result = cpm.rejection_sampling_sys(cpms2, 'sys', sys_fun_rs, 0.05, sys_st_monitor = 0, known_prob = known_prob, sys_st_prob = prob_pf, rand_seed = 1)


### Save data ###
fout_cpm = output_path.joinpath(f'cpms_rs_{key}.pk')
with open(fout_cpm, 'wb') as fout:
    pickle.dump(cpms_rs, fout)

