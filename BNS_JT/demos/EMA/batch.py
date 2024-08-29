from pathlib import Path
import pandas as pd
import json
import networkx as nx
import matplotlib.pyplot as plt
import pdb
import pickle
import numpy as np
import typer
from typing_extensions import Annotated

from BNS_JT import model, config, trans, variable, brc, branch, cpm


HOME = Path(__file__).parent

app = typer.Typer()

@app.command()
def create_model():

    # nodes
    file_node_coord = HOME.joinpath('ema_node_km.txt')
    nodes = pd.read_csv(file_node_coord, header=None, delimiter='\t')
    nodes = nodes.rename({0: 'node', 1: 'pos_x', 2: 'pos_y'}, axis=1)
    nodes['node'] = nodes['node'].apply(lambda x: f'n{x}')
    nodes = nodes.set_index('node')
    nodes_coords = {k: (v['pos_x'], v['pos_y']) for k, v in nodes.iterrows()}

    # arcs
    file_edge = HOME.joinpath('EMA_net.txt')
    df = pd.read_csv(file_edge, header=None, delimiter='\t')
    df = df.rename({0: 'origin', 1: 'destination'}, axis=1)
    df['edge'] = df.apply(lambda x: f'e{x.name + 1}', axis=1)
    df = df.set_index('edge')
    arcs = {k: [f'n{v["origin"]}', f'n{v["destination"]}'] for k, v in df.iterrows()}

    arcs_avg_kmh = {x: 100 for x in df.index}

    ODs = {'od1': ('n22', 'n53')}

    outfile = HOME.joinpath('./model.json')
    dic_model = trans.create_model_json_for_tranportation_network(arcs, nodes_coords, arcs_avg_kmh, ODs, outfile)

    # scenarios
    damage_states = ['ds1', 'ds2', 'ds3']
    delay_rate = [10, 2, 1]
    s1 = {'s1': {}}

    for k in arcs.keys():
        s1['s1'][k] = [x * dic_model['node_conn_df'][k]['weight'] for x in delay_rate]

    wfile = HOME.joinpath('./scenarios.json')
    _dic = trans.create_scenario_json_for_trans_network(damage_states, s1, wfile)


@app.command()
def main(key: Annotated[str, typer.Argument()] = 'od2',
         max_sf: Annotated[int, typer.Argument()] = 100):

    rnd_state = np.random.RandomState(1)

    cfg = config.Config(HOME.joinpath('./config.json'))

    od_pair = cfg.infra['ODs'][key]

    # randomly assign probabilities
    probs_set = {0: {0: 0.01, 1: 0.04, 2: 0.95},
                 1: {0: 0.03, 1: 0.12, 2: 0.85},
                 2: {0: 0.06, 1: 0.24, 2: 0.70},
                 }
    probs_key = rnd_state.choice(3, size=len(cfg.infra['edges']))

    probs = {k: probs_set[v] for k, v in zip(cfg.infra['edges'].keys(), probs_key)}

    # variables
    varis = {}
    cpms = {}
    for k, v in cfg.infra['edges'].items():
        varis[k] = variable.Variable(name=k, values = cfg.scenarios['scenarios']['s1'][k])
        cpms[k] = cpm.Cpm(variables = [varis[k]], no_child=1,
                          C = np.array([0, 1, 2]).T, p = [probs[k][0], probs[k][1], probs[k][2]])

    # Intact state of component vector: zero-based index
    comps_st_itc = {k: len(v.values) - 1 for k, v in varis.items()} # intact state (i.e. the highest state)

    thres = 2
    st_br_to_cs = {'f': 0, 's': 1, 'u': 2}
    #csys_by_od, varis_by_od = {}, {}
    # branches by od_pair
    #for k, od_pair in cfg.infra['ODs'].items():

    d_time_itc, _, path_itc = trans.get_time_and_path_given_comps(comps_st_itc, cfg.infra['G'], od_pair, varis)

    # system function
    sys_fun = trans.sys_fun_wrap(cfg.infra['G'], od_pair, varis, thres * d_time_itc)
    #brs, rules, _ = gen_bnb.proposed_branch_and_bound(sys_fun, varis, max_br=cfg.max_branches, output_path=cfg.output_path, key=f'EMA_{od_pair}', flag=True)
    brs1, rules1, sys_rules1, monitor1 = brc.run(varis, probs, sys_fun, max_sf=max_sf, max_nb=10_000, surv_first=True)
    brs2, rules2, sys_rules2, monitor2 = brc.run(varis, probs, sys_fun, max_sf=max_sf, max_nb=50_000, surv_first=True, rules=rules1)

    csys, varis = brc.get_csys(brs2, varis, st_br_to_cs)

    varis['sys'] = variable.Variable(name='sys', values=list(st_br_to_cs.keys()))
    cpms['sys'] = cpm.Cpm(variables = [varis['sys']] + [varis[k] for k in cfg.infra['edges'].keys()], no_child=1, C=csys, p=np.ones((len(csys),1), dtype=np.float32))

    ### Data Store ###
    fout_br = cfg.output_path.joinpath(f'brs_{key}.pk')
    with open(fout_br, 'wb') as fout:
        pickle.dump(brs2, fout)

    fout_cpm = cfg.output_path.joinpath(f'cpms_{key}.pk')
    with open(fout_cpm, 'wb') as fout:
        pickle.dump(cpms, fout)

    fout_varis = cfg.output_path.joinpath(f'varis_{key}.pk')
    with open(fout_varis, 'wb') as fout:
        pickle.dump(varis, fout)

    fout_rules = cfg.output_path.joinpath(f'rules_{key}.pk')
    with open(fout_rules, 'wb') as fout:
        pickle.dump(rules2, fout)

    monitor = {}
    for k in monitor2.keys():
        monitor[k] = monitor1[k] + monitor2[k]

    fout_monitor = cfg.output_path.joinpath(f'monitor_{key}.pk')
    with open(fout_monitor, 'wb') as fout:
        pickle.dump(monitor, fout)


if __name__=='__main__':

    #create_model()
    #main()
    app()

