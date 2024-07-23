from pathlib import Path
import pandas as pd
import json
import networkx as nx
import matplotlib.pyplot as plt
import pdb
import numpy as np
import typer


from BNS_JT import model, config, trans, variable, cpm, operation, brc


HOME = Path(__file__).parent

app = typer.Typer()


@app.command()
def create_model():

    # nodes
    nodes = pd.DataFrame([['source', -2.0, 0.0],
                      ['x1', 0.0, 0.5],
                      ['x2', 0.0, 0.25],
                      ['x3', 0.0, 0.0],
                      ['x4', -1.0, -0.5],
                      ['x5', 0.0, -0.5],
                      ['x6', 1.0, -0.5],
                      ['x7', 2.0, 0.0],
                      ['x8', 3.0, 0.0],
                      ['sink', 4.0, 0.0]],
                      columns=['node', 'pos_x', 'pos_y'])
    nodes = nodes.set_index('node')
    nodes_coords = {k: (v['pos_x'], v['pos_y']) for k, v in nodes.iterrows()}

    # arcs
    df = pd.DataFrame([['e1', 'source', 'x1'],
        ['e2', 'source', 'x2'],
        ['e3', 'source', 'x3'],
        ['e4', 'source', 'x4'],
        ['e5', 'x4', 'x5'],
        ['e6', 'x5', 'x6'],
        ['e7', 'x6', 'x7'],
        ['e8', 'x7', 'x8'],
        ['e9', 'x8', 'sink'],
        ['e10', 'x1', 'x7'],
        ['e11', 'x2', 'x7'],
        ['e12', 'x3', 'x7'],
        ], columns=['edge', 'origin', 'destination'])

    df = df.set_index('edge')
    arcs = {k: [v["origin"], v["destination"]] for k, v in df.iterrows()}

    ODs = {'od1': ('source', 'sink')}

    outfile = HOME.joinpath('./model.json')
    dic_model = trans.create_model_json_for_graph_network(arcs, nodes_coords, ODs, outfile)
    """
    # scenarios
    damage_states = ['s', 'f']
    delay_rate = [10, 2, 1]
    s1 = {'s1': {}}

    for k in arcs.keys():
        s1['s1'][k] = [x * dic_model['node_conn_df'][k]['weight'] for x in delay_rate]

    #wfile = HOME.joinpath('./scenarios_road.json')
    #_dic = trans.create_scenario_json_for_trans_network(damage_states, s1, wfile)
    """

@app.command()
def plot():

    cfg = config.Config(HOME.joinpath('./config.json'))

    trans.plot_graph(cfg.infra['G'], HOME.joinpath('rbd.png'))


@app.command()
def main():

    cfg = config.Config(HOME.joinpath('./config.json'))

    st_br_to_cs = {'f':0, 's':1, 'u': 2}
    od_pair = ('source', 'sink')

    # ignoring source and sink which are always intact
    nodes_except_const = list(cfg.infra['nodes'].keys())
    [nodes_except_const.remove(x) for x in od_pair]

    varis = {}
    cpms = {}
    probs = {}
    for k in nodes_except_const:
        varis[k] = variable.Variable(name=k, values=['f', 's'])

        cpms[k] = cpm.Cpm(variables = [varis[k]], no_child=1,
                          C = np.array([0, 1]).T, p = [0.1, 0.9])
        probs[k] = [0.5, 0.5]

    sys_fun = trans.sys_fun_wrap(cfg.infra['G'], od_pair, varis)
    brs, rules, sys_res, _ = brc.run(varis, probs, sys_fun, max_sf=cfg.max_branches, max_nb=cfg.max_branches)

    csys_by_od, varis_by_od = brc.get_csys(brs, varis, st_br_to_cs)

    varis['sys'] = variable.Variable(name='sys', values=['f', 's'])
    cpms['sys'] = cpm.Cpm(variables = [varis[k] for k in ['sys'] + nodes_except_const],
                          no_child = 1,
                          C = csys_by_od.copy(),
                          p = np.ones(csys_by_od.shape[0]))

    """
    [[0 2 2 2 2 2 2 0 2]
     [0 2 2 2 2 2 2 1 0]
     [0 0 0 0 0 2 2 1 1]
     [0 0 0 0 1 0 2 1 1]
     [0 0 0 0 1 1 0 1 1]
     [1 0 0 0 1 1 1 1 1]
     [1 0 0 1 2 2 2 1 1]
     [1 0 1 2 2 2 2 1 1]
     [1 1 2 2 2 2 2 1 1]]
    """

    # inference 
    M = [cpms[k] for k in nodes_except_const + ['sys']]
    var_elim = nodes_except_const[:]
    var_elim.remove('x1')
    var_elim = [varis[k] for k in var_elim]
    M_VE = operation.variable_elim(M, var_elim)

    # compute P(x1=1|sys=1) = P(x1, sys) / P(sys)
    pf_sys1 = M_VE.p[M_VE.C[:, 0]==1].sum()
    pf_x1_sys1 = M_VE.p[(M_VE.C == [1, 1]).all(axis=1)].sum()
    pf_x1Gsys1 = pf_x1_sys1/pf_sys1

    print(pf_x1Gsys1)

if __name__=='__main__':

    app()

