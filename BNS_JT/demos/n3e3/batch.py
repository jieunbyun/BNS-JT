import typer
import pdb
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from BNS_JT import model, config, cpm, variable, trans, brc, branch

HOME = Path(__file__).parent

app = typer.Typer()



@app.command()
def main():

    cfg = config.Config(HOME.joinpath('./config.json'))
    """
    G = nx.MultiGraph()
    for k, x in cfg.infra['edges'].items():
        G.add_edge(x['origin'], x['destination'], weight=x['weight'], label=k)

    for k, v in cfg.infra['nodes'].items():
        G.add_node(k, pos=(v['pos_x'], v['pos_y']), label=k)
    """
    h = config.networkx_to_graphviz(cfg.infra['G'])
    outfile = HOME.joinpath('graph_n3e3')
    h.render(outfile, format='png', cleanup=True)

    probs = {'e1': {0: 0.1, 1: 0.9},
             'e2': {0: 0.2, 1: 0.8},
             'e3': {0: 0.3, 1: 0.7},
             }

    od_pair = ('n1', 'n3')
    st_br_to_cs = {'f': 0, 's': 1, 'u': 2}

    #comps_st_itc = {}
    varis = {}
    for k, v in cfg.infra['edges'].items():

        varis[k] = variable.Variable(name=k, values=['f', 's'])
        #cpms[k] = cpm.Cpm(variables = [varis[k]], no_child=1,
        #                  C = np.array([0, 1]).T, p = [probs[k], 1 - probs[k]])
        #comps_st_itc[k] = 1

    #d_time_itc, _ = get_time_and_path_multi_dest(comps_st_itc, od_pair[0], [od_pair[1]], cfg.infra['edges'], varis)
    sys_fun = trans.sys_fun_wrap(cfg.infra['G'], od_pair)

    brs, rules, sys_res, monitor = brc.run(
        varis, probs, sys_fun, cfg.data['MAX_SYS_FUN'], cfg.max_branches)

    csys, varis = brc.get_csys(brs, varis, st_br_to_cs)
    print(csys)

if __name__=='__main__':
    app()

