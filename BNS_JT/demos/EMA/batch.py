from pathlib import Path
import pandas as pd
import json
import networkx as nx
import matplotlib.pyplot as plt
import pdb
import numpy as np

from BNS_JT import model, config, trans, variable, gen_bnb


HOME = Path(__file__).parent


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

    ODs = {'od1': ('n1', 'n3')}

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


def main():

    cfg = config.Config(HOME.joinpath('./config.json'))

    csys_by_od, varis_by_od = model.get_branches_by_od(cfg)

    print(csys_by_od)

    print(varis_by_od)

if __name__=='__main__':

    #create_model()
    main()

