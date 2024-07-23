from pathlib import Path
import pandas as pd
import json
import networkx as nx
import matplotlib.pyplot as plt
import pdb
import numpy as np

from BNS_JT import model, config, trans, variable


HOME = Path(__file__).parent


def create_model():

    # nodes
    nodes = pd.DataFrame([['n1', -2.0, 3.0],
                      ['n2', -2.0, -3.0],
                      ['n3', 2.0, -2.0],
                      ['n4', 1.0, 1.0],
                      ['n5', 0.0, 0.0]],
                      columns=['node', 'pos_x', 'pos_y'])
    #df = pd.read_csv(file_node_coord, header=None, delimiter='\t')
    #df = df.rename({0: 'node', 1: 'pos_x', 2: 'pos_y'}, axis=1)
    #df['node'] = df['node'].apply(lambda x: f'n{x}')
    nodes = nodes.set_index('node')
    nodes_coords = {k: (v['pos_x'], v['pos_y']) for k, v in nodes.iterrows()}

    # arcs
    df = pd.DataFrame([['e1', 'n1', 'n2'],
        ['e2', 'n1', 'n5'],
        ['e3', 'n2', 'n5'],
        ['e4', 'n3', 'n4'],
        ['e5', 'n3', 'n5'],
        ['e6', 'n4', 'n5']], columns=['edge', 'origin', 'destination'])

    df = df.set_index('edge')
    arcs = {k: [v["origin"], v["destination"]] for k, v in df.iterrows()}

    arcs_avg_kmh = {'e1': 40,
                    'e2': 40,
                    'e3': 40,
                    'e4': 30,
                    'e5': 30,
                    'e6': 20}

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

