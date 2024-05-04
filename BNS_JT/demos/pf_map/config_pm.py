import copy
import logging
import configparser
import pandas as pd
import json
import networkx as nx
import numpy as np

from pathlib import Path
from BNS_JT.utils import read_nodes

class Config_pm(object):

    def __init__(self, cfg_fname):

        with open(cfg_fname, 'r') as f:
            cfg = json.load(f)

        HOME = Path(cfg_fname).absolute().parent

        file_model = HOME.joinpath(cfg['MODEL_NAME'])
        assert file_model.exists(), f"{file_model} does not exist"
        self.infra = read_model_from_json(file_model)

        self.max_branches = cfg['MAX_BRANCHES']
        if cfg['MAX_SYS_FUN']:
            self.max_sys_fun = cfg['MAX_SYS_FUN']
        else:
            self.max_sys_fun = float('inf')
        self.sys_bnd_wr = cfg['SYS_BND_WIDTH_RATIO']

        self.cov_t = cfg['MCS_COV']

        self.key = cfg['KEY']


def read_model_from_json( file_input ):

    with open(file_input, 'r') as f:
        model = json.load(f)

    # read node information
    nodes = {}
    for k, v in model['node_list'].items():
        nodes[k] = v

    edges = {}
    for k, v in model['edge_list'].items():
        edges[k] = v

    origins = []
    for k, v in model['origin_list'].items():
        origins += [v['ID']]

    eq = {}
    for k, v in model['eq_scenario'].items():
        eq[k] = v

    frag = {}
    for k, v in model['fragility_data'].items():
        frag[k] = v
    
    thres = model['system']['delay_thres']

    return {'nodes': nodes, 'edges': edges, 'origins': origins, 'eq': eq, 'frag': frag, 'thres': thres}



    
    


