import json
import networkx as nx

from pathlib import Path

class Config_mf(object):
    """
    """
    def __init__(self, file_cfg):
        """
        :param file_cfg: config file
        """

        with open(file_cfg, 'r') as f:
            cfg = json.load(f)

        HOME = Path(file_cfg).absolute().parent

        file_model = HOME.joinpath(cfg['MODEL_NAME'])
        assert file_model.exists(), f'{file_model} does not exist'
        self.infra = read_mf_model_from_json(file_model)

        self.max_branches = cfg['MAX_BRANCHES']
        if cfg['MAX_SYS_FUN']:
            self.max_sys_fun = cfg['MAX_SYS_FUN']
        else:
            self.max_sys_fun = float('inf')
        self.sys_bnd_wr = cfg['SYS_BND_WIDTH_RATIO'] 


def read_mf_model_from_json(file_input):

    with open(file_input, 'r') as f:
        # ensure that damage states are ordered
        #model = json.load(f, object_pairs_hook=OrderedDict)
        model = json.load(f)

    # read node information
    nodes = {}
    for k, v in model['node_list'].items():
        nodes[k] = v

    edges = {}
    for k, v in model['edge_list'].items():
        edges[k] = v

    ODs={}
    for k, v in model['system'].items():
        ODs[k] = {"pair": (v['origin'], v['destination']), "target_flow": v["target_flow"], "is_bi_dir": v["edge_bidirectional"], "key": v['key']}

    # create a graph
    G = nx.Graph()
    for k, v in edges.items():
        G.add_edge(v['origin'], v['destination'])

    for k, v in nodes.items():
        G.add_node(k, pos=(v['pos_x'], v['pos_y']))

    # create length attribute and time 
    return {'G': G, 'nodes': nodes, 'edges': edges, 'ODs': ODs}