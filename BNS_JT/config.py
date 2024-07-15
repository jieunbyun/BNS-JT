"""
Config module

"""
import copy
import logging
import configparser
import pandas as pd
import json
import networkx as nx
import graphviz as gv
from pathlib import Path

#from BNS_JT.utils import read_nodes


template_nodes = {
    "component_type": None,
    "component_class": None,
    "cost_fraction": None,
    "node_type": None,
    "node_cluster": None,
    "operating_capacity": None,
    "pos_x": None,
    "pos_y": None,
    "damages_states_constructor": None
    }

template_edges = {
    "origin": None,
    "destination": None,
    "link_capacity": None,
    "weight": None,
    }

graph_types = ['Graph',
               'DiGraph',
               'MultiGraph',
               'MultiDiGraph']


class Config(object):
    """
    """
    REQ = ['MODEL_NAME', 'MAX_BRANCHES', 'CONFIGURATION_ID', 'OUTPUT_PATH']

    def __init__(self, file_cfg):
        """
        :param file_cfg: config file
        """

        with open(file_cfg, 'r') as f:
            data = json.load(f)

        assert all(x in data for x in self.REQ), f'{self.REQ} should be defined in the config file'

        HOME = Path(file_cfg).absolute().parent

        # process required
        self.file_model = HOME.joinpath(data['MODEL_NAME'])
        assert self.file_model.exists(), f'{self.file_model} does not exist'

        self.infra = read_model_from_json(self.file_model)

        # scenario can be empty
        if data['SCENARIO_NAME']:
            self.file_scenarios = HOME.joinpath(data['SCENARIO_NAME'])
            assert self.file_scenarios.exists(), f'{file_scenarios} does not exist'
            self.scenarios = read_scenarios_from_json(self.file_scenarios)
            self.no_ds = len(self.scenarios['damage_states'])
            data.pop('SCENARIO_NAME')
        else:
            print(f'scenario to be added later')

        self.key = data['CONFIGURATION_ID']

        self.max_branches = data['MAX_BRANCHES']

        self.output_path = HOME.joinpath(data['OUTPUT_PATH'])
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True, exist_ok=True)
            print(f'{self.output_path} created')

        [data.pop(x) for x in self.REQ]
        self.data = data.copy()


def read_model_from_json(file_input):

    with open(file_input, 'r') as f:
        # ensure that damage states are ordered
        #model = json.load(f, object_pairs_hook=OrderedDict)
        model = json.load(f)

    # read node information
    nodes = {}
    for k, v in model['component_list'].items():
        nodes[k] = v

    edges = {}
    for k, v in model['node_conn_df'].items():
        edges[k] = v

    ODs={}
    for k, v in model['sysout_setup'].items():
        ODs[k] = (v['origin'], v['destination'])

    # create a graph
    try:
        gtype = model['graph_type']
    except KeyError:
        G = nx.Graph()
    else:
        assert gtype in graph_types, f'graph_type should be one of {graph_types}'
        if gtype.lower() == 'graph':
            G = nx.Graph()
        elif gtype.lower() == 'digraph':
            G = nx.DiGraph()
        elif gtype.lower() == 'multigraph':
            G = nx.MultiGraph()
        elif gtype.lower() == 'multidigraph':
            G = nx.MultiDiGraph()

    for k, v in edges.items():
        G.add_edge(v['origin'], v['destination'], label=k, weight=v['weight'], key=k)

    for k, v in nodes.items():
        G.add_node(k, pos=(v['pos_x'], v['pos_y']), label=k, key=k)

    # create length attribute and time 
    return {'G': G, 'nodes': nodes, 'edges': edges, 'ODs': ODs}


def read_scenarios_from_json(file_scenarios):

    with open(file_scenarios, 'r') as f:
        # ensure that damage states are ordered
        scenarios = json.load(f)

    return scenarios


def convert_csv_to_json(df, template):
    """
    df:
    template:
    """
    data = {}
    for k, v in df.iterrows():
        template.update(**v)
        data[str(k)] = copy.deepcopy(template)

    return data


def dict_to_json(dict_pf, damage_states, filename=None):
    """
    to create a json file for scenario
    dict_pf: dict of prob. of failiure
    damage_states: list of strings
    """

    assert isinstance(dict_pf, dict), 'dict_pf should be a dict'
    assert isinstance(damage_states, list), 'damage_states should be a list'

    no_scenarios = len(dict_pf[next(iter(dict_pf))])
    df = []
    for i in range(no_scenarios):
        tmp = [(v[i][0], 1-v[i][0]) for _, v in dict_pf.items()]
        df.append(tmp)

    df = pd.DataFrame(df).T
    df['index'] = dict_pf.keys()
    df = df.set_index('index')
    df = df.rename({k: f's{k+1}' for k in range(no_scenarios)}, axis=1)
    json_str = json.loads(df.to_json())

    if filename:
        with open(filename, 'w') as w:
            json.dump({'damage_states': damage_states,
                       'scenarios': json_str}, w, indent=4)
            print(f'{filename} is written')


def networkx_to_graphviz(g):
    """Convert `networkx` graph `g` to `graphviz.Digraph`.

    @type g: `networkx.Graph` or `networkx.DiGraph`
    @rtype: `graphviz.Digraph`
    """
    if g.is_directed():
        h = gv.Digraph()
    else:
        h = gv.Graph()
    for u, d in g.nodes(data=True):
        h.node(str(u), label=d['label'])
    for u, v, d in g.edges(data=True):
        h.edge(str(u), str(v), label=d['label'])
    return h


def plot_graphviz(G, outfile='graph'):

    h = networkx_to_graphviz(G)
    h.render(outfile, format='png', cleanup=True)
    print(f'{outfile}.png is created')






