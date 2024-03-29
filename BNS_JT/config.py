"""
Config module

"""
import copy
import logging
import configparser
import pandas as pd
import json
import networkx as nx

from pathlib import Path
from BNS_JT.utils import read_nodes


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


class Config(object):
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
        self.infra = read_model_from_json(file_model)

        # scenario can be empty
        if cfg['SCENARIO_NAME']:

            file_scenarios = HOME.joinpath(cfg['SCENARIO_NAME'])
            assert file_scenarios.exists(), f'{file_scenarios} does not exist'
            self.scenarios = read_scenarios_from_json(file_scenarios)
            self.no_ds = len(self.scenarios['damage_states'])

        else:
            print(f'scenario to be added later')

        self.key = cfg['CONFIGURATION_ID']

        self.max_branches = cfg['MAX_BRANCHES']

        self.output_path = HOME.joinpath(cfg['OUTPUT_PATH'])
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True, exist_ok=True)
            print(f'{self.output_path} created')


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
    G = nx.Graph()
    for k, v in edges.items():
        G.add_edge(v['origin'], v['destination'], label=k, time=v['weight'])

    for k, v in nodes.items():
        G.add_node(k, pos=(v['pos_x'], v['pos_y']))

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




