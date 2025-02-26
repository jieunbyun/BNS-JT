from pathlib import Path
import pandas as pd
import json
import networkx as nx
import matplotlib.pyplot as plt
import pdb
import geopandas as gpd
import pickle
import numpy as np
import typer
from collections import namedtuple
import shapely
import polyline
from scipy import stats
from itertools import islice
from typing_extensions import Annotated

from BNS_JT import model, config, trans, variable, brc, branch, cpm, operation

app = typer.Typer()

HOME = Path(__file__).parent

DAMAGE_STATES = ['Slight', 'Moderate', 'Extensive', 'Complete']

KEYS = ['LATITUDE', 'LONGITUDE', 'HAZUS_CLASS', 'K3d', 'Kskew', 'Ishape']

GDA94 = 'EPSG:4283'  # GDA94
GDA94_ = 'EPSG:3577'

Route = namedtuple('route', ['path', 'edges', 'weight'])


def k_shortest_paths(G, source, target, k, weight=None):
    return list(
        islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
    )


def update_nodes_given_dmg(file_dmg: str, nodes: dict, valid_paths) -> None:

    # selected nodes
    sel_nodes = [x for _, v in valid_paths.items() for x in v['path']]
    to_be_removed = set(nodes.keys()).difference(sel_nodes)
    [nodes.pop(k) for k in to_be_removed]

    # assign cpms given scenario
    if file_dmg.endswith('csv'):
        probs = pd.read_csv(file_dmg, index_col=0)
        probs.index = probs.index.astype('str')

    elif file_dmg.endswith('json'):
        with open(file_dmg, 'rb') as f:
            tmp = json.load(f)

        probs = {}
        for item in tmp['features']:
            probs[item['properties']['name']] = {
                    'Slight': item['properties']['shaking']['green'],
                    'Moderate': item['properties']['shaking']['yellow'],
                    'Extensive': item['properties']['shaking']['orange'],
                    'Complete': item['properties']['shaking']['red'],
                    }

        # from percent 
        probs = pd.DataFrame.from_dict(probs).T
        if probs.max().max() > 1:
            print(f'Probability should be 0 to 1 scale: {probs.max().max():.2f}')
            probs *= 0.01

    # failiure defined 
    probs['failure'] = probs['Extensive'] + probs['Complete']
    probs = probs.to_dict('index')

    for k, v in nodes.items():
        try:
            v['failure'] = probs[k]['failure']

        except KeyError:
            v['failure'] = 0.0


def create_shpfile_nodes(nodes: dict, outfile: str) -> None:
    """


    """
    keys = ['pos_x', 'pos_y', 'failure']
    _dic = {}
    for k, v in nodes.items():
        _dic[k] = {d: v[d] for d in keys}

    df = pd.DataFrame.from_dict(_dic).T
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df.pos_x, y=df.pos_y), crs=GDA94)
    df.to_file(outfile)
    print(f'{outfile} is written')


def create_shpfile(json_file: str, prob_by_path: dict, outfile: str) -> None:
    """


    """
    with open(json_file, 'r') as f:
        directions_result = json.load(f)

    keys = sorted(prob_by_path.keys())

    _dic = {}
    for k, item in zip(keys, directions_result):
        poly_line = item['overview_polyline']['points']
        geometry_points = [(x[1], x[0]) for x in polyline.decode(poly_line)]

        distance_km = sum([x['distance']['value']  for x in item['legs']]) / 1000.0
        duration_mins = sum([x['duration']['value']  for x in item['legs']]) / 60.0

        _dic[k] = {'dist_km': distance_km,
                   'duration_m': duration_mins,
                   'line': shapely.LineString(geometry_points),
                   'id': k,
                   'prob': prob_by_path.get(k, 0)
                   }

    df = pd.DataFrame.from_dict(_dic).T
    df = gpd.GeoDataFrame(df, geometry='line', crs= GDA94)
    df.to_file(outfile)
    print(f'{outfile} is written')


def get_kshape(Ishape, sa03, sa10):
    """
    sa03, sa10 can be vector
    """
    if isinstance(sa03, float):
        if Ishape:
            Kshape = np.min([1.0, 2.5*sa10/sa03])
        else:
            Kshape  = 1.0
    else:
        if Ishape:
            Kshape = np.minimum(1.0, 2.5*sa10/sa03)
        else:
            Kshape  = np.ones_like(sa03)
    return Kshape


def compute_pe_by_ds(ps, kshape, Kskew, k3d, sa10):

    factor = {'Slight': kshape,
              'Moderate': Kskew*k3d,
              'Extensive': Kskew*k3d,
              'Complete': Kskew*k3d}

    return pd.Series({ds: stats.lognorm.cdf(sa10, 0.6, scale=factor[ds]*ps[ds])
            for ds in DAMAGE_STATES})


@app.command()
def dmg_bridge(file_bridge: str, file_gm: str):

    # HAZUS Methodology
    bridge_param = pd.read_csv(
        HOME.joinpath('bridge_classification_damage_params.csv'), index_col=0)

    # read sitedb data 
    df_bridge = pd.read_csv(file_bridge, index_col=0)[KEYS].copy()

    # read gm
    gm = pd.read_csv(file_gm, index_col=0, skiprows=1)

    # weighted
    dmg = []
    for i, row in df_bridge.iterrows():
        _df = (gm['lat']-row['LATITUDE'])**2 + (gm['lon']-row['LONGITUDE'])**2
        idx = _df.idxmin()

        if _df.loc[idx] < 0.01:
            row['SA03'] = gm.loc[idx, 'gmv_SA(0.3)']
            row['SA10'] = gm.loc[idx, 'gmv_SA(1.0)']
            row['Kshape'] = get_kshape(row['Ishape'], row['SA03'], row['SA10'])

            df_pe = compute_pe_by_ds(
                bridge_param.loc[row['HAZUS_CLASS']], row['Kshape'], row['Kskew'], row['K3d'], row['SA10'])

            #df_pb = get_pb_from_pe(df_pe)
            #df_pe['Kshape'] = row['Kshape']
            #df_pe['SA10'] = row['SA10']

            dmg.append(df_pe)

        else:
            print('Something wrong {}:{}'.format(i, _df.loc[idx]))

    dmg = pd.DataFrame(dmg)
    dmg.index = df_bridge.index
    #df_bridge = pd.concat([df_bridge, tmp], axis=1)

    # convert to edge prob
    #dmg = convert_dmg(tmp)

    dir_path = Path(file_gm).parent
    file_output = dir_path.joinpath(Path(file_gm).stem + '_dmg.csv')
    dmg.to_csv(file_output)
    print(f'{file_output} saved')

    return dmg


def convert_dmg(dmg):

    cfg = config.Config(HOME.joinpath('./config.json'))

    tmp = {}
    for k, v in cfg.infra['edges'].items():

        try:
            p0 = 1 - dmg.loc[v['origin']]['Extensive']
        except KeyError:
            p0 = 1.0

        try:
            p1 = 1 - dmg.loc[v['destination']]['Extensive']
        except KeyError:
            p0 = 1.0
        finally:
            tmp[k] = {'F': 1 - p0 * p1}

    tmp = pd.DataFrame(tmp).T
    tmp['S'] = 1 - tmp['F']

    return tmp


@app.command()
def plot_alt():

    G = nx.Graph()

    cfg = config.Config(HOME.joinpath('./config.json'))
    #cfg.infra['G'].edges()

    # read routes
    for route_file in Path('../bridge/').glob('route*.txt'):
        #print(_file)

        route = [x.strip() for x in pd.read_csv(route_file, header=None, dtype=str, )[0].to_list()]

        for item in zip(route[:-1], route[1:]):
            try:
                label = cfg.infra['G'].edges[item]['label']
            except KeyError:
                try:
                    label = cfg.infra['G'].edges[item[::-1]]['label']
                except KeyError:
                    print(f'Need to add {item}')
                else:
                    G.add_edge(item[0], item[1], label=label, key=label)
            else:
                G.add_edge(item[0], item[1], label=label, key=label)

            for i in item:
                G.add_node(i, pos=(0, 0), label=i, key=i)

    config.plot_graphviz(G, outfile=HOME.joinpath('wheat_graph_a'))
    # check against model
    print(nx.is_isomorphic(G, cfg.infra['G']))
    a = set([(x[1], x[0]) for x in G.edges]).union(G.edges)
    b = set([(x[1], x[0]) for x in cfg.infra['G'].edges]).union(cfg.infra['G'].edges)
    print(set(a).difference(b))
    print(set(b).difference(a))
    print(len(G.edges), len(cfg.infra['G'].edges))
    print(len(G.nodes), len(cfg.infra['G'].nodes))

    """
    origin = route[0]
    if origin in bridges.index:
        origin = bridges.loc[origin][['lat', 'lng']].values.tolist()

    dest = route[-1]
    if dest in bridges.index:
        dest = bridges.loc[dest][['lat', 'lng']].values.tolist()

    route_btw = [{'lat': bridges.loc[x, 'lat'], 'lng': bridges.loc[x, 'lng'], 'id': x} for x in route[1:-1]]


    # create edges

    # plot
    """


@app.command()
def plot():

    cfg = config.Config(HOME.joinpath('./config.json'))
    trans.plot_graph(cfg.infra['G'], HOME.joinpath('wheat.png'))
    config.plot_graphviz(cfg.infra['G'], outfile=HOME.joinpath('wheat_graph'))



def get_vari_cpm_given_path(route, edge_names, weight):

    name = '_'.join((*od_pair, str(idx)))
    path_names.append(name)

    vari = {name: variable.Variable(name=name, values = [np.inf, weight])}

    n_child = len(path)

    # Event matrix of series system
    # Initialize an array of shape (n+1, n+1) filled with 1
    Cpath = np.ones((n_child + 1, n_child + 1), dtype=int)
    for i in range(1, n_child + 1):
        Cpath[i, 0] = 0
        Cpath[i, i] = 0 # Fill the diagonal below the main diagonal with 0
        Cpath[i, i + 1:] = 2 # Fill the lower triangular part (excluding the diagonal) with 2

    #for i in range(1, n_edge + 1):
    ppath = np.array([1.0]*(n_child + 1))

    cpm = {name: cpm.Cpm(variables = [varis[name]] + [varis[n] for n in path], no_child=1, C=Cpath, p=ppath)}

    return vari, cpm


@app.command()
def setup_model(key: str='Wooroloo-Merredin',
                route_file: str=None) -> str:

    cfg = config.Config(HOME.joinpath('./config.json'))

    od_pair = cfg.infra['ODs'][key]
    od_name = '_'.join(od_pair)

    # variables
    cpms = {}
    varis = {k: variable.Variable(name=k, values = ['f', 's'])
             for k in cfg.infra['nodes'].keys()}

    G = cfg.infra['G']
    d_time_itc = nx.shortest_path_length(G, source=od_pair[0], target=od_pair[1], weight='weight')

    if route_file:
        with open(route_file, 'r') as f:
            selected_paths = json.load(f)

        selected_paths = [item for item in selected_paths.values()]

    else:
        try:
            no_paths = cfg.data['NO_PATHS']
        except KeyError:
            selected_paths = nx.all_simple_paths(G, od_pair[0], od_pair[1])
        else:
            selected_paths = k_shortest_paths(G, source=od_pair[0], target=od_pair[1], k=no_paths, weight='weight')

    valid_paths = []
    for path in selected_paths:
        # Calculate the total weight of the path
        path_edges = [(u, v) for u, v in zip(path[:-1], path[1:])]
        path_weight = sum(G[u][v]['weight'] for u, v in path_edges)

        # if takes longer than thres * d_time_itc, we consider the od pair is disconnected; moved to config
        if path_weight < cfg.data['THRESHOLD'] * d_time_itc:

            # Collect the edge names if they exist, otherwise just use the edge tuple
            edge_names = [G[u][v].get('key', (u, v)) for u, v in path_edges]
            valid_paths.append((path, edge_names, path_weight))

    # Sort valid paths by weight
    valid_paths = sorted(valid_paths, key=lambda x: x[2])

    # convert to namedtuple
    keys = ['path', 'edges', 'weight']
    valid_paths = {'_'.join((*od_pair, str(i))): dict(zip(keys, item)) for i, item in enumerate(valid_paths)}
    path_names = list(valid_paths.keys())

    for name, route in valid_paths.items():

        varis[name] = variable.Variable(name=name, values = [np.inf, route['weight']])

        n_child_p1 = len(route['path']) + 1

        # Event matrix of series system
        # Initialize an array of shape (n+1, n+1) filled with 1
        Cpath = np.ones((n_child_p1, n_child_p1), dtype=int)
        for i in range(1, n_child_p1):
            Cpath[i, 0] = 0
            Cpath[i, i] = 0 # Fill the diagonal below the main diagonal with 0
            Cpath[i, i + 1:] = 2 # Fill the lower triangular part (excluding the diagonal) with 2

        #for i in range(1, n_edge + 1):
        ppath = np.ones(n_child_p1)
        cpms[name] = cpm.Cpm(variables = [varis[name]] + [varis[n] for n in route['path']],
                             no_child=1, C=Cpath, p=ppath)

    # create varis, cpms for od_pair
    vals = [np.inf] + [varis[p].values[1] for p in path_names[::-1]]
    varis[od_name] = variable.Variable(name=od_name, values=vals)

    n_path = len(path_names)
    Csys = np.zeros((n_path + 1, n_path + 1), dtype=int)
    for i in range(n_path):
        Csys[i, 0] = n_path - i
        Csys[i, i + 1] = 1
        Csys[i, i + 2:] = 2
    psys = np.ones(n_path + 1)
    cpms[od_name] = cpm.Cpm(variables = [varis[od_name]] + [varis[p] for p in path_names], no_child=1, C=Csys, p=psys)

    # save 
    output_model = cfg.output_path.joinpath(f'model_{od_name}_{len(valid_paths)}.pk')
    with open(output_model, 'wb') as f:
        dump = {'cpms': cpms,
                'varis': varis,
                'valid_paths': valid_paths,
                'cfg': cfg}
        pickle.dump(dump, f)
    print(f'{output_model} saved')

    # route_file
    if not route_file:
        route_for_dictions = {k: v['path'] for k, v in valid_paths.items()}
        file_output = cfg.output_path.joinpath(f'model_{od_name}_{len(valid_paths)}.json')
        with open(file_output, 'w') as f:
            json.dump(route_for_dictions, f, indent=4)
        print(f'{file_output} saved')

    return output_model

@app.command()
def single(key: str,
          file_dmg: str,
          route_file: str=None) -> None:
    """
    key: 'York-Merredin'
    file_dmg:
    route_file:
    """
    file_model = setup_model(key=key, route_file=route_file)

    reliability(file_model=file_model, file_dmg=file_dmg)

    inference(file_model=file_model, file_dmg=file_dmg)


@app.command()
def batch(file_dmg: str):

    cfg = config.Config(HOME.joinpath('./config.json'))

    for key in cfg.infra['ODs']:
        batch(file_dmg, key)



@app.command()
def inference(file_model: str, file_dmg: str) -> None:

    with open(file_model, 'rb') as f:
        dump = pickle.load(f)

    cfg = dump['cfg']
    cpms = dump['cpms']
    varis = dump['varis']
    valid_paths = dump['valid_paths']
    path_names = list(valid_paths.keys())
    no_paths = len(path_names)

    update_nodes_given_dmg(file_dmg, cfg.infra['nodes'], valid_paths)

    for k, v in cfg.infra['nodes'].items():
        cpms[k] = cpm.Cpm(variables = [varis[k]], no_child=1,
                            C = np.array([0, 1]).T, p = [v['failure'], 1 - v['failure']])

    od_name = '_'.join(path_names[0].split('_')[:-1])
    VE_ord = list(cfg.infra['nodes'].keys()) + path_names
    vars_inf = operation.get_inf_vars(cpms, od_name, VE_ord)

    Mod = operation.variable_elim([cpms[k] for k in vars_inf], [v for v in vars_inf if v != od_name])
    #Mod.sort()

    plt.figure()
    p_flat = Mod.p.flatten()
    elapsed = [varis[od_name].values[int(x[0])] for x in Mod.C]

    elapsed_in_mins = [f'{(x/60):.1f}' for x in elapsed]
    elapsed_in_mins = [f'{no_paths-x[0]}: {y}' if x[0] else y for x, y in zip(Mod.C, elapsed_in_mins)]
    plt.bar(range(len(p_flat)), p_flat, tick_label=elapsed_in_mins)

    plt.xlabel("Travel time (mins)")
    plt.ylabel("Probability")
    file_output = cfg.output_path.joinpath(f'{Path(file_dmg).stem}_{od_name}_travel.png')
    plt.savefig(file_output, dpi=100)
    print(f'{file_output} saved')

    prob_by_path = {f'{od_name}_{no_paths-x[0]}': y for x, y in zip(Mod.C, p_flat) if x[0]}

    # create shp file
    json_file = Path(file_model).parent.joinpath(Path(file_model).stem + '_direction.json')
    outfile = cfg.output_path.joinpath(f'{Path(file_model).stem}_{Path(file_dmg).stem}_direction.shp')
    create_shpfile(json_file, prob_by_path, outfile)

    # create shp file for node failure
    outfile = cfg.output_path.joinpath(f'{Path(file_model).stem}_{Path(file_dmg).stem}_nodes.shp')
    create_shpfile_nodes(cfg.infra['nodes'], outfile)


@app.command()
def reliability(file_model: str, file_dmg: str) -> None:
    """
    compute the likelihood of routes being available
    file_model:
    file_dmg

    """

    with open(file_model, 'rb') as f:
        dump = pickle.load(f)

    cfg = dump['cfg']
    cpms = dump['cpms']
    varis = dump['varis']
    valid_paths = dump['valid_paths']
    path_names = list(valid_paths.keys())
    no_paths = len(path_names)

    update_nodes_given_dmg(file_dmg, cfg.infra['nodes'], valid_paths)

    for k, v in cfg.infra['nodes'].items():
        cpms[k] = cpm.Cpm(variables = [varis[k]], no_child=1,
                            C = np.array([0, 1]).T, p = [v['failure'], 1 - v['failure']])

    od_name = '_'.join(path_names[0].split('_')[:-1])

    paths_rel = {}
    VE_ord = list(cfg.infra['nodes'].keys()) + path_names
    for path in path_names:
        vars_inf = operation.get_inf_vars(cpms, path, VE_ord)
        Mpath = operation.variable_elim([cpms[k] for k in vars_inf], [v for v in vars_inf if v!=path])
        paths_rel[path] = Mpath.p[1][0]

    fig, ax = plt.subplots()
    ax.bar(range(len(paths_rel)), paths_rel.values(), tick_label=list(paths_rel.keys()))
    fig.autofmt_xdate()
    ax.set_xlabel(f"Route: {od_name}")
    ax.set_ylabel("Reliability")

    file_output = cfg.output_path.joinpath(f'{Path(file_dmg).stem}_{od_name}_routes.png')
    fig.savefig(file_output, dpi=100)
    print(f'{file_output} saved')

    # create shp file
    json_file = Path(file_model).parent.joinpath(Path(file_model).stem + '_direction.json')
    if json_file.exists():
        outfile = cfg.output_path.joinpath(f'{Path(file_model).stem}_{Path(file_dmg).stem}_route.shp')
        create_shpfile(json_file, paths_rel, outfile)

        # nodes
        outfile = cfg.output_path.joinpath(f'{Path(file_model).stem}_{Path(file_dmg).stem}_nodes.csv')
        df_pf = pd.Series({x: v['failure'] for x, v in cfg.infra['nodes'].items()}, name='failure')
        df_pf.sort_values(ascending=False).to_csv(outfile, float_format='%.2f')


if __name__=='__main__':

    app()

