import batch
import numpy as np
from scipy import interpolate
from pathlib import Path
from scipy.stats import norm
from BNS_JT import variable, cpm, trans, config
import copy, pickle, time
import concurrent.futures
from multiprocessing import freeze_support
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import networkx as nx
from matplotlib.lines import Line2D
from scipy.stats import multivariate_normal, norm

#import typer
#app = typer.Typer()

HOME = Path(__file__).parent
output_path = HOME.joinpath('./output')
output_path.mkdir(parents=True, exist_ok=True)


def cal_prob_dep( node, cfg, eq_name, arcs, output_path ):

    pf, MEAN, COV, ln_Sa = batch.cal_edge_dist_output(cfg, eq_name)

    start_ = time.time()

    with open( output_path / f"cpms_{node}.pk", 'rb') as f:
        cpms = pickle.load(f)

    cpm_node = cpms[node]

    """if len(cpm_node.Cs) > 0:
        m_unk = cpm.Cpm( [cpm_node.variables[0]], 1, np.array([2]), np.array([1.0]) ) # remove unknown state instances
        is_cmp = cpm_node.iscompatible( m_unk )
        cpm_node = cpm_node.get_subset( [i for i, v in enumerate(is_cmp) if not v] )"""

    p_node = np.zeros_like(cpm_node.p)

    for j, c in enumerate(cpm_node.C):

        mean_ = np.zeros((0,1), dtype=float)
        cov_ = np.zeros((0,0), dtype=float)
        arcs_idx_ = []

        for j2, (k,v) in enumerate(arcs.items()):
            c_idx = next((i for i,x in enumerate(cpm_node.variables) if x.name==k), None)

            if c[c_idx] != 2: # no composite state

                if c[c_idx] == 1: # survival
                    mean_ = np.vstack((mean_, -MEAN[k]))
                elif c[c_idx] == 0: # failure
                    mean_ = np.vstack((mean_, MEAN[k]))

                cov1_ = np.zeros((len(arcs_idx_), 1), dtype=float)
                for i,c_idx2 in enumerate(arcs_idx_):
                    cov1_[i] = COV[c_idx2, j2]
                cov_ = np.hstack( (cov_, cov1_) )
                cov1_ = np.vstack( (cov1_, COV[j2, j2]) )
                cov_ = np.vstack( (cov_, cov1_.T ) )

                arcs_idx_.append(j2)

        mean_ = np.squeeze(mean_)
        p_ = multivariate_normal.cdf(np.zeros((len(arcs_idx_),), dtype=float), mean_, cov_, allow_singular=True)
        p_node[j] = p_

    p_node = p_node / sum(p_node) # normalisation due to numerical errors from mvn cdf
    cpm_node.p = p_node

    if len(cpm_node.Cs) == 0:

        pf_low = cpm.get_prob(cpm_node, [node], [0])  
        pf_unk = cpm.get_prob(cpm_node, [node], [2] )
        pf = pf_low   
        pf_bnd = pf_unk
        cov = 0.0

    else:
        m_unk = cpm.Cpm( [cpm_node.variables[0]], 1, np.array([2]), np.array([1.0]) ) # remove unknown state instances
        is_cmp = cpm_node.iscompatible( m_unk )
        cpm_node = cpm_node.get_subset( [i for i, v in enumerate(is_cmp) if not v] )

        ps_ = np.zeros_like(cpm_node.q, dtype=float)

        for i in range(len(cpm_node.q)):
            
            cs_ = cpm_node.Cs[i]
            mean_ = np.zeros((0,1), dtype=float)
            cov_ = np.zeros((0,0), dtype=float)
            arcs_idx_ = []

            for j2, (k,v) in enumerate(arcs.items()):
                c_idx = next((i for i,x in enumerate(cpm_node.variables) if x.name==k), None)

                if cs_[c_idx] == 1: # survival
                    mean_ = np.vstack((mean_, -MEAN[k] + ln_Sa[k]))
                elif cs_[c_idx] == 0: # failure
                    mean_ = np.vstack((mean_, MEAN[k] - ln_Sa[k]))

                cov1_ = np.zeros((len(arcs_idx_), 1), dtype=float)
                for i2,c_idx2 in enumerate(arcs_idx_):
                    cov1_[i2] = COV[c_idx2, j2]
                cov_ = np.hstack( (cov_, cov1_) )
                cov1_ = np.vstack( (cov1_, COV[j2, j2]) )
                cov_ = np.vstack( (cov_, cov1_.T ) )

                arcs_idx_.append(j2)

            mean_ = np.squeeze(mean_)
            p_ = multivariate_normal.pdf(np.zeros((len(arcs),), dtype=float), mean_, cov_, allow_singular=True)
            ps_[i] = p_

        cpm_node.ps = ps_

        prob, cov, cint_95 = cpm.get_prob_and_cov( cpm_node, [node], [0], method='Bayesian' )

        pf_low = cpm.get_prob(cpm_node, [node], [0])  
        pf_unk = cpm.get_prob(cpm_node, [node], [2] )
        pf = cint_95[0]   
        pf_bnd = cint_95[1] - cint_95[0]  

    end_ = time.time()
    sec = end_ - start_

    print(f"Node {node} done.")

    return node, pf, sec, pf_bnd, cov
    

#@app.command()
def eval_pfs_dep( config_fname, eq_name, fout_name ):
    # e.g. config_fname, eq_name, fout_name = 'config.json', 's1', 'pf_dep.txt'
    
    #cfg = config_pm.Config_pm(HOME / f"input/{config_fname}")
    cfg = batch.config_custom(config_fname, eq_name)
    node_coords = {}
    for k, v in cfg.infra['nodes'].items():
        node_coords[k] = (v['pos_x'], v['pos_y'])

    arcs = {}
    for k, v in cfg.infra['edges'].items():
        arcs[k] = [v['origin'], v['destination']]

    epi_loc = cfg.infra['eq'][eq_name]['epicentre']
    os_list = cfg.infra['origins']

    # calculate failure probabilities when dependence exists
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for node in node_coords:
            if node != 'epi' and node not in os_list:
                futures.append( executor.submit(cal_prob_dep, node, cfg, eq_name, arcs, output_path) )

    pfs = {}
    pfs_bnd = {}
    covs = {}
    times = {}
    for future in concurrent.futures.as_completed(futures):
        node, pf, sec, pf_bnd, cov = future.result()
        pfs[node], pfs_bnd[node], covs[node], times[node] = pf, pf_bnd, cov, sec

    """for node in node_coords:
        if node != 'epi' and node not in os_list:
            pf, sec, pf_bnd, cov = cal_prob_dep( node, cfg, eq_name, arcs, output_path )    
            pfs[node], pfs_bnd[node], covs[node], times[node] = pf, pf_bnd, cov, sec
            print(f"Node {node} done.")"""

    with open(output_path / fout_name, 'w') as f:
        for node in node_coords:
            if node != 'epi' and node not in os_list:
                f.write(f"{node}\t{pfs[node]:.4e}\t{pfs_bnd[node]:.4e}\t{covs[node]:.4e}\t{times[node]}\n")

def update_pf1(node, arcs, varis, new_probs):

    print(f"Update_pf: Analyzing node {node}...")

    start = time.time()
    with open( output_path / f"cpms_{node}.pk", 'rb') as f:
        cpms = pickle.load(f)

    for k, v in arcs.items():
        cpms[k].p = np.array([new_probs[k][0], new_probs[k][1]], dtype=float).T

    if len(cpms[node].Cs) > 0:

        ps = np.zeros_like(cpms[k].q, dtype=float)
        for i in range(len(cpms[k].q)):
            ps[i] = cpm.get_prob(cpms[k], [k], cpms[k].Cs[i])
        cpms[k].ps = ps

        m_unk = cpm.Cpm( [varis[node]], 1, np.array([2]), np.array([1.0]) ) # remove unknown state instances
        is_cmp = cpms[node].iscompatible( m_unk )
        cpms[node] = cpms[node].get_subset( [i for i, v in enumerate(is_cmp) if not v] )

        M_node = cpm.prod_Msys_and_Mcomps(cpms[node], [cpms[k] for k in arcs])
        prob, cov, cint_95 = cpm.get_prob_and_cov( M_node, [node], [0], method='Bayesian' )
        pf = cint_95[0]
        pf_bnd = cint_95[1] - cint_95[0]

    else:
        M_node = cpm.prod_Msys_and_Mcomps(cpms[node], [cpms[k] for k in arcs])
        pf = cpm.get_prob(M_node, [node], [0])
        pf_bnd = cpm.get_prob(M_node, [node], [2])
        cov = 0.0
    
    end = time.time()
    sec = end-start

    print(f"Update_pf: Node {node} done.")

    return node, pf, sec, pf_bnd, cov

def update_pfs( config_fname, eq_name, fout_name ):
    # e.g. config_fname, eq_name, fout_name = 'config.json', 's1', 'pf_dep.txt'
    
    #cfg = config_pm.Config_pm(HOME / f"input/{config_fname}")
    cfg = batch.config_custom(config_fname, eq_name)
    node_coords = {}
    for k, v in cfg.infra['nodes'].items():
        node_coords[k] = (v['pos_x'], v['pos_y'])

    arcs = {}
    for k, v in cfg.infra['edges'].items():
        arcs[k] = [v['origin'], v['destination']]

    with open(output_path / "varis.pk", 'rb') as f:
        varis = pickle.load(f)

    epi_loc = cfg.infra['eq'][eq_name]['epicentre']
    os_list = cfg.infra['origins']
    pf, MEAN, COV, ln_Sa = batch.cal_edge_dist_output(cfg, eq_name)
    new_probs = {k: {0:v, 1:1-v} for k,v in pf.items()}

    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for node in node_coords:
            if node != 'epi' and node not in os_list:
                futures.append( executor.submit(update_pf1, node, arcs, varis, new_probs) )

    pfs = {}
    pfs_bnd = {}
    covs = {}
    times = {}
    for future in concurrent.futures.as_completed(futures):
        node, pf, sec, pf_bnd, cov = future.result()
        pfs[node], pfs_bnd[node], covs[node], times[node] = pf, pf_bnd, cov, sec

    """for node in node_coords:
        if node != 'epi' and node not in os_list:
            pf, sec, pf_bnd, cov = update_pf1(node, arcs, varis, new_probs)    
            pfs[node], pfs_bnd[node], covs[node], times[node] = pf, pf_bnd, covs, sec
            print(f"Node {node} done.")"""

    with open(output_path / fout_name, 'w') as f:
        for node in node_coords:
            if node != 'epi' and node not in os_list:
                f.write(f"{node}\t{pfs[node]:.4e}\t{pfs_bnd[node]:.4e}\t{covs[node]:.4e}\t{times[node]}\n")

if __name__=='__main__':

    #update_pfs( 'config.json', 's2', 'pf_upd.txt' )
    eval_pfs_dep( 'config.json', 's1', 'pf_dep.txt' )

    # for debugging
    """eq_name = 's1'
    cfg = batch.config_custom('config.json', eq_name)
    node_coords = {}
    for k, v in cfg.infra['nodes'].items():
        node_coords[k] = (v['pos_x'], v['pos_y'])
    arcs = {}
    for k, v in cfg.infra['edges'].items():
        arcs[k] = [v['origin'], v['destination']]
    epi_loc = cfg.infra['eq'][eq_name]['epicentre']
    os_list = cfg.infra['origins']

    node, pf, sec, pf_bnd, cov = cal_prob_dep( 'n43', cfg, eq_name, arcs, output_path )"""

    """with open(output_path / "varis.pk", 'rb') as f:
        varis = pickle.load(f)

    epi_loc = cfg.infra['eq'][eq_name]['epicentre']
    os_list = cfg.infra['origins']
    pf, MEAN, COV, ln_Sa = batch.cal_edge_dist_output(cfg, eq_name)
    new_probs = {k: {0:v, 1:1-v} for k,v in pf.items()}
    node, pf, sec, pf_bnd, cov = update_pf1( 'n29', arcs, varis, new_probs )
    ddd = 1"""

    #app()
