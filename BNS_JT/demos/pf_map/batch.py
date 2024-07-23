import json
import numpy as np
from scipy import interpolate
from pathlib import Path
from scipy.stats import norm
import copy, pickle, time
import concurrent.futures
from multiprocessing import freeze_support
from scipy.stats import beta
import typer

from BNS_JT import variable, cpm, trans, config, brc

HOME = Path(__file__).parent

app = typer.Typer()


def read_model_from_json_custom(file_input):

    with open(file_input, 'r') as f:
        model = json.load(f)

    # added
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

    return {'origins': origins, 'eq': eq, 'frag': frag, 'thres': thres}


# Edge failure probability calculation
def cal_edge_dist(cfg, eq_name):

    # Distance to epicentre
    Rrup = {}
    epi_loc = cfg.infra['eq'][eq_name]['epicentre']

    for k, v in cfg.infra['edges'].items():
        org = cfg.infra['nodes'][v['origin']]
        dest = cfg.infra['nodes'][v['destination']]
        Rrup[k] = shortest_distance(org.values(), dest.values(), epi_loc)

    # GMPE model (Campbell 2003)
    Mw = cfg.infra['eq'][eq_name]['Mw']
    ln_Sa, std_al, std_ep = gmpe_cam03(Mw, Rrup)

    dmg_st = 2 # Extensive damage

    # mean and covariance matrix between failures of roads
    no_edges = len(cfg.infra['edges'])
    vari = {}
    cov = np.zeros(shape=(no_edges, no_edges))

    for i1, (k1, v1) in enumerate(cfg.infra['edges'].items()):

        mean_ = cfg.infra['frag'][v1['fragility_type']]['Sa_g'][dmg_st]
        std_ = cfg.infra['frag'][v1['fragility_type']]['Sa_g_dispersion']

        vari[k1] = std_al**2 + std_ep[k1]**2 + std_**2

        for i2, k2 in enumerate(cfg.infra['edges'].keys()):

            if k1 == k2:
                cov[i1, i2] = std_al**2 + std_ep[k1]**2 + std_**2
            else:
                cov[i1, i2] = std_al**2

        v1['mean'] = np.log(mean_) - ln_Sa[k1]

        # failure probability
        v1['pf'] = norm.cdf(0, v1['mean'], np.sqrt(vari[k1]))


def cal_edge_dist_output(cfg, eq_name):

    # Distance to epicentre
    Rrup = {}
    epi_loc = cfg.infra['eq'][eq_name]['epicentre']

    for k, v in cfg.infra['edges'].items():
        org = cfg.infra['nodes'][v['origin']]
        dest = cfg.infra['nodes'][v['destination']]
        Rrup[k] = shortest_distance(org.values(), dest.values(), epi_loc)

    # GMPE model (Campbell 2003)
    Mw = cfg.infra['eq'][eq_name]['Mw']
    ln_Sa, std_al, std_ep = gmpe_cam03(Mw, Rrup)

    dmg_st = 2 # Extensive damage

    # mean and covariance matrix between failures of roads
    no_edges = len(cfg.infra['edges'])
    vari = {}
    COV = np.zeros(shape=(no_edges, no_edges))

    pf, MEAN = {}, {}
    for i1, (k1, v1) in enumerate(cfg.infra['edges'].items()):

        mean_ = cfg.infra['frag'][v1['fragility_type']]['Sa_g'][dmg_st]
        std_ = cfg.infra['frag'][v1['fragility_type']]['Sa_g_dispersion']

        vari[k1] = std_al**2 + std_ep[k1]**2 + std_**2

        for i2, k2 in enumerate(cfg.infra['edges'].keys()):

            if k1 == k2:
                COV[i1, i2] = std_al**2 + std_ep[k1]**2 + std_**2
            else:
                COV[i1, i2] = std_al**2

        MEAN[k1] = np.log(mean_) - ln_Sa[k1]

        # failure probability
        pf[k1] = norm.cdf(0, MEAN[k1], np.sqrt(vari[k1]))

    return pf, MEAN, COV, ln_Sa


def shortest_distance(line_pt1, line_pt2, pt):
    """
    Calculate the shortest distance between a point and a line.

    The line is defined by two points (x1, y1) and (x2, y2).
    The point is defined by (x0, y0).

    Returns:
    - distance: The shortest distance between the line and the point. In case the projection of the point is outside the line segment, the distance is calculated to the closest endpoint.
    """
    x1, y1 = line_pt1
    x2, y2 = line_pt2
    x0, y0 = pt

    # Calculate the dot products
    dot1 = ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / ((x2 - x1)**2 + (y2 - y1)**2)
    dot2 = ((x0 - x2) * (x1 - x2) + (y0 - y2) * (y1 - y2)) / ((x1 - x2)**2 + (y1 - y2)**2)

    # Check if the projection of (x0, y0) is outside the line segment
    if dot1 < 0:
        return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
    elif dot2 < 0:
        return np.sqrt((x0 - x2)**2 + (y0 - y2)**2)

    # Calculate the shortest distance to the line
    numerator = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
    denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    distance = numerator / denominator

    return distance


def gmpe_cam03(Mw, Rrup):
    """
    Calculate Sa (g) at T=0.1s of each edge (given as keys in Rrup) using the GMPE model by Campbell (2003).

    INPUTS:
    Mw: moment magnitude
    Rrup: a dictionary with edges as keys and their distances (km) to epicenter as values

    OUTPUTS:
    ln_Sa: a dictionary with edges as keys and their Sa values (g) as values
    std_al: a dictionary with edges as keys and their aleatory standard deviations as values
    std_ep: a dictionary with edges as keys and their epistemic standard deviations as values
    """

    assert Mw >= 6.0 and Mw <= 8.5, f'Mw should be between 6.0 and 8.5; currently given as {Mw}.'

    # Parameters#########
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13 = -0.6104, 0.451, -0.2090, -1.158, -0.00255, 0.000141, 0.299, 0.503, 1.067, -0.482, 1.110, -0.0793, 0.543
    r1, r2 = 70, 130
    M1 = 7.16
    std_eps_Mw = np.array([5.0, 5.4, 5.8, 6.2, 6.6, 7.0, 7.4, 7.8, 8.2])
    std_eps_rrup = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0, 40.0, 50.0, 70.0])
    std_eps_val = np.array([[0.24, 0.25, 0.25, 0.21, 0.18, 0.13, 0.06, 0.10, 0.14, 0.17, 0.23],
                           [0.22, 0.23, 0.23, 0.22, 0.19, 0.16, 0.10, 0.11, 0.14, 0.18, 0.23],
                           [0.19, 0.20, 0.21, 0.20, 0.19, 0.17, 0.14, 0.15, 0.18, 0.21, 0.27],
                           [0.15, 0.15, 0.16, 0.16, 0.17, 0.16, 0.14, 0.15, 0.18, 0.21, 0.27],
                           [0.12, 0.13, 0.13, 0.13, 0.14, 0.14, 0.13, 0.15, 0.18, 0.21, 0.27],
                           [0.13, 0.13, 0.13, 0.13, 0.14, 0.14, 0.14, 0.15, 0.17, 0.20, 0.27],
                           [0.14, 0.13, 0.13, 0.14, 0.15, 0.16, 0.16, 0.16, 0.18, 0.21, 0.27],
                           [0.15, 0.15, 0.14, 0.15, 0.15, 0.17, 0.18, 0.18, 0.20, 0.22, 0.28],
                           [0.17, 0.17, 0.16, 0.16, 0.17, 0.18, 0.20, 0.21, 0.22, 0.23, 0.29]])

    Mw_grid, rrup_grid = np.meshgrid(std_eps_Mw, std_eps_rrup)
    std_ep_f = interpolate.bisplrep( Mw_grid, rrup_grid, std_eps_val, s=0 )
    ######################

    ln_Sa, std_ep = {}, {}
    for e, r in Rrup.items():

        # ln Sa
        f1 = c2 * Mw + c3 * (8.5 - Mw)**2

        R = np.sqrt( r**2 + (c7 * np.exp(c8*Mw))**2 )
        f2 = c4 * np.log(R) + (c5 + c6 * Mw) * r

        if r <= r1:
            f3 = 0
        elif r <= r2:
            f3 = c7 * np.log(r/r1)
        else:
            f3 = c7 * np.log(r/r1) + c8 * np.log(r/r2)

        ln_Sa[e] = c1 + f1 + f2 + f3

        # epistemic standard deviation
        Mw_new = np.clip(Mw, min(std_eps_Mw), max(std_eps_Mw))
        r_new = np.clip(r, min(std_eps_rrup), max(std_eps_rrup))
        std_ep[e] = interpolate.bisplev(Mw_new, r_new, std_ep_f)

    # aleatory standard deviation (the same for all raods)
    if Mw < M1:
        std_al = c11 + c12
    else:
        std_al = c13

    return ln_Sa, std_al, std_ep


def mcs_unknown(brs_u, probs, sys_fun_rs, cpms, sys_name, cov_t, sys_st_monitor, sys_st_prob, rand_seed=None):
    """
    Perform Monte Carlo simulation for the unknown state.

    INPUTS:
    brs_u: Unspecified branches (list)
    probs: a dictionary of failure probabilities for each component
    sys_fun_rs: System function
    cpms: a list of cpms containing component events and system event
    sys_name: a string of the system event's name in cpms
    cov_t: a target c.o.v.
    sys_st_monitor: System state to monitor (e.g. 0)
    sys_st_prob: known probability of sys_st_monitor
    rand_seed: Random seed

    OUTPUTS:
    result: Results of the Monte Carlo simulation
    """

    # Set the random seed
    if rand_seed:
        np.random.seed(rand_seed)

    brs_u_probs = [b[4] for b in brs_u]
    brs_u_prob = sum(brs_u_probs)

    samples = []
    samples_sys = np.empty((0, 1), dtype=int)
    sample_probs = []

    nsamp, nfail = 0, 0
    pf, cov = 0.0, 1.0

    while cov > cov_t:

        nsamp += 1

        sample1 = {}
        s_prob1 = {}

        # select a branch
        br_id = np.random.choice(range(len(brs_u)), p=brs_u_probs / brs_u_prob)
        br = brs_u[br_id]

        for e in br.down.keys():
            d = br.down[e]
            u = br.up[e]

            if d < u: # (fail, surv)
                st = np.random.choice(range(d, u + 1), p=[probs[e][d], probs[e][u]])
            else:
                st = d

            sample1[e] = st
            s_prob1[e] = probs[e][st]

        # system function run
        val, sys_st = sys_fun_rs(sample1)

        samples.append(sample1)
        sample_probs.append(s_prob1)
        samples_sys = np.vstack((samples_sys, [sys_st]))

        if st == sys_st_monitor:
            nfail += 1

        if nsamp > 9:
            prior = 0.01
            a, b = prior + nfail, prior + (nsamp-nfail) # Bayesian estimation assuming beta conjucate distribution

            pf_s = a / (a+b)
            var_s = a*b / (a+b)**2 / (a+b+1)
            std_s = np.sqrt(var_s)

            pf = sys_st_prob + brs_u_prob *pf_s
            std = brs_u_prob * std_s
            cov = std/pf

            conf_p = 0.95 # confidence interval
            low = beta.ppf(0.5*(1-conf_p), a, b)
            up = beta.ppf(1 - 0.5*(1-conf_p), a, b)
            cint = sys_st_prob + brs_u_prob * np.array([low, up])

        if nsamp % 1000 == 0:
            print(f'nsamp: {nsamp}, pf: {pf:.4e}, cov: {cov:.4e}')

    # Allocate samples to CPMs
    Csys = np.zeros((nsamp, len(probs)), dtype=int)
    Csys = np.hstack((samples_sys, Csys))

    for i, v in enumerate(cpms[sys_name].variables[1:]):
        Cv = np.array([s[v.name] for s in samples], dtype=int).T
        cpms[v.name].Cs = Cv
        cpms[v.name].q = np.array([p[v.name] for p in sample_probs], dtype=float).T
        cpms[v.name].sample_idx = np.arange(nsamp, dtype=int)

        Csys[:, i+1] = Cv.flatten()

    cpms[sys_name].Cs = Csys
    cpms[sys_name].q = np.ones((nsamp,1), dtype=float)
    cpms[sys_name].sample_idx = np.arange(nsamp, dtype=int)

    result = {'pf': pf, 'cov': cov, 'nsamp': nsamp, 'cint_low': cint[0], 'cint_up': cint[1]}

    return cpms, result


def config_custom(file_cfg, eq_name):

    cfg = config.Config(file_cfg)

    if cfg.data['MAX_SYS_FUN']:
        cfg.max_sys_fun = cfg.data['MAX_SYS_FUN']
    else:
        cfg.max_sys_fun = float('inf')

    cfg.sys_bnd_wr = cfg.data['SYS_BND_WIDTH_RATIO']

    cfg.cov_t = cfg.data['MCS_COV']

    # append cfg attributes
    added = read_model_from_json_custom(cfg.file_model)
    cfg.infra.update(added)

   # roads' failure probability
    cal_edge_dist(cfg, eq_name)

    return cfg


@app.command()
def run_MCS(file_cfg, eq_name, node):

    cfg = config_custom(file_cfg, eq_name)

    node_coords = {}
    for k, v in cfg.infra['nodes'].items():
        node_coords[k] = (v['pos_x'], v['pos_y'])

    arcs = {}
    for k, v in cfg.infra['edges'].items():
        arcs[k] = [v['origin'], v['destination']]

    with open(cfg.output_path.joinpath("varis.pk"), 'rb') as f:
        varis = pickle.load(f)

    dests = cfg.infra['origins']
    thres = cfg.infra['thres']
    comps_st_itc = {k: 1 for k in arcs.keys()}

    d_time_itc, _, _ = trans.get_time_and_path_multi_dest(comps_st_itc, cfg.infra['G'], node, dests, varis)
    sys_fun = trans.sys_fun_wrap(cfg.infra['G'], {'origin': node, 'dests': dests}, varis, thres * d_time_itc)

    pf, _, _, _ = cal_edge_dist_output(cfg, eq_name)
    probs = {k: {0:v, 1:1-v} for k,v in pf.items()}

    start = time.time()
    pf, cov, nsamp = brc.run_MCS_indep_comps(probs, sys_fun, cov_t=0.01)
    end = time.time()
    print(f'pf: {pf:.4e}, cov: {cov:.4e}, nsamp: {nsamp:d}, time: {end-start:.2e}')

    with open(cfg.output_path.joinpath(f'mcs_{node}_{eq_name}.txt'), 'w') as fout:
        fout.write(f"pf: {pf:.4e} \n")
        fout.write(f"cov: {cov:.4e} \n")
        fout.write(f"no_samples: {nsamp:d}")
        fout.write(f"time (sec): {end-start:.4e} \n")


def process_node(cfg, node, comps_st_itc, st_br_to_cs, arcs, varis, probs, cpms):

    print(f'-----Analysis begins for node: {node}-----')

    dests = cfg.infra['origins']
    thres = cfg.infra['thres']

    #st_br_to_cs = {'f': 0, 's': 1, 'u': 2}

    if node not in dests:

        d_time_itc, _, _ = trans.get_time_and_path_multi_dest(comps_st_itc, cfg.infra['G'], node, dests, varis)
        sys_fun = trans.sys_fun_wrap(cfg.infra['G'], {'origin': node, 'dests': dests}, varis, thres * d_time_itc)

        """brs, rules, sys_res1, monitor1 = brc.run( {k: varis[k] for k in arcs.keys()}, probs, sys_fun, 0.01*cfg.max_sys_fun, 0.01*cfg.max_branches, cfg.sys_bnd_wr, surv_first=False)
        brs, rules, sys_res2, monitor2 = brc.run( {k: varis[k] for k in arcs.keys()}, probs, sys_fun, cfg.max_sys_fun, cfg.max_branches, cfg.sys_bnd_wr, surv_first=True, rules=rules)
        monitor = {k: v + monitor2[k] for k, v in monitor1.items() if k != 'out_flag'}
        monitor['out_flag'] = [monitor1['out_flag'], monitor2['out_flag']]"""

        brs, rules, sys_res, monitor = brc.run({k: varis[k] for k in arcs.keys()}, probs, sys_fun, cfg.max_sys_fun, cfg.max_branches, cfg.sys_bnd_wr, surv_first=True)

        csys, varis = brc.get_csys(brs, varis, st_br_to_cs)
        #varis[node] = variable.Variable(node, values = ['f', 's', 'u'])
        vari_node = variable.Variable(node, values = ['f', 's', 'u'])
        cpms[node] = cpm.Cpm( [vari_node] + [varis[k] for k in arcs.keys()], 1, csys, np.ones((len(csys),1), dtype=float) )

        pf_u, pf_l = monitor['pf_up'][-1], monitor['pf_low'][-1]

        if (monitor['out_flag'] == 'max_sf' or monitor['out_flag'] == 'max_nb'):

            print(f'*[node {node}] MCS on unknown started..*')

            #csys = csys[ csys[:,0] != st_br_to_cs['u'] ] # remove unknown state instances
            #cpms[node] = cpm.Cpm( [varis[node]] + [varis[k] for k in arcs.keys()], 1, csys, np.ones((len(csys),1), dtype=float) )

            def sys_fun_rs(x):
                val, st, _ = sys_fun(x)
                return val, st_br_to_cs[st]

            start = time.time()
            #cpm2_, result_rs = cpm.rejection_sampling_sys(cpms, node, sys_fun_rs, cfg.cov_t, 0, 1.0-pf_l-pf_u, pf_l, rand_seed=0 )
            brs_u = [b for b in brs if b.up_state == 'u' or b.down_state == 'u' or b.up_state!=b.down_state]
            cpms, result_mcs = mcs_unknown(brs_u, probs, sys_fun_rs, cpms, node, cov_t=0.01, sys_st_monitor=0, sys_st_prob=pf_l, rand_seed=1)

            #cpms[node] = copy.deepcopy(cpm2_)
            #cpm_node = copy.deepcopy(cpm2_)
            end = time.time()

            #sys_pfs[node] = result_['pf']
            sys_pf_node = result_mcs['pf']
            #sys_nsamps[node] = result_['nsamp']
            sys_nsamp_node = result_mcs['nsamp']

            """fout_rs = output_path.joinpath(f'rs_{node}.txt')
            with open(fout_rs, 'w') as f:
                for k, v in result_rs.items():
                    if k in ['pf', 'cov']:
                        f.write(f"{k}\t{v:.4e}\n")
                    elif k in ['nsamp', 'nsamp_tot']:
                        f.write(f"{k}\t{v:d}\n")
                f.write(f"time (sec)\t{end-start:.4e}\n")"""
            result_mcs['time'] = end-start
            print(f'*[node {node}] MCS on unknown completed*')

        else:
            #cpms[node] = cpm.Cpm( [varis[node]] + [varis[k] for k in arcs.keys()], 1, csys, np.ones((len(csys),1), dtype=float) )
            #cpm_node = cpm.Cpm( [vari_node] + [varis[k] for k in arcs.keys()], 1, csys, np.ones((len(csys),1), dtype=float) )
            #sys_pfs[node] = pf_u
            sys_pf_node = pf_u
            #sys_nsamps[node] = 0
            sys_nsamp_node = 0
            result_mcs = None

        # save results
        """fout_monitor = output_path.joinpath(f'brc_{node}.pk')
        with open(fout_monitor, 'wb') as fout:
            pickle.dump(monitor, fout)"""

    else:

        #sys_pfs[node] = 0
        sys_pf_node = 0
        #sys_nsamps[node] = 0
        sys_nsamp_node = 0
        monitor = None
        result_mcs = None
        cpms = None
        vari_node = None
        rules = None

    print(f'-----Analysis completed for node: {node}-----')

    return node, vari_node, cpms, sys_pf_node, sys_nsamp_node, rules, monitor, result_mcs


@app.command()
def debug():

    file_cfg = HOME.joinpath('./input/config.json')
    eq_name = 's1'
    node =  'n1'

    cfg = config_custom(file_cfg, eq_name)

    # For debugging
    cfg.max_branches = 100

    probs = {k: {0: v['pf'], 1: 1 - v['pf']} for k, v in cfg.infra['edges'].items()}

    # arcs and nodes
    arcs = {}
    for k, v in cfg.infra['edges'].items():
        arcs[k] = [v['origin'], v['destination']]

    node_coords = {}
    for k, v in cfg.infra['nodes'].items():
        node_coords[k] = (v['pos_x'], v['pos_y'])

    arc_len = trans.get_arcs_length(arcs, node_coords)
    speed = 100.0 # (km/h) assume homogeneous speed for all roads
    arc_time = {k: v/speed for k, v in arc_len.items()}

    # variables
    varis = {}
    cpms = {}
    comps_st_itc = {}
    for k, v in cfg.infra['edges'].items():
        varis[k] = variable.Variable(name=k, values = [np.inf, arc_time[k]])
        cpms[k] = cpm.Cpm([varis[k]], 1, C=np.array([[0],[1]]), p = np.array([v['pf'], 1 - v['pf']]))
        comps_st_itc[k] = len(varis[k].values) - 1

    st_br_to_cs = {'f': 0, 's': 1, 'u': 2}

    # Run the analysis for single node
    _ = process_node(cfg, node, comps_st_itc, st_br_to_cs, arcs, varis, probs, cpms)


@app.command()
def main(file_cfg, eq_name):

    cfg = config_custom(file_cfg, eq_name)

    probs = {k: {0: v['pf'], 1: 1 - v['pf']} for k, v in cfg.infra['edges'].items()}

    # arcs and nodes
    arcs = {}
    for k, v in cfg.infra['edges'].items():
        arcs[k] = [v['origin'], v['destination']]

    node_coords = {}
    for k, v in cfg.infra['nodes'].items():
        node_coords[k] = (v['pos_x'], v['pos_y'])

    arc_len = trans.get_arcs_length(arcs, node_coords)
    speed = 100.0 # (km/h) assume homogeneous speed for all roads
    arc_time = {k: v/speed for k, v in arc_len.items()}

    # variables
    varis = {}
    cpms = {}
    comps_st_itc = {}
    for k, v in cfg.infra['edges'].items():
        varis[k] = variable.Variable(name=k, values = [np.inf, arc_time[k]])
        cpms[k] = cpm.Cpm([varis[k]], 1, C=np.array([[0],[1]]), p = np.array([v['pf'], 1 - v['pf']]))
        comps_st_itc[k] = len(varis[k].values) - 1

    st_br_to_cs = {'f': 0, 's': 1, 'u': 2}

    # Run the analysis in parallel
    futures = []
    with concurrent.futures.ProcessPoolExecutor() as exec:
        for node in cfg.infra['nodes'].keys():
        #for node in ['n15', 'n53']: # for test
            res1 = exec.submit(process_node, cfg, node, comps_st_itc, st_br_to_cs, arcs, varis, probs, cpms)
            futures.append(res1)

    # Collect the results
    sys_pfs_low, sys_pfs_up, sys_nsamps, covs = {}, {}, {}, {}
    for future in concurrent.futures.as_completed(futures):
        node, vari_node, cpms, sys_pf_node, sys_nsamp_node, rules, monitor, result_mcs = future.result()

        if vari_node is not None:
            varis[node] = vari_node

            fout_cpm = cfg.output_path.joinpath(f'cpms_{node}.pk')
            with open(fout_cpm, 'wb') as fout:
                pickle.dump(cpms, fout)

            sys_pfs_low[node] = monitor['pf_low'][-1]
            sys_pfs_up[node] = monitor['pf_up'][-1]
            sys_nsamps[node] = sys_nsamp_node
            covs[node] = 0.0

            fout_monitor = cfg.output_path.joinpath(f'brc_{node}.pk')
            with open(fout_monitor, 'wb') as fout:
                pickle.dump(monitor, fout)

        if result_mcs is not None:
            fout_rs = cfg.output_path.joinpath(f'rs_{node}.txt')

            sys_pfs_low[node] = result_mcs['cint_low']
            sys_pfs_up[node] = result_mcs['cint_up']
            covs[node] = result_mcs['cov']
            with open(fout_rs, 'w') as f:
                for k, v in result_mcs.items():
                    if k in ['pf', 'cov']:
                        f.write(f"{k}\t{v:.4e}\n")
                    elif k in ['nsamp', 'nsamp_tot']:
                        f.write(f"{k}\t{v:d}\n")
                f.write(f"time (sec)\t{result_mcs['time']:.4e}\n")

        fout_rules = cfg.output_path.joinpath(f'rules_{node}.pk')
        with open(fout_rules, 'wb') as fout:
            pickle.dump(rules, fout)

    # save results
    os_list = cfg.infra['origins']
    fout = cfg.output_path.joinpath(f'result.txt')
    with open(fout, 'w') as f:
        for node in node_coords:
            if node != 'epi' and node not in os_list:
                f.write(f'{node}\t{sys_pfs_low[node]:.4e}\t{sys_pfs_up[node]:.4e}\t{sys_nsamps[node]}\t{covs[node]}\n')

    fout_varis = cfg.output_path.joinpath(f'varis.pk')
    with open(fout_varis, 'wb') as fout:
        pickle.dump(varis, fout)

    print(f'-----All nodes completed. Results saved-----')


@app.command()
def run_single(file_cfg, eq_name, node):

    cfg = config_custom(file_cfg, eq_name)

    probs = {k: {0: v['pf'], 1: 1 - v['pf']} for k, v in cfg.infra['edges'].items()}

    # arcs and nodes
    arcs = {}
    for k, v in cfg.infra['edges'].items():
        arcs[k] = [v['origin'], v['destination']]

    node_coords = {}
    for k, v in cfg.infra['nodes'].items():
        node_coords[k] = (v['pos_x'], v['pos_y'])

    arc_len = trans.get_arcs_length(arcs, node_coords)
    speed = 100.0 # (km/h) assume homogeneous speed for all roads
    arc_time = {k: v/speed for k, v in arc_len.items()}

    # variables
    varis = {}
    cpms = {}
    comps_st_itc = {}
    for k, v in cfg.infra['edges'].items():
        varis[k] = variable.Variable(name=k, values = [np.inf, arc_time[k]])
        cpms[k] = cpm.Cpm([varis[k]], 1, C=np.array([[0],[1]]), p = np.array([v['pf'], 1 - v['pf']]))
        comps_st_itc[k] = len(varis[k].values) - 1

    st_br_to_cs = {'f': 0, 's': 1, 'u': 2}

    # Run the analysis for single node
    _ = process_node(cfg, node, comps_st_itc, st_br_to_cs, arcs, varis, probs, cpms)


@app.command()
def batch_comp():

    # To compare results with MCS results
    for node in ['n64', 'n67', 'n29', 'n62', 'n63', 'n65']:
        print(f"{node} begins..")
        run_MCS(HOME.joinpath('./input/config.json'), 's1', node)
        run_MCS(HOME.joinpath('./input/config.json'), 's2', node)

@app.command()
def parallel():
    freeze_support()
    main(HOME.joinpath('./input/config.json'), 's1')


if __name__ == '__main__':
    app()
