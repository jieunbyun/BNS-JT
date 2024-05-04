import config_pm
import numpy as np
from scipy import interpolate
from pathlib import Path
from scipy.stats import norm
from BNS_JT import variable, cpm, gen_bnb, trans
import copy, pickle, time

HOME = Path(__file__).parent
output_path = HOME.joinpath('./output')
output_path.mkdir(parents=True, exist_ok=True)

# Edge failure probability calculation
def cal_edge_dist(infra_data, eq_name):

    # Distance to epicentre
    Rrup = {}
    epi_loc = infra_data['eq'][eq_name]['epicentre']

    for k, v in infra_data['edges'].items():
        x1, y1 = infra_data['nodes'][v['origin']]['pos_x'], infra_data['nodes'][v['origin']]['pos_y']
        x2, y2 = infra_data['nodes'][v['destination']]['pos_x'], infra_data['nodes'][v['destination']]['pos_y']
        Rrup[k] = shortest_distance(x1, y1, x2, y2, epi_loc[0], epi_loc[1])

    # GMPE model (Campbell 2003)
    Mw = infra_data['eq'][eq_name]['Mw']
    ln_Sa, std_al, std_ep = gmpe_cam03(Mw, Rrup)

    # fragility curves
    frag_mean_beta = {}
    dam_st = 2 # Extensive damage
    for k, v in infra_data['edges'].items():

        mean = infra_data['frag'][v['fragility_type']]['Sa_g'][dam_st]
        beta = infra_data['frag'][v['fragility_type']]['Sa_g_dispersion']

        frag_mean_beta[k] = (mean, beta)

    # mean and covariance matrix between failures of roads
    VAR = {}
    COV = np.zeros(shape=(len(ln_Sa), len(ln_Sa)))
    for i1, (k1, std_ep1) in enumerate(std_ep.items()):

        std_beta = frag_mean_beta[k1][1]

        for i2, k2 in enumerate(std_ep.keys()):

            if k1==k2:
                COV[i1, i2] = std_al**2 + std_ep1**2 + std_beta**2
                VAR[k1] = std_al**2 + std_ep1**2 + std_beta**2
            else:
                COV[i1,i2] = std_al**2

    MEAN = {k: np.log(frag_mean_beta[k][0]) - v for k, v in ln_Sa.items()}


    # failure probability
    pf = {k: norm.cdf(0, MEAN[k], np.sqrt(VAR[k])) for k in MEAN.keys()}

    return pf, MEAN, VAR, COV, Rrup, ln_Sa, std_al, std_ep, frag_mean_beta


def shortest_distance(x1, y1, x2, y2, x0, y0):
    """
    Calculate the shortest distance between a point and a line.

    The line is defined by two points (x1, y1) and (x2, y2).
    The point is defined by (x0, y0).

    Returns:
    - distance: The shortest distance between the line and the point. In case the projection of the point is outside the line segment, the distance is calculated to the closest endpoint.
    """
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

def gmpe_cam03( Mw, Rrup ):
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


## ANALYSIS ##
cfg_name = 'config.json'
eq_name = 's1'

cfg = config_pm.Config_pm(HOME.joinpath('./input/'+cfg_name))

# raods' failure probability
pf, MEAN, VAR, COV, Rrup, ln_Sa, std_al, std_ep, frag_mean_beta = cal_edge_dist(cfg.infra, eq_name)
probs = {k: {0:v, 1:1-v} for k,v in pf.items()}

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
for k, v in cfg.infra['edges'].items():
    varis[k] = variable.Variable(name=k, values = [np.inf, arc_time[k]])

# cpms
cpms = {}
for k, v in cfg.infra['edges'].items():
    cpms[k] = cpm.Cpm([varis[k]], 1, C=np.array([[0],[1]]), p = np.array([pf[k], 1-pf[k]]))

comps_st_itc = {k: len(v.values)-1 for k,v in varis.items()}
st_br_to_cs = {'f': 0, 's': 1, 'u': 2}

dests = cfg.infra['origins']
thres = cfg.infra['thres']

sys_pfs = {}
sys_nsamps = {}

for node in cfg.infra['nodes'].keys():
#for node in ['n68']: # for test
    print(f'-----Analysis begins for node: {node}-----')

    if node not in dests:

        d_time_itc, _ = trans.get_time_and_path_multi_dest(comps_st_itc, node, dests, arcs, varis)
        sys_fun = trans.sys_fun_wrap({'origin': node, 'dests': dests}, arcs, varis, thres * d_time_itc)

        brs, rules, sys_res1, monitor1 = gen_bnb.run_brc( {k: varis[k] for k in arcs.keys()}, probs, sys_fun, 0.01*cfg.max_sys_fun, 0.01*cfg.max_branches, cfg.sys_bnd_wr, surv_first=False)
        brs, rules, sys_res2, monitor2 = gen_bnb.run_brc( {k: varis[k] for k in arcs.keys()}, probs, sys_fun, cfg.max_sys_fun, cfg.max_branches, cfg.sys_bnd_wr, surv_first=True, rules=rules)
        monitor = {k: v + monitor2[k] for k, v in monitor1.items() if k != 'out_flag'}
        monitor['out_flag'] = [monitor1['out_flag'], monitor2['out_flag']]

        csys, varis = gen_bnb.get_csys_from_brs(brs, varis, st_br_to_cs)
        varis[node] = variable.Variable(node, values = ['f', 's', 'u'])

        pf_u, pf_l = monitor['pf_up'][-1], monitor['pf_low'][-1]
        if monitor['out_flag'][-1] == 'max_sf' or monitor['out_flag'][-1] == 'max_nb':
            print(f'*Rejection sampling started..*')

            csys = csys[ csys[:,0] != st_br_to_cs['u'] ] # remove unknown state instances
            cpms[node] = cpm.Cpm( [varis[node]] + [varis[k] for k in arcs.keys()], 1, csys, np.ones((len(csys),1), dtype=float) )

            def sys_fun_rs(x):
                val, st, _ = sys_fun(x)
                if st == 's':
                    return val, 1
                elif st == 'f':
                    return val, 0

            start = time.time()
            cpm2_, result_ = cpm.rejection_sampling_sys(cpms, node, sys_fun_rs, cfg.cov_t, 0, 1.0-pf_l-pf_u, pf_l, rand_seed=0 )
            cpms[node] = copy.deepcopy(cpm2_)
            end = time.time()

            sys_pfs[node] = result_['pf']
            sys_nsamps[node] = result_['cov']

            fout_rs = output_path.joinpath(f'rs_{node}.txt')
            with open(fout_rs, 'w') as f:
                for k, v in result_.items():
                    if k in ['pf', 'cov']:
                        f.write(f"{k}\t{v:.4e}\n")
                    elif k in ['nsamp', 'nsamp_tot']:
                        f.write(f"{k}\t{v:d}\n")
                f.write(f"time (sec)\t{end-start:.4e}\n")

        else:
            cpms[node] = cpm.Cpm( [varis[node]] + [varis[k] for k in arcs.keys()], 1, csys, np.ones((len(csys),1), dtype=float) )
            sys_pfs[node] = pf_u
            sys_nsamps[node] = 0

        # save results
        fout_monitor = output_path.joinpath(f'brc_{node}.pk')
        with open(fout_monitor, 'wb') as fout:
            pickle.dump(monitor, fout)

    else:

        sys_pfs[node] = 0
        sys_nsamps[node] = 0

    print(f'***Analysis completed***')

# save results
fout = output_path.joinpath(f'result.txt')
with open(fout, 'w') as f:
    for k, v in sys_pfs.items():
        f.write(f'{k}\t{v:.4e}\t{sys_nsamps[k]}\n')

fout_varis = output_path.joinpath(f'varis.pk')
with open(fout_varis, 'wb') as fout:
    pickle.dump(varis, fout)

fout_cpm = output_path.joinpath(f'cpms.pk')
with open(fout_cpm, 'wb') as fout:
    pickle.dump(cpms, fout)

print(f'-----All nodes completed. Results saved-----')

