import numpy as np
import itertools
from dask.distributed import Variable

from BNS_JT import variable, cpm, branch, trans


def setup_model(cfg):
    """
    cfg: instance of config class

    """
    # path_times by od 
    path_times = trans.get_all_paths_and_times(cfg.infra['ODs'].values(), cfg.infra['G'], key='time')

    branches = get_branches(cfg, path_times)

    # combination of multiple ODs and scenarios
    od_scen_pairs = itertools.product(cfg.infra['ODs'].keys(), cfg.scenarios['scenarios'].keys())

    cpms_by_od_scen = {}
    varis_by_od_scen = {}

    for od, scen in od_scen_pairs:

        cpms_by_od_scen[(od, scen)], varis_by_od_scen[(od, scen)] = model_given_od_scen(cfg, path_times, od, scen, branches[od])

    return cpms_by_od_scen, varis_by_od_scen


def get_branches(cfg, path_times):

    # FIXME: only works for binary ATM
    lower = {k: 0 for k, _ in cfg.infra['edges'].items()}
    upper = {k: 1 for k, _ in cfg.infra['edges'].items()}

    # set of branches by od pair
    branches = {}
    for k, v in cfg.infra['ODs'].items():
        values = [np.inf] + sorted([y for _, y in path_times[v]], reverse=True)
        varis = variable.Variable(name=k, B=np.eye(len(values)), values=values)

        path_time_idx = trans.get_path_time_idx(path_times[v], varis)

        fl = trans.eval_sys_state(path_time_idx, lower, 1)
        fu = trans.eval_sys_state(path_time_idx, upper, 1)

        bstars = [(lower, upper, fl, fu)]

        branch.branch_and_bound(bstars, path_time_idx, arc_cond=1, output_path=cfg.output_path, key=cfg.key)

        branches[k] = branch.get_sb_saved_from_job(cfg.output_path,
                cfg.key)

    return branches


def model_given_od_scen(cfg, path_times, od, scen, branches):

    # Arcs (components): P(X_i | GM = GM_ob ), i = 1 .. N (= nArc)
    cpms = {}
    varis = {}

    # FIXME: only works for binary ATM
    B = np.vstack([np.eye(cfg.no_ds), np.ones(cfg.no_ds)])

    # scenario dependent
    for k, values in cfg.scenarios['scenarios'][scen].items():

        varis[k] = variable.Variable(name=k, B=B, values=cfg.scenarios['damage_states'])
        cpms[k] = cpm.Cpm(variables = [varis[k]],
                  no_child = 1,
                  C = np.arange(len(values))[:, np.newaxis],
                  p = values)

    # Travel times (systems): P(OD_j | X1, ... Xn) j = 1 ... nOD
    values = [np.inf] + sorted([y for _, y in path_times[cfg.infra['ODs'][od]]], reverse=True)
    varis[od] = variable.Variable(name=od, B=np.eye(len(values)), values=values)

    variables = {k: varis[k] for k in cfg.infra['edges'].keys()}
    c = branch.get_cmat_from_branches(branches, variables)

    cpms[od] = cpm.Cpm(variables = [varis[od]] + list(variables.values()),
                       no_child = 1,
                       C = c,
                       p = np.ones(c.shape[0]),
                       )

    return cpms, varis


def compute_prob(cfg, cpms, varis, var_elim, key, idx_state, flag):
    """
    var_elim: list of variable to be eliminated
    """

    assert isinstance(var_elim, list), 'var_elim should be a list'
    assert isinstance(key, str), 'key should be a str'

    ## Repeat inferences again using new functions -- the results must be the same.
    # Probability of delay and disconnection
    #M = [cpms_arc[k] for k in list(arcs.keys()) + list(var_ODs.keys())]
    M = [cpms[k] for k in var_elim + [key]]
    var_elim = [varis[i] for i in var_elim]
    M_VE2 = cpm.variable_elim(M, var_elim)

    # Retrieve example results
    # Prob. of disconnection
    #prob = np.zeros(len(varis[key].values))
    #for idx_state in enumerate(varis[key].values):
    prob = cpm.get_prob(M_VE2, [varis[key]], [idx_state], flag)

    # Prob. of delay
    #ODs_prob_delay2[j] = cpm.get_prob(M_VE2, [vars_arc[idx]], [1-1], flag=False) # Any state greater than 1 means delay.

    return prob, M_VE2

