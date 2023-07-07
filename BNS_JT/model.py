import numpy as np
from BNS_JT import variable, cpm, branch
from Trans import trans



def setup_model(cfg):
    """
    cfg: instance of config class

    """

    # Arcs (components): P(X_i | GM = GM_ob ), i = 1 .. N (= nArc)
    cpms = {}
    varis = {}

    # FIXME: only works for binary ATM
    B = np.vstack([np.eye(cfg.no_ds), np.ones(cfg.no_ds)])

    # only value is related to the scenario 
    s1 = list(cfg.scenarios['scenarios'].keys())[0]
    for k, values in cfg.scenarios['scenarios'][s1].items():
        varis[k] = variable.Variable(name=k, B=B, values=cfg.scenarios['damage_states'])
        cpms[k] = cpm.Cpm(variables = [varis[k]],
                          no_child = 1,
                          C = np.arange(len(values))[:, np.newaxis],
                          p = values)

    path_times = trans.get_all_paths_and_times(cfg.infra['ODs'].values(), cfg.infra['G'], key='time')

    # FIXME: only works for binary ATM
    lower = {k: 0 for k, _ in cfg.infra['edges'].items()}
    upper = {k: 1 for k, _ in cfg.infra['edges'].items()}

    # Travel times (systems): P(OD_j | X1, ... Xn) j = 1 ... nOD
    variables = {k: varis[k] for k in cfg.infra['edges'].keys()}
    for k, v in cfg.infra['ODs'].items():

        values = [np.inf] + sorted([y for _, y in path_times[v]], reverse=True)

        varis[k] = variable.Variable(name=k, B=np.eye(len(values)), values=values)

        path_time_idx = trans.get_path_time_idx(path_times[v], varis[k])

        # FIXME
        sb = branch.branch_and_bound(path_time_idx, lower, upper, arc_condn=1)

        c = branch.get_cmat_from_branches(sb, variables)

        cpms[k] = cpm.Cpm(variables = [varis[k]] + list(variables.values()),
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

