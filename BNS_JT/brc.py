import pandas as pd
import copy
from pathlib import Path
from collections import Counter
from itertools import chain
import matplotlib.pyplot as plt
import warnings
import sys
import pickle
import numpy as np
import time
#import editdistance

from BNS_JT import variable, branch


def run(probs, sys_fun, rules=None, brs = None, max_sf = np.inf, max_nb = np.inf, pf_bnd_wr=0.0, max_rules = np.inf, surv_first=True, active_decomp = 20, final_decomp = True, display_freq = 200):

    """
    Run the BRC algorithm to find (1) non-dominated rules and
    (2) branches for system reliability analysis.

    Attributes:
        probs (dictionaty): {comp_name (str): probabilities (list)}
        sys_fun: a system function
            One attribute:
                comp_state (dictionary): {comp_name (str): state (int)}
            Returns (orders need to be kept):
                system value (any type)
                system state ('s' or 'f')
                minimum rule for system state (dictionary):
                    {comp_name (str): state (int)}

        **Information from previous analysis (optional when available)**
            rules (dictionary): {'s': list of rules, 'f': list of rules}
            brs (list): branches from previous analysis

        **Iteration termination conditions**
            max_sf (int): maximum number of system function runs
            max_nb (int): maximum number of branches
            pf_bnd_wr (float, non-negative): bound of system failure probability
                in ratio (width / lower bound)
            max_rules (int): the maximum number of rules
        **Decomposition options**
            surv_first: True if survival branches are considered first
            active_decomp: True if branches are re-obtained at each iteration
                False if branches are never re-evaluated
                int (>0) if regularly re-evaluated
            final_decomp: True if final decomposition is performed
                (only when active_decomp = False)
        **Display options**
            display_freq (int): frequency of displaying the current progress

    Returns:
        brs (list): branches
        rules (dictionary): {'s': list of rules, 'f': list of rules}
        sys_res (pandas.DataFrame): system function results
        monitor (dictionary): monitoring information
    """

    if not rules:
        rules = {'s': [], 'f': []}

    sys_res = pd.DataFrame(data={'sys_val': [],
                                 'comp_st': [],
                                 'comp_st_min': []}) # system function results

    monitor, ctrl = init_monitor()

    while ctrl['no_sf'] < max_sf:

        start = time.time() # monitoring purpose

        if active_decomp is True:
            brs, _ = decomp_depth_first(rules, probs, max_nb)  # S2
        elif active_decomp is False or active_decomp==0:

            if brs is None:
                brs = []
            brs, _ = decomp_depth_first(rules, probs, max_nb, brs) # existing branches are not reassessed.
        else:
            if ctrl['no_sf'] % active_decomp == 0:
                brs, _ = decomp_depth_first(rules, probs, max_nb)  # S2

            else:
                if brs is None:
                    brs = []
                brs, _ = decomp_depth_first(rules, probs, max_nb, brs) # existing branches are not reassessed.

        x_star = get_comp_st(brs, surv_first, probs)  # S4-1

        if x_star == None:
            monitor['out_flag'] = 'complete'
            break

        elif len(brs) >= max_nb:
            monitor['out_flag'] = 'max_nb'
            break

        elif ctrl['pr_bu'] < ctrl['pf_low'] * pf_bnd_wr:
            monitor['out_flag'] = 'pf_bnd'
            break

        elif len(rules['s']+rules['f']) >= max_rules:
            monitor['out_flag'] = 'max_rules'
            break

        else:
            rule, sys_res_ = run_sys_fn(x_star, sys_fun, probs) # S4-2, S5

            rules = update_rule_set(rules, rule) # S6
            sys_res = pd.concat([sys_res, sys_res_], ignore_index=True)
            ctrl['no_sf'] += 1

            monitor['no_sf'].append(ctrl['no_sf'])
            monitor, ctrl = update_monitor(monitor, brs, rules, start) # S7

            if ctrl['no_sf'] % display_freq == 0:
                print(f"[System function runs {ctrl['no_sf']}]..")
                display_msg(monitor)

        if ctrl['no_sf'] == max_sf:
            monitor['out_flag'] = 'max_sf'

    try:
        if final_decomp and (active_decomp is False or active_decomp > 1):
            nbr_old = len(brs)
            brs, _ = decomp_depth_first(rules, probs, max_nb)
            print(f"\n*Final decomposition is completed with {len(brs)} branches (originally {nbr_old} branches).")

        monitor, ctrl = update_monitor(monitor, brs, rules, start)

        print(f"\n***Analysis completed with f_sys runs {ctrl['no_sf']}: out_flag = {monitor['out_flag']}***")
        display_msg(monitor)

    except NameError: # analysis is terminated before the first system function run
        print(f'\n***Analysis terminated without any evaluation***')

    return brs, rules, sys_res, monitor


def init_monitor():

    monitor = {'pf_up': [], # upper bound on pf
               'pf_low': [], # lower bound on pf
               'pr_bu': [], # prob. of unknown branches
               'no_br': [], # number of branches
               'no_bs': [], # number of branches-survival (br_s_ns)
               'no_bf': [], # number of branches-failure (br_f_ns)
               'no_bu': [], # number of branches-unknown (br_us_ns)
               'no_rs': [], # number of rules-survival (r_s_ns)
               'no_rf': [], # number of rules-failure (r_f_ns)
               'no_ra': [], # number of rules (no_rs + no_rf)
               'no_sf': [0], # number of system function runs (sf_ns)
               'time': [], # time taken for each iteration (sec)
               'min_len_rf': [], # min. length of rules-failure
               'avg_len_rf': [], # avg. length of rules-failure
               'out_flag': None, # outflag ('complete', 'max_nb', 'pf_bnd', 'max_sf'),
               'max_bu': 0 # max. number of branches-unknown 
              }

    # init for ctrl
    # no_sf, pr_bf, pr_bs = 0, 0, 0
    ctrl = {'no_sf': 0, # no_sf
            'pr_bu': 1, # 1 - pr_bf - pr_bs
            'pf_low': 0, # pr_bf
            'no_bu': 1, # no_bu
            }

    return monitor, ctrl


def update_monitor(monitor, brs, rules, start):

    end = time.time() # monitoring purpose

    pr_bf = sum([br.p for br in brs if br.up_state == 'f']) # prob. of failure branches
    pr_bs = sum([br.p for br in brs if br.down_state == 's']) # prob. of survival branches

    monitor['pf_low'].append(pr_bf) # lower bound on pf
    monitor['pf_up'].append(1.0 - pr_bs)  # upper bound of pf
    monitor['pr_bu'].append(1.0 - pr_bf - pr_bs) # prob. of unknown branches

    no_rf = len(rules['f'])
    no_rs = len(rules['s'])

    monitor['no_rf'].append(no_rf)
    monitor['no_rs'].append(no_rs)
    monitor['no_ra'].append(no_rs + no_rf)

    no_br = len(brs)
    no_bf = sum([b.up_state == 'f' for b in brs])
    no_bs = sum([b.down_state == 's' for b in brs])
    no_bu = no_br - no_bf - no_bs

    monitor['no_bf'].append(no_bf)
    monitor['no_bs'].append(no_bs)
    monitor['no_bu'].append(no_bu)
    monitor['no_br'].append(no_br)

    monitor['time'].append(end - start)

    try:
        min_len_rf = min([len(x) for x in rules['f']])
        avg_len_rf = sum([len(x) for x in rules['f']]) / no_rf
    except ValueError:
        min_len_rf = 0
        avg_len_rf = 0

    monitor['min_len_rf'].append(min_len_rf)
    monitor['avg_len_rf'].append(avg_len_rf)

    # get the latest value for ctrl
    keys = ['no_sf', 'pr_bu', 'pf_low', 'no_bu']
    ctrl = {}
    for k in keys:
        try:
            ctrl[k] = monitor[k][-1]
        except TypeError:
            ctrl[k] = monitor[k]

    return monitor, ctrl


def display_msg(monitor):

    last = {}
    for k, v in monitor.items():
        try:
            last[k] = v[-1]
        except TypeError:
            last[k] = v

    print(f"The # of found non-dominated rules (f, s): {last['no_ra']} ({last['no_rf']}, {last['no_rs']})")
    print(f"Probability of branchs (f, s, u): ({last['pf_low']:.4e}, {1-last['pf_up']:.2e}, {last['pr_bu']:.4e})")
    print(f"The # of branches (f, s, u), (min, avg) len of rf: {last['no_br']} ({last['no_bf']}, {last['no_bs']}, {last['no_bu']}), ({last['min_len_rf']}, {last['avg_len_rf']:.2f})")
    print(f"Elapsed seconds (average per round): {sum(monitor['time']):1.2e} ({np.mean(monitor['time']):1.2e})")


def plot_monitoring(monitor, output_file='monitor.png'):
    """

    """

    # bounds vs no. of branches
    fig = plt.figure(figsize=(6, 4*3))
    ax = fig.add_subplot(311)
    ax.plot(monitor['no_br'], monitor['pf_low'], linestyle='--', color='blue')
    ax.plot(monitor['no_br'], monitor['pf_up'], linestyle='--', color='blue')
    ax.set_xlabel('No. of branches')
    ax.set_ylabel('System failure prob. bounds')

    # bounds vs sys fn runs
    ax = fig.add_subplot(312)
    ax.plot(monitor['no_sf'], monitor['pf_low'], linestyle='--', color='blue')
    ax.plot(monitor['no_sf'], monitor['pf_up'], linestyle='--', color='blue')
    ax.set_xlabel('No. of system function runs')
    ax.set_ylabel('System failure prob. bounds')

    # no. of rules vs sys fn runs
    ax = fig.add_subplot(313)
    ax.plot(monitor['no_ra'], monitor['pf_low'], linestyle='--', color='blue')
    ax.plot(monitor['no_ra'], monitor['pf_up'], linestyle='--', color='blue')
    ax.set_xlabel('No. of rules')
    ax.set_ylabel('System failure prob. bounds')

    #output_file = Path(sys.argv[1]).joinpath(output_file)
    fig.savefig(output_file, dpi=200)
    print(f'{output_file} created')


def get_csys(brs, varis, st_br_to_cs):
    """

    """
    c_sys = np.empty(shape=(0, len(brs[0].up.keys()) + 1), dtype=int)

    for br in brs:
        varis, c = br.get_c(varis, st_br_to_cs)
        c_sys = np.vstack([c_sys, c])

    return c_sys, varis


def get_state(comp, rules):
    """
    Args:
        comp (dict): component state vector in dictionary
                     e.g., {'x1': 0, 'x2': 0 ... }
        rules (list): a list of rules
                     e.g., {({'x1': 2, 'x2': 2}, 's')}
    Returns:
        str: system state ('s', 'f', or 'u')
    """
    assert isinstance(comp, dict), f'comp should be a dict: {type(comp)}'
    assert isinstance(rules, dict), f'rules should be a dict: {type(rules)}'

    # the survival rule is satisfied
    s_rules = [i for (i,rule) in enumerate(rules['s']) if all([comp[k] >= v for k, v in rule.items()])]
    no_s = len(s_rules)

    f_rules = [i for (i,rule) in enumerate(rules['f']) if all([comp[k] <= v for k, v in rule.items()])]
    no_f = len(f_rules)

    # no compatible rules. the state remains unknown
    if no_s == no_f == 0:
        state = 'u'
    elif no_s >= no_f:
        state = 's'
    else:
        state = 'f'

    if no_s > 0 and no_f > 0:
        rules_s = [rules['s'][i] for i in s_rules]
        rules_f = [rules['f'][i] for i in f_rules]

        print(f"Conflicting rules found: {rules_s} vs. {rules_f}. The given system is not coherent.")

    return state


def update_rule_set(rules, new_rule):
    """
    rules: list of rules
           e.g., [({'x1': 2, 'x2': 2}, 's')]
    new_rule: a rule
             e.g., ({'x1': 2}, 's')
    """
    assert isinstance(new_rule, tuple), f'rule should be a tuple: {type(new_rule)}'
    add_rule = True

    n_rule, n_state = new_rule
    #rules_old = copy.deepcopy(rules)

    if n_state == 's':

        for rule in rules['s'][:]:

            if set(n_rule).issubset(rule) and all([rule[k] >= v for k, v in n_rule.items()]):
                rules['s'].remove(rule)

            elif set(rule).issubset(n_rule) and all([n_rule[k] >= v for k, v in rule.items()]):
                add_rule = False
                break

    elif n_state == 'f':

        for rule in rules['f'][:]:

            if set(n_rule).issubset(rule) and all([rule[k] <= v for k, v in n_rule.items()]):
                rules['f'].remove(rule)

            elif set(rule).issubset(n_rule) and all([n_rule[k] <= v for k, v in rule.items()]):
                add_rule = False
                break

    if add_rule:
        rules[n_state].append(n_rule)

    return rules


def run_sys_fn(comp, sys_fun, probs):
    """
    comp: component vector state in dictionary
    e.g., {'x1': 0, 'x2': 0, ... }
    sys_fun
    rules: list of rules
           e.g., {({'x1': 2, 'x2': 2}, 's')}
    """

    assert isinstance(comp, dict), f'comp should be a dict: {type(comp)}'

    # S4-2: get system state given comp
    sys_val, sys_st, comp_st_min = sys_fun(comp)

    sys_res = pd.DataFrame({'sys_val': [sys_val], 'comp_st': [comp], 'comp_st_min': [comp_st_min]})

    if comp_st_min:
        rule = comp_st_min, sys_st

    else:
        if sys_st == 's':
            rule = {k: v for k, v in comp.items() if v}, sys_st # the rule is the same as up_dict but includes only components whose state is greater than the worst one (i.e. 0)
        else:
            rule = {k: v for k, v in comp.items() if v < len(probs[k].keys()) - 1}, sys_st # the rule is the same as up_dict but includes only components whose state is less than the best one

    return rule, sys_res


def init_branch(probs, rules):
    """
    initialise a branch set (x_min, x_max, s(x_min), s(x_max), 1)
    """

    down = {x: 0 for x in probs.keys()}
    up = {k: len(v.keys()) - 1 for k, v in probs.items()}

    down_state = get_state(down, rules)
    up_state = get_state(up, rules)

    return [branch.Branch(down, up, down_state, up_state, 1.0)]


def decomp_depth_first(rules, probs, max_nb, brs = []):
    """
    depth-first decomposition of event space using given rules
    """

    if len(brs) < 1:
        brs = init_branch(probs, rules)  # D1
    #crules = [brs[0].get_compat_rules(rules)]
    crules = [br.get_compat_rules(rules) for br in brs]

    go = True
    while go:

        # D2: sort branches from higher to lower p 
        sorted_brs = sorted(zip(brs, crules), key=lambda x: x[0].p, reverse=True)
        brs, crules = [list(x) for x in zip(*sorted_brs)]

        brs_new = []
        crules_new = []

        for i, (br, cr) in enumerate(sorted_brs, 1):

            br.eval_state(rules)

            # specified branch or no compatible rule exists
            if ((br.down_state == br.up_state) and (br.down_state != 'u')) or (len(cr['f']) + len(cr['s']) == 0):
                brs_new.append(br)
                crules_new.append({'s':[], 'f':[]})

            else:
                # D6??
                xd, xd_st = br.get_decomp_comp_using_probs(cr, probs)

                # D3: evaluate sl and su
                for up_flag in [True, False]:
                    br_new = br.get_new_branch(rules, probs, xd, xd_st, up_flag)
                    crule_new = br_new.get_compat_rules(rules)
                    brs_new.append(br_new)
                    crules_new.append(crule_new)

                #n_br = len(brs_new) + len(brs) - i # the current number of branches

                if len(brs_new) + len(brs) > max_nb:

                    go = False
                    brs_new += brs[i:]
                    crules_new += crules[i:]
                    break

        brs = copy.deepcopy(brs_new)
        crules = copy.deepcopy(crules_new)

        if go and sum([len(r['f']) + len(r['s']) for r in crules]) == 0:
           go = False

    return brs, crules


def get_comp_st(brs, surv_first=True, probs=None):
    """
    get a component vector state from branches(brs)
    'brs' is a list of branches obtained by depth-first decomposition
    """

    if surv_first:

        brs = sorted(brs, key=lambda x: x.p, reverse=True)

        x_star = None
        for br in brs: # look at up_state first
            if br.up_state == 'u':
                x_star = br.up
                break

        if x_star == None:
            for br in brs: # if all up states are known then down states
                if br.down_state == 'u':
                    x_star = br.down
                    break

    else:

        worst = {x: 0 for x in probs.keys()}
        best = {k: len(v.keys()) - 1 for k, v in probs.items()}

        brs_new = []
        for br in brs:
            if br.up_state == 'u':
                p_new = branch.approx_prob_by_comps(br.up, best, probs)
                br_new = branch.Branch(br.up, best, 'u', 's', p_new)
                brs_new.append(br_new)

            if br.down_state =='u':
                p_new = branch.approx_prob_by_comps(worst, br.down, probs)
                b_new = branch.Branch(worst, br.down, 'f', 'u', p_new)
                brs_new.append(br_new)

        x_star = None
        if brs_new:
            brs_new = sorted(brs_new, key=lambda x: x.p, reverse=True)
            if brs_new[0].up_state == 'u':
                x_star = brs_new[0].up
            elif brs_new[0].down_state == 'u':
                x_star = brs_new[0].down

    return x_star


def run_MCS_indep_comps(probs, sys_fun, cov_t = 0.01):
    nsamp, nfail = 0, 0
    cov = 1.0
    while cov > cov_t:

        # generate samples
        nsamp += 1
        samp = {}
        for k, v in probs.items():
            st1 = np.random.choice(list(v.keys()), size=1, p=list(v.values()))
            samp[k] = st1[0]

        # run system function
        _, sys_st, _ = sys_fun(samp)

        if sys_st == 'f':
            nfail += 1

            pf = nfail / nsamp
            if nfail > 5:
                std = np.sqrt( pf*(1-pf) / nsamp )
                cov = std / pf

        if nsamp % 20000 == 0:
            print(f'nsamp: {nsamp}, cov: {cov}, pf: {pf}')

    return pf, cov, nsamp

def eval_rules_prob( rules_list, s_or_f, probs ):
    
    probs_list = []

    if s_or_f == 'f': # rules_list contain failure rules
        for r in rules_list:
            p = 1.0
            for k, v in r.items():
                p *= sum([probs[k][x] for x in range(v+1)])

            probs_list.append(p)

    elif s_or_f == 's':
        max_st = {k: max(probs[k].keys()) for k in probs.keys()}

        for r in rules_list:
            p = 1.0
            for k, v in r.items():
                p *= sum([probs[k][x] for x in range(v, max_st[k]+1)])

            probs_list.append(p)
    else:
        raise ValueError('s_or_f input variable must be a string either "s" or "f".')
    
    return probs_list