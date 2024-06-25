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

from BNS_JT import variable, branch


def init_branch(down, up, rules):
    """
    returns initial branch based on given rules and states
    Args:
        down: (dict): all components in the worst state
        up: (dict): all components in the best state
        rules (list): rules list
    Returns:
        list: initial branch
    """

    down_state = get_state(down, rules)
    up_state = get_state(up, rules)

    return [branch.Branch(down, up, down_state, up_state, 1.0)]


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
               'no_sf': 0, # number of system function runs (sf_ns)
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

    pr_bf = sum([b[4] for b in brs if b.up_state == 'f']) # prob. of failure branches
    pr_bs = sum([b[4] for b in brs if b.down_state == 's']) # prob. of survival branches

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



def proposed_branch_and_bound_using_probs(sys_fun, varis, probs, max_sf, output_path=Path(sys.argv[0]).parent, key=None, flag=False):

    assert isinstance(varis, dict), f'varis must be a dict: {type(varis)}'
    assert isinstance(probs, dict), f'probs must be a dict: {type(probs)}'

    # Initialisation
    sys_res = pd.DataFrame(data={'sys_val': [], 'comp_st': [], 'comp_st_min': []}) # system function results 
    """
    no_iter, pr_bf, pr_bs, pr_bu, no_bu =  0, 0, 0, 1, 1
    no_rf, no_rs, len_rf, len_rs = 0, 0, 0, 0
    max_bu = 0
    """

    rules = {'s': [], 'f': []} # a list of known rules
    brs_new = []
    worst = {x: 0 for x in varis.keys()} # all components in the worst state
    best = {k: len(v.values) - 1 for k, v in varis.items()} # all components in the best state

    brs = init_branch(worst, best, rules)

    # For monitoring purpose (store for each iteration)
    monitor, ctrl = init_monitor()

    stop_br = False
    while ctrl['no_bu'] and (ctrl['no_sf'] < max_sf):

        start = time.time() # monitoring purpose

        #no_iter += 1

        if stop_br and ctrl['no_sf'] % 20 == 0:
            print(f"[System function runs {ctrl['no_sf']}]..")
            display_msg(monitor)

        stop_br = False
        brs = sorted(brs, key=lambda x: x.p, reverse=True)

        for br in brs:

            if (br.down_state == br.up_state) and (br.down_state != 'u'):
                brs_new.append(br)

            else:
                c_rules = get_compat_rules(br.down, br.up, rules)

                if (not c_rules['s']) or (not c_rules['f']):
                    if not c_rules['s']:
                        x_star = br.up
                    else:
                        x_star = br.down

                    # run system function
                    rule, sys_res_ = run_sys_fn(x_star, sys_fun, varis)
                    monitor['no_sf'] += 1

                    rules = update_rule_set(rules, rule)
                    sys_res = pd.concat([sys_res, sys_res_], ignore_index=True)

                    ########## FOR MONITORING ##################
                    monitor, ctrl = update_monitor(monitor, brs, rules, start)
                    #########################################

                    brs = init_branch(worst, best, rules)
                    brs_new = []
                    #no_iter = 0
                    # for loop exit
                    stop_br = True
                    break

                else:
                    xd, xd_st = get_decomp_comp_using_probs(br.down, br.up, c_rules, probs)

                    # for upper
                    br_new, _ = get_br_new(br, rules, probs, xd, xd_st)
                    brs_new.append(br_new)

                    # for lower
                    br_new, _ = get_br_new(br, rules, probs, xd, xd_st, up_flag=False)
                    brs_new.append(br_new)

        # for the for loop
        if stop_br == False:
            brs = brs_new
            brs_new = []
            monitor, ctrl = update_monitor(monitor, brs, rules, start)

        if ctrl['no_sf'] >= max_sf:
            print(f'*** Terminated due to the # of system function runs: {no_sf} >= {max_sf}')

    ########## FOR MONITORING ##################
    monitor, ctrl = update_monitor(monitor, brs, rules, start)
    #########################################

    print(f'**Algorithm Terminated**')
    print(f"[System function runs {ctrl['no_sf']}]..")
    display_msg(monitor)

    if flag:
        output_file = output_path.joinpath(f'brs_{key}.pk')
        with open(output_file, 'wb') as fout:
            pickle.dump(brs, fout)
        print(f'{output_file} is saved')

    return brs, rules, sys_res, monitor



def proposed_branch_and_bound(sys_fun, varis, max_br, output_path=Path(sys.argv[0]).parent, key=None, flag=False):

    assert isinstance(varis, dict), f'varis must be a dict: {type(varis)}'

    # Initialisation
    no_sf = 0 # number of system function runs so far
    sys_res = pd.DataFrame(data={'sys_val': [], 'comp_st': [], 'comp_st_min': []}) # system function results 
    no_iter, no_bf, no_bs, no_bu =  0, 0, 0, 1
    no_rf, no_rs, len_rf, len_rs = 0, 0, 0, 0
    max_bu = 0

    rules = {'s': [], 'f': []} # a list of known rules
    brs_new = []
    worst = {x: 0 for x in varis.keys()} # all components in the worst state
    best = {k: len(v.values) - 1 for k, v in varis.items()} # all components in the best state

    brs = init_branch(worst, best, rules)

    while no_bu and len(brs) < max_br:

        no_iter += 1
        print(f'[System function runs {no_sf}]..')
        print(f'The # of found non-dominated rules (f, s): {no_rf + no_rs} ({no_rf}, {no_rs})')
        #print('The # of branching: ', no_iter)
        #print(f'The # of branches (f, s, u): {len(brs)} ({no_bf}, {no_bs}, {no_bu})')
        stop_br = False

        for br in brs:

            if (br.down_state == br.up_state) and (br.down_state != 'u'):
                brs_new.append(br)

            else:
                c_rules = get_compat_rules(br.down, br.up, rules)

                if (not c_rules['s']) or (not c_rules['f']):
                    if not c_rules['s']:
                        x_star = br.up
                    else:
                        x_star = br.down

                    # run system function
                    rule, sys_res_ = run_sys_fn(x_star, sys_fun, varis)
                    no_sf += 1

                    rules = update_rule_set(rules, rule)
                    sys_res = pd.concat([sys_res, sys_res_], ignore_index=True)
                    brs = init_branch(worst, best, rules)
                    brs_new = []
                    no_iter = 0
                    # for loop exit
                    stop_br = True
                    break

                else:
                    xd, xd_st = get_decomp_comp(br.down, br.up, c_rules)

                    # for upper
                    up = br.up.copy()
                    up[xd] = xd_st - 1
                    up_state = get_state(up, rules)
                    brs_new.append(branch.Branch(br.down, up, br.down_state, up_state))

                    # for lower
                    down = br.down.copy()
                    down[xd] = xd_st
                    down_state = get_state(down, rules)
                    brs_new.append(branch.Branch(down, br.up, down_state, br.up_state))

        # for the for loop
        if stop_br == False:
            brs = brs_new
            brs_new = []

        #ok = any([(b.up_state == 'u') or (b.down_state == 'u') or (b.down_state != b.up_state) for b in brs])  # exit for loop
        no_bf = sum([(b.up_state == 'f') for b in brs]) # no. of failure branches
        no_bs = sum([(b.down_state == 's') for b in brs]) # no. of survival branches
        no_bu = sum([(b.up_state == 'u') or (b.down_state == 'u') or (b.down_state != b.up_state) for b in brs]) # no. of unknown branches
        if no_bu > max_bu:
            max_bu = no_bu

        no_rf = len(rules['f']) # no. of failure rules
        no_rs = len(rules['s']) # no. of survival rules
        """
        if no_rf > 0:
            len_rf = sum([len(x) for x in rules['f']])/no_rf # mean length of failure rules
        else:
            len_rf = 0

        if no_rs > 0:
            len_rs = sum([len(x) for x in rules['s']])/no_rs # mean length of survival rules
        else:
            len_rs = 0
        """
        print(f'# of unknown branches to go: {no_bu}, {max_bu}\n')
        if len(brs) >= max_br:
            print(f'*** Terminated due to the # of branches: {len(brs)} >= {max_br}')

    print(f'**Algorithm Terminated**')

    if flag:
        output_file = output_path.joinpath(f'brs_{key}.pk')
        with open(output_file, 'wb') as fout:
            pickle.dump(brs, fout)
        print(f'{output_file} is saved')

    return brs, rules, sys_res


def get_csys_from_brs(brs, varis, st_br_to_cs):
    """


    """
    c_sys = np.empty(shape=(0, len(brs[0].up.keys()) + 1), dtype=int)

    for br in brs:
        varis, c = get_c_from_br(br, varis, st_br_to_cs)
        c_sys = np.vstack([c_sys, c])

    return c_sys, varis


def get_c_from_br(br, varis, st_br_to_cs):
    """
    return updated varis and state
    br: a single branch
    varis: a dictionary of variables
    st_br_to_cs: a dictionary that maps state in br to state in C matrix of a system event
    """
    names = list(br.up.keys())
    cst = np.zeros(len(names) + 1, dtype=int) # (system, compponents)

    if br.down_state == br.up_state:
        cst[0] = st_br_to_cs[br.down_state]
    else:
        cst[0] = st_br_to_cs['u']

    for i, x in enumerate(names):
        down = br.down[x]
        up = br.up[x]

        if up > down:
            states = list(range(down, up + 1))
            varis[x], st = variable.get_composite_state(varis[x], states)
        else:
            st = up

        cst[i + 1] = st

    return varis, cst


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
    no_s = sum([all([comp[k] >= v for k, v in rule.items()])
                for rule in rules['s']])

    no_f = sum([all([comp[k] <= v for k, v in rule.items()])
                 for rule in rules['f']])

    # no compatible rules. the state remains unknown
    if no_s == no_f == 0:
        state = 'u'
    elif no_s >= no_f:
        state = 's'
    else:
        state = 'f'

    if no_s > 0 and no_f > 0:
        rules_s = [rule for rule in rules['s'] if all([comp[k] >= v for k, v in rule.items()])]
        rules_f = [rule for rule in rules['f'] if all([comp[k] <= v for k, v in rule.items()])]

        print(f"Conflicting rules found: {rules_s} vs. {rules_f}. The given system is not coherent.")

    return state


def get_compat_rules(lower, upper, rules):
    """
    lower: lower bound on component vector state in dictionary
           e.g., {'x1': 0, 'x2': 0 ... }
    upper: upper bound on component vector state in dictionary
           e.g., {'x1': 2, 'x2': 2 ... }
    rules: dict of rules
           e.g., {'s': [{'x1': 2, 'x2': 2}],
                  'f': [{'x1': 2, 'x2': 0}]}
    """
    assert isinstance(lower, dict), f'lower should be a dict: {type(lower)}'
    assert isinstance(upper, dict), f'upper should be a dict: {type(upper)}'
    assert isinstance(rules, dict), f'rules should be a dict: {type(rules)}'

    compat_rules = {'s': [], 'f': []}
    for rule in rules['s']:
        if all([upper[k] >= v for k, v in rule.items()]):
            c_rule = {k: v for k, v in rule.items() if v > lower[k]}
            if c_rule:
                compat_rules['s'].append(c_rule)

    for rule in rules['f']:
        if all([lower[k] <= v for k, v in rule.items()]):
            c_rule = {k: v for k, v in rule.items() if v < upper[k]}
            if c_rule:
                compat_rules['f'].append(c_rule)

    return compat_rules


def get_compat_rules_list(lower, upper, rules):
    """
    lower: lower bound on component vector state in dictionary
           e.g., {'x1': 0, 'x2': 0 ... }
    upper: upper bound on component vector state in dictionary
           e.g., {'x1': 2, 'x2': 2 ... }
    rules: list of rules
           e.g., {({'x1': 2, 'x2': 2}, 's')}
    """
    assert isinstance(lower, dict), f'lower should be a dict: {type(lower)}'
    assert isinstance(upper, dict), f'upper should be a dict: {type(upper)}'
    assert isinstance(rules, list), f'rules should be a list: {type(rules)}'

    compat_rules = []
    for rule in rules:
        if rule[1] == 's':
            if all([upper[k] >= v for k, v in rule[0].items()]): # the survival rule is satisfied
                compat_rules.append(rule)

        elif rule[1] == 'f':
            if all([lower[k] <= v for k, v in rule[0].items()]): # the failure rule is compatible
                compat_rules.append(rule)

    return compat_rules


def approx_joint_prob_compat_rule(lower, upper, rule, rule_st, probs):

    assert isinstance(lower, dict), f'lower should be a dict: {type(lower)}'
    assert isinstance(upper, dict), f'upper should be a dict: {type(upper)}'
    assert isinstance(rule, dict), f'rules should be a dict: {type(rules)}'

    p = 1.0
    if rule_st == 's':
        for x, v in rule.items():
            p *= sum([probs[x][i] for i in range(v, upper[x] + 1)])

    elif rule_st == 'f':
        for x, v in rule.items():
            p *= sum([probs[x][i] for i in range(lower[x], v + 1)])

    return p


def approx_branch_prob(lower, upper, probs):
    """

    """
    p = 1.0

    #for dx, ux in zip(lower.items(), upper.items()):
    for k, v in lower.items():
        p *= sum([probs[k][x] for x in range(v, upper[k] + 1)])
        #p *= sum([probs[dx[0]][x] for x in range(dx[1], ux[1] + 1)])

    return p


def get_decomp_comp_using_probs(lower, upper, rules, probs):
    """
    lower: lower bound on component vector state in dictionary
           e.g., {'x1': 0, 'x2': 0 ... }
    upper: upper bound on component vector state in dictionary
           e.g., {'x1': 2, 'x2': 2 ... }
    rules: list of rules
           e.g., {({'x1': 2, 'x2': 2}, 's')}
    probs: dict
    """
    assert isinstance(lower, dict), f'lower should be a dict: {type(lower)}'
    assert isinstance(upper, dict), f'upper should be a dict: {type(upper)}'
    assert isinstance(rules, dict), f'rules should be a dict: {type(rules)}'
    """
    #amended rules
    a_rules = []
    for rule in rules:
        if rule[1] == 'f':
            a_rules.append(({k: v + 1 for k, v in rule[0].items()}, 'f'))
        else:
            a_rules.append(rule)
    """
    # get an order of x by their frequency in rules
    _rules = [x for rule in rules.values() for x in rule]
    _rules_st = [k for k, rule in rules.items() for x in rule]
    comp = Counter(chain.from_iterable(_rules))
    comp = [x[0] for x in comp.most_common()]

    P = {}
    for rule, rule_st in zip(_rules, _rules_st):

        P[tuple(rule)] = approx_joint_prob_compat_rule(lower, upper, rule, rule_st, probs)

    # get an order R by P  (higher to lower)
    a_rules = sorted(_rules, key=lambda x: P[tuple(x)], reverse=True)

    for rule in a_rules:
        for c in comp:
            if c in rule:
                if rule in rules['s']:
                    st = rule[c]
                else:
                    st = rule[c] + 1

                if (lower[c] < st) and (st <= upper[c]):
                    xd = c, st
                    break
        else:
            continue
        break

    return xd


def get_decomp_comp(lower, upper, rules):
    """
    lower: lower bound on component vector state in dictionary
           e.g., {'x1': 0, 'x2': 0 ... }
    upper: upper bound on component vector state in dictionary
           e.g., {'x1': 2, 'x2': 2 ... }
    rules: list of rules
           e.g., {({'x1': 2, 'x2': 2}, 's')}
    """
    assert isinstance(lower, dict), f'lower should be a dict: {type(lower)}'
    assert isinstance(upper, dict), f'upper should be a dict: {type(upper)}'
    assert isinstance(rules, dict), f'rules should be a dict: {type(rules)}'

    # get an order of x by their frequency in rules
    _rules = [x for rule in rules.values() for x in rule]
    comp = Counter(chain.from_iterable(_rules))
    comp = [x[0] for x in comp.most_common()]

    # get an order R by cardinality
    a_rules = sorted(_rules, key=len)

    for rule in a_rules:
        for c in comp:
            if c in rule:
                if (rule in rules['s']) and (lower[c] < rule[c]) and (rule[c] <= upper[c]):
                    xd = c, rule[c]
                    break
                if (rule in rules['f']) and (lower[c] <= rule[c]) and (rule[c] < upper[c]):
                    xd = c, rule[c] + 1
                    break
        else:
            continue
        break

    return xd


def get_decomp_comp_old(lower, upper, rules):
    """
    lower: lower bound on component vector state in dictionary
           e.g., {'x1': 0, 'x2': 0 ... }
    upper: upper bound on component vector state in dictionary
           e.g., {'x1': 2, 'x2': 2 ... }
    rules: list of rules
           e.g., {({'x1': 2, 'x2': 2}, 's')}
    """
    assert isinstance(lower, dict), f'lower should be a dict: {type(lower)}'
    assert isinstance(upper, dict), f'upper should be a dict: {type(upper)}'
    assert isinstance(rules, list), f'rules should be a list: {type(rules)}'

    #amended rules
    a_rules = []
    for rule in rules:
        if rule[1] == 'f':
            a_rules.append(({k: v + 1 for k, v in rule[0].items()}, 'f'))
        else:
            a_rules.append(rule)

    # get an order of x by their frequency in rules
    comp = Counter(chain.from_iterable([rule[0].keys() for rule in a_rules]))
    comp = [x[0] for x in comp.most_common()]

    for c in comp:
        # get an order of states of x by freq in rule
        values = sorted([rule[0][c] for rule in a_rules if c in rule[0]])
        values = [x[0] for x in Counter(values).most_common()]

        for v in values:
            if (lower[c] < v) and (v <= upper[c]) and (lower[c] < upper[c]):
                xd = c, v
                break
        else:
            continue
        break

    return xd


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

    if n_state == 's':

        for rule in rules['s']:

            if set(n_rule).issubset(rule) and all([rule[k] >= v for k, v in n_rule.items()]):
                rules['s'].remove(rule)

            elif set(rule).issubset(n_rule) and all([n_rule[k] >= v for k, v in rule.items()]):
                add_rule = False
                break

    elif n_state == 'f':

        for rule in rules['f']:

            if set(n_rule).issubset(rule) and all([rule[k] <= v for k, v in n_rule.items()]):
                rules['f'].remove(rule)

            elif set(rule).issubset(n_rule) and all([n_rule[k] <= v for k, v in rule.items()]):
                add_rule = False
                break

    if add_rule:
        rules[n_state].append(n_rule)

    return rules


def run_sys_fn(comp, sys_fun, varis):
    """
    comp: component vector state in dictionary
    e.g., {'x1': 0, 'x2': 0, ... }
    sys_fun
    rules: list of rules
           e.g., {({'x1': 2, 'x2': 2}, 's')}
    """

    assert isinstance(comp, dict), f'comp should be a dict: {type(comp)}'
    sys_val, sys_st, comp_st_min = sys_fun(comp)
    sys_res = pd.DataFrame({'sys_val': [sys_val], 'comp_st': [comp], 'comp_st_min': [comp_st_min]})

    if comp_st_min:
        rule = comp_st_min, sys_st
    else:
        if sys_st == 's':
            rule = {k: v for k, v in comp.items() if v}, sys_st # the rule is the same as up_dict_i but includes only components whose state is greater than the worst one (i.e. 0)
        else:
            rule = {k: v for k, v in comp.items() if v < len(varis[k].values) - 1}, sys_st # the rule is the same as up_dict_i but includes only components whose state is less than the best one

    return rule, sys_res


def get_br_new(br, rules, probs, xd, xd_st, up_flag=True):
    """

    """
    if up_flag:
        up = br.up.copy()
        up[xd] = xd_st - 1
        up_st = get_state(up, rules)

        down = br.down
        down_st = br.down_state

    else:
        up = br.up
        up_st = br.up_state

        down = br.down.copy()
        down[xd] = xd_st
        down_st = get_state(down, rules)

    p = approx_branch_prob(down, up, probs)
    br_new = branch.Branch(down, up, down_st, up_st, p)

    if down_st == up_st:
        if (~up_flag and down_st != 'u') or (up_flag and up_st != 'u'):
           cr_new = {'s':[], 'f':[]}
        else:
           cr_new = get_compat_rules(br_new.down, br_new.up, rules)
    else:
        cr_new = get_compat_rules(br_new.down, br_new.up, rules)

    return br_new, cr_new


def decomp_depth_first(varis, rules, probs, max_nb):
    """
    depth-first decomposition of event space using given rules
    """
    worst = {x: 0 for x in varis.keys()}
    best = {k: len(v.values) - 1 for k, v in varis.items()}

    brs = init_branch(worst, best, rules)
    brs_cr = [get_compat_rules(brs[0].down, brs[0].up, rules)]
    brs_new = []
    brs_new_cr = []

    go = True
    while go:

        brs_crs = sorted(zip(brs, brs_cr), key=lambda x: x[0].p, reverse=True)
        brs, brs_cr = zip(*brs_crs)
        brs, brs_cr = list(brs), list(brs_cr)

        for i, (br, c_rules) in enumerate(zip(brs, brs_cr)):

            if (br.down_state == br.up_state) and (br.down_state != 'u'):
                brs_new.append(br)
                brs_new_cr.append({'s':[], 'f':[]})

            elif len(c_rules['f']) + len(c_rules['s']) < 1: # c_rules is empty                    
                brs_new.append(br)
                brs_new_cr.append({'s':[], 'f':[]})

            else:
                xd, xd_st = get_decomp_comp_using_probs(br.down, br.up, c_rules, probs)

                # for upper
                br_new, cr_new = get_br_new(br, rules, probs, xd, xd_st)
                brs_new.append(br_new)
                brs_new_cr.append(cr_new)

                # for lower
                br_new, cr_new = get_br_new(br, rules, probs, xd, xd_st, up_flag=False)
                brs_new.append(br_new)
                brs_new_cr.append(cr_new)

                n_br = len(brs_new) + len(brs) - (i + 1) # the current number of branches
                if n_br >= max_nb:
                    go = False

                    try:
                        brs_new = brs_new + brs[(i + 1):]
                        brs_new_cr = brs_new_cr + brs_cr[(i + 1):]
                    except TypeError:
                        pass
                    break

        brs = copy.deepcopy(brs_new)
        brs_cr = copy.deepcopy(brs_new_cr)
        brs_new = []
        brs_new_cr = []

        if go:
            ncr = [len(cr['f']) + len(cr['s']) for cr in brs_cr]
            if all(x==0 for x in ncr):
                go = False

    return brs, brs_cr


def get_st_decomp(brs, surv_first=True, varis=None, probs=None):
    """
    'brs' is a list of branches obtained by depth-first decomposition
    """
    brs = sorted(brs, key=lambda x: x.p, reverse=True)

    if surv_first:
        x_star = None
        for b in brs: # look at up_state first
            if b.up_state == 'u':
                x_star = b.up
                break

        if x_star == None:
            for b in brs: # if all up states are known then down states
                if b.down_state == 'u':
                    x_star = b.down
                    break

    else:
        worst = {x: 0 for x in varis.keys()}
        best = {k: len(v.values) - 1 for k, v in varis.items()}

        brs_new = []
        for b in brs:
            if b.up_state == 'u':
                p_new = approx_branch_prob(b.up, best, probs)
                b_new = branch.Branch(b.up, best, 'u', 's', p_new)
                brs_new.append(b_new)

            if b.down_state =='u':
                p_new = approx_branch_prob(worst, b.down, probs)
                b_new = branch.Branch(worst, b.down, 'f', 'u', p_new)
                brs_new.append(b_new)

        if len(brs_new) < 1:
            x_star = None

        else:
            brs_new = sorted(brs_new, key=lambda x: x.p, reverse=True)
            if brs_new[0].up_state == 'u':
                x_star = brs_new[0].up
            elif brs_new[0].down_state == 'u':
                x_star = brs_new[0].down

    return x_star



def run_brc(varis, probs, sys_fun, max_sf, max_nb, pf_bnd_wr=0.0, surv_first=True, rules=None):

    """
    INPUTS:
    varis: a dictionary of variables
    probs: a dictionary of probabilities
    sys_fun: a system function
    **Iteration termination conditions**
    max_sf: maximum number of system function runs
    max_nb: maximum number of branches
    pf_bnd_wr: bound of system failure probability in ratio (width / lower bound)
    surv_first: True if survival branches are considered first
    ************************************
    rules: provided if there are some known rules
    """


    if rules == None:
        rules = {'s': [], 'f': []}

    sys_res = pd.DataFrame(data={'sys_val': [], 'comp_st': [], 'comp_st_min': []}) # system function results

    monitor, ctrl = init_monitor()

    while ctrl['no_sf'] < max_sf:

        start = time.time() # monitoring purpose

        brs, _ = decomp_depth_first(varis, rules, probs, max_nb)
        x_star = get_st_decomp(brs, surv_first, varis, probs)

        if x_star == None:
            monitor['out_flag'] = 'complete'
            break

        elif len(brs) >= max_nb:
            monitor['out_flag'] = 'max_nb'
            break

        elif ctrl['pr_bu'] < ctrl['pf_low']*pf_bnd_wr:
            monitor['out_flag'] = 'pf_bnd'
            break

        else:
            rule, sys_res_ = run_sys_fn(x_star, sys_fun, varis)

            rules = update_rule_set(rules, rule)
            sys_res = pd.concat([sys_res, sys_res_], ignore_index=True)
            monitor['no_sf'] += 1

            monitor, ctrl = update_monitor(monitor, brs, rules, start)

            if ctrl['no_sf'] % 20 == 0:
                print(f"[System function runs {ctrl['no_sf']}]..")
                display_msg(monitor)

        if not (ctrl['no_sf'] < max_sf):
            monitor['out_flag'] = 'max_sf'

    brs, _ = decomp_depth_first(varis, rules, probs, max_nb)

    try:
        monitor, ctrl = update_monitor(monitor, brs, rules, start)

        print(f"*** Analysis completed with f_sys runs {ctrl['no_sf']}: out_flag = {monitor['out_flag']} ***")
        display_msg(monitor)

    except NameError: # analysis is terminated before the first system function run
        print(f'***Analysis terminated without any evaluation***')

    return brs, rules, sys_res, monitor


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

        if nsamp%20000 == 0:
            print(f'nsamp: {nsamp}, cov: {cov}, pf: {pf}')

    return pf, cov, nsamp
