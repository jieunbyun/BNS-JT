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


def plot_monitoring(monitor, output_file='monitor.png'):
    """

    """

    # bounds vs no. of branches
    fig = plt.figure(figsize=(6, 4*3))
    ax = fig.add_subplot(311)
    ax.plot(monitor['br_ns'], monitor['pf_low'], linestyle='--', color='blue')
    ax.plot(monitor['br_ns'], monitor['pf_up'], linestyle='--', color='blue')
    ax.set_xlabel('No. of branches')
    ax.set_ylabel('System failure prob. bounds')

    # bounds vs sys fn runs
    ax = fig.add_subplot(312)
    ax.plot(monitor['sf_ns'], monitor['pf_low'], linestyle='--', color='blue')
    ax.plot(monitor['sf_ns'], monitor['pf_up'], linestyle='--', color='blue')
    ax.set_xlabel('No. of system function runs')
    ax.set_ylabel('System failure prob. bounds')

    # no. of rules vs sys fn runs
    ax = fig.add_subplot(313)
    ax.plot(monitor['r_ns'], monitor['pf_low'], linestyle='--', color='blue')
    ax.plot(monitor['r_ns'], monitor['pf_up'], linestyle='--', color='blue')
    ax.set_xlabel('No. of rules')
    ax.set_ylabel('System failure prob. bounds')

    #output_file = Path(sys.argv[1]).joinpath(output_file)
    fig.savefig(output_file, dpi=200)
    print(f'{output_file} created')



def proposed_branch_and_bound_using_probs(sys_fun, varis, probs, max_sf, output_path=Path(sys.argv[0]).parent, key=None, flag=False):

    assert isinstance(varis, dict), f'varis must be a dict: {type(varis)}'
    assert isinstance(probs, dict), f'probs must be a dict: {type(probs)}'

    # Initialisation
    no_sf = 0 # number of system function runs so far
    sys_res = pd.DataFrame(data={'sys_val': [], 'comp_st': [], 'comp_st_min': []}) # system function results 
    no_iter, pr_bf, pr_bs, pr_bu, no_bu =  0, 0, 0, 1, 1
    no_rf, no_rs, len_rf, len_rs = 0, 0, 0, 0
    max_bu = 0

    rules = {'s': [], 'f': []} # a list of known rules
    brs_new = []
    worst = {x: 0 for x in varis.keys()} # all components in the worst state
    best = {k: len(v.values) - 1 for k, v in varis.items()} # all components in the best state

    brs = init_branch(worst, best, rules)

    # For monitoring purpose (store for each iteration)
    monitor = {'pf_up': [], # upper bound on pf
               'pf_low': [], # lower bound on pf
               'br_ns': [], # number of branches
               'sf_ns': [], # number of system function runs
               'r_ns': [], # number of rules
              }

    stop_br = False
    while no_bu and no_sf < max_sf:

        no_iter += 1

        if stop_br and no_sf % 20 == 0:
            print(f'[System function runs {no_sf}]..')
            print(f'The # of found non-dominated rules (f, s): {no_rf + no_rs} ({no_rf}, {no_rs})')
            print(f'Probability of branchs (f, s, u): ({pr_bf:.2f}, {pr_bs:.2f}, {pr_bu:.2f})')
            #print('The # of branching: ', no_iter)
            print(f'The # of branches (f, s, u), min len of rf: {len(brs)} ({no_bf}, {no_bs}, {no_bu}), {min_len_rf:.2f}')  
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
                    no_sf += 1

                    rules = update_rule_set(rules, rule)
                    sys_res = pd.concat([sys_res, sys_res_], ignore_index=True)

                    ########## FOR MONITORING ##################
                    pr_bf = sum([b[4] for b in brs if b.up_state == 'f']) # prob. of failure branches
                    pr_bs = sum([b[4] for b in brs if b.down_state == 's']) # prob. of survival branches
                    no_bu = sum([(b.up_state == 'u') or (b.down_state == 'u') or (b.down_state != b.up_state) for b in brs]) # no. of unknown branches
                    pr_bu = sum([b[4] for b in brs if (b.up_state == 'u') or (b.down_state == 'u') or (b.down_state != b.up_state)])

                    try:
                        min_len_rf = min( [len(x) for x in rules['f']] )
                    except ValueError:
                        min_len_rf = 0

                    no_rf = len(rules['f']) # no. of failure rules
                    no_rs = len(rules['s']) # no. of survival rules

                    no_bf = sum([b.up_state == 'f' for b in brs])
                    no_bs = sum([b.down_state == 's' for b in brs])

                    monitor['r_ns'].append(len(rules['f']) + len(rules['s']))
                    monitor['pf_up'].append(1.0-pr_bs)
                    monitor['pf_low'].append(pr_bf)
                    monitor['br_ns'].append(len(brs))
                    monitor['sf_ns'].append(no_sf)
                    #########################################

                    brs = init_branch(worst, best, rules)
                    brs_new = []
                    no_iter = 0
                    # for loop exit
                    stop_br = True
                    break

                else:
                    xd, xd_st = get_decomp_comp_using_probs(br.down, br.up, c_rules, probs)

                    # for upper
                    up = br.up.copy()
                    up[xd] = xd_st - 1
                    up_state = get_state(up, rules)
                    p = approx_branch_prob(br.down, up, probs)
                    brs_new.append(branch.Branch(br.down, up, br.down_state, up_state, p))

                    # for lower
                    down = br.down.copy()
                    down[xd] = xd_st
                    down_state = get_state(down, rules)
                    p = approx_branch_prob(down, br.up, probs)
                    brs_new.append(branch.Branch(down, br.up, down_state, br.up_state, p))

        # for the for loop
        if stop_br == False:
            brs = brs_new
            brs_new = []

            #ok = any([(b.up_state == 'u') or (b.down_state == 'u') or (b.down_state != b.up_state) for b in brs])  # exit for loop
            #pr_bf = sum([b[4] for b in brs if b.up_state == 'f']) # prob. of failure branches
            #pr_bs = sum([b[4] for b in brs if b.down_state == 's']) # prob. of survival branches
            no_bu = sum([(b.up_state == 'u') or (b.down_state == 'u') or (b.down_state != b.up_state) for b in brs]) # no. of unknown branches
            #pr_bu = sum([b[4] for b in brs if (b.up_state == 'u') or (b.down_state == 'u') or (b.down_state != b.up_state)])
            if no_bu > max_bu:
                max_bu = no_bu

        #no_bf = sum([b.up_state == 'f' for b in brs])
        #no_bs = sum([b.down_state == 's' for b in brs])

        #no_rf = len(rules['f']) # no. of failure rules
        #no_rs = len(rules['s']) # no. of survival rules

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
        #print(f'# of unknown branches to go: {no_bu}, {max_bu}\n')
        if no_sf >= max_sf:
            print(f'*** Terminated due to the # of system function runs: {no_sf} >= {max_sf}')

    print(f'**Algorithm Terminated**')
    #print(f'[System function runs {no_sf}]..')
    #print(f'The # of found non-dominated rules (f, s): {no_rf + no_rs} ({no_rf}, {no_rs})')
    #print(f'Probability of branchs (f, s, u): ({pr_bf}, {pr_bs}, {pr_bu})')


    ########## FOR MONITORING ##################
    pr_bf = sum([b[4] for b in brs if b.up_state == 'f']) # prob. of failure branches
    pr_bs = sum([b[4] for b in brs if b.down_state == 's']) # prob. of survival branches
    no_bu = sum([(b.up_state == 'u') or (b.down_state == 'u') or (b.down_state != b.up_state) for b in brs]) # no. of unknown branches
    pr_bu = sum([b[4] for b in brs if (b.up_state == 'u') or (b.down_state == 'u') or (b.down_state != b.up_state)])

    no_bf = sum([b.up_state == 'f' for b in brs])
    no_bs = sum([b.down_state == 's' for b in brs])

    monitor['r_ns'].append(len(rules['f']) + len(rules['s']))
    monitor['pf_up'].append(float(1.0-pr_bs))
    monitor['pf_low'].append(float(pr_bf))
    monitor['br_ns'].append(len(brs))
    monitor['sf_ns'].append(no_sf)
    #########################################

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

def decomp_df(varis, rules, probs, max_nb): # depth-first decomposition of event space using given rules

    worst = {x: 0 for x in varis.keys()}
    best = {k: len(v.values) - 1 for k, v in varis.items()}

    brs = init_branch(worst, best, rules)
    brs_cr = [get_compat_rules(brs[0].down, brs[0].up, rules)]
    brs_new = []
    brs_new_cr = []

    stop = False
    while stop == False:
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
                up = br.up.copy()
                up[xd] = xd_st - 1
                up_state = get_state(up, rules)
                p = approx_branch_prob(br.down, up, probs)
                br_new = branch.Branch(br.down, up, br.down_state, up_state, p)
                brs_new.append(br_new)
                if (br_new.down_state == br_new.up_state) and (br_new.up_state != 'u'):
                    brs_new_cr.append({'s':[], 'f':[]})
                else:
                    cr_new = get_compat_rules(br_new.down, br_new.up, rules)
                    brs_new_cr.append(cr_new)

                # for lower
                down = br.down.copy()
                down[xd] = xd_st
                down_state = get_state(down, rules)
                p = approx_branch_prob(down, br.up, probs)
                br_new = branch.Branch(down, br.up, down_state, br.up_state, p)
                brs_new.append(br_new)
                if (br_new.down_state == br_new.up_state) and (br_new.down_state != 'u'):
                    brs_new_cr.append({'s':[], 'f':[]})
                else:
                    cr_new = get_compat_rules(br_new.down, br_new.up, rules)
                    brs_new_cr.append(cr_new)

                n_br = len(brs_new) + (len(brs)-(i+1)) # the current number of branches
                if n_br >= max_nb:
                    stop = True

                    try:
                        brs_new = brs_new + brs[(i+1):]
                        brs_new_cr = brs_new_cr + brs_cr[(i+1):]
                    except TypeError:
                        ddd = 1
                    break

        brs = copy.deepcopy( brs_new )
        brs_cr = copy.deepcopy( brs_new_cr )
        brs_new = []
        brs_new_cr = []

        if stop==False:
            ncr = [len(cr['f'])+len(cr['s']) for cr in brs_cr]
            if all(x==0 for x in ncr):
                stop = True

    return brs, brs_cr

def get_st_decomp( brs, surv_first=True, varis = None, probs=None ): # 'brs' is a list of branches obtained by depth-first decomposition

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
                b_new = branch.Branch( b.up, best, 'u', 's', p_new )
                brs_new.append(b_new)

            if b.down_state =='u':
                p_new = approx_branch_prob(worst, b.down, probs)
                b_new = branch.Branch( worst, b.down, 'f', 'u', p_new )
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


def run_brc(varis, probs, sys_fun, max_sf, max_nb, pf_bnd_wr = 0.0, surv_first=True, rules=None):

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

    monitor = {'pf_up': [], # upper bound on pf
               'pf_low': [], # lower bound on pf
               'br_s_ns': [], # number of branches-survival
               'br_f_ns': [], # number of branches=failure
               'br_u_ns': [], # number of branches-unknown
               'r_s_ns': [], # number of rules-survival
               'r_f_ns': [], # number of rules-failure
               'sf_ns': [], # number of system function runs
               'time': [] # time taken for each iteration (sec)
              }

    no_sf = 0
    pr_bf, pr_bs = 0.0, 0.0
    while no_sf < max_sf:
        start = time.time() # monitoring purpose

        brs, _ = decomp_df(varis, rules, probs, max_nb)
        x_star = get_st_decomp( brs, surv_first, varis, probs )

        if x_star == None:
            monitor['out_flag'] = 'complete'
            break
        elif len(brs) >= max_nb:
            monitor['out_flag'] = 'max_nb'
            break
        elif (1-pr_bf-pr_bs) < pr_bf*pf_bnd_wr:
            monitor['out_flag'] = 'pf_bnd'
            break 
        else:
            rule, sys_res_ = run_sys_fn(x_star, sys_fun, varis)

            rules = update_rule_set(rules, rule)
            sys_res = pd.concat([sys_res, sys_res_], ignore_index=True)
            no_sf += 1

            end = time.time() # monitoring purpose

            ######## monitoring ############
            pr_bf = sum([b[4] for b in brs if b.up_state == 'f']) # prob. of failure branches
            pr_bs = sum([b[4] for b in brs if b.down_state == 's']) # prob. of survival branches
        
            no_rf = len(rules['f'])
            no_rs = len(rules['s'])

            no_bf = sum([b.up_state == 'f' for b in brs])
            no_bs = sum([b.down_state == 's' for b in brs])
            no_bu = len(brs) - no_bf - no_bs

            monitor['r_f_ns'].append( no_rf )
            monitor['r_s_ns'].append( no_rs )
            monitor['pf_up'].append(1.0-pr_bs)
            monitor['pf_low'].append(pr_bf)
            monitor['br_s_ns'].append(no_bs)
            monitor['br_f_ns'].append(no_bf)
            monitor['br_u_ns'].append(no_bu)
            monitor['sf_ns'].append(no_sf)
            monitor['time'].append(end-start)

            try:
                min_len_rf = min( [len(x) for x in rules['f']] )
                avg_len_rf = sum( [len(x) for x in rules['f']] ) / no_rf
            except ValueError:
                min_len_rf = 0
                avg_len_rf = 0

            if no_sf % 20 == 0:
                print(f'[System function runs {no_sf}]..')
                print(f'The # of found non-dominated rules (f, s): {no_rf + no_rs} ({no_rf}, {no_rs})')
                print(f'Probability of branchs (f, s, u): ({pr_bf:.4e}, {pr_bs:.2e}, {1.0-pr_bs-pr_bf:.4e})')
                print(f'The # of branches (f, s, u), (min, avg) len of rf: {len(brs)} ({no_bf}, {no_bs}, {no_bu}), ({min_len_rf}, {avg_len_rf:.2f})')
            #################################

        if not (no_sf < max_sf):
            monitor['out_flag'] = 'max_sf'

    brs, _ = decomp_df(varis, rules, probs, max_nb)
    try:
        out_flag = monitor['out_flag']
        print(f'***Analysis completed with f_sys runs {no_sf}: out_flag = {out_flag}***')
        print(f'The # of found non-dominated rules (f, s): {no_rf + no_rs} ({no_rf}, {no_rs})')
        print(f'Probability of branchs (f, s, u): ({pr_bf:.4e}, {pr_bs:.2e}, {1.0-pr_bs-pr_bf:.4e})')
        print(f'The # of branches (f, s, u), (min, avg) len of rf: {len(brs)} ({no_bf}, {no_bs}, {no_bu}), ({min_len_rf}, {avg_len_rf:.2f})')
    except NameError: # analysis is terminated before the first system function run
        print(f'***Analysis terminated without any evaluation***')

    return brs, rules, sys_res, monitor

"""
# FIXME: NOT USED ANYMORE
def add_rule(rules, rules_st, new_rule, fail_or_surv):
    # Update a rules list by removing dominated rules and adding a new rule
    rmv_inds = []
    add_rule = True

    for i, rule in enumerate(rules):

        if all([k in rule for k in new_rule.keys()]):# does all keys in rule 1 exist for rule_i?

            if fail_or_surv == rules_st[i] == 's':

                if all([rule[k] >= v for k, v in new_rule.items()]): # this rule is dominated by the new rule
                    rmv_inds.append(i)

                elif all([k in new_rule for k in rule.keys()]) and all([new_rule[k] >= v for k,v in rule.items()]):
                    add_rule = False
                    break # the new rule is dominated by an existing one. no further investigation required (assuming that a given list is a set of non-dominated rules)

            elif fail_or_surv == rules_st[i] == 'f':

                if all([rule[k] <= v for k,v in new_rule.items()]): # this rule is dominated by the new rule
                    rmv_inds.append(i)

                elif all([k in new_rule for k in rule.keys()]) and all([new_rule[k] <= v for k,v in rule.items()]):
                    add_rule = False
                    break # the new rule is dominated by an existing one. no further investigation required (assuming that a given list is a set of non-dominated rules)

    #rules_new = copy.deepcopy(rules)
    #rules_st_new = copy.deepcopy(rules_st)

    for i in rmv_inds[::-1]:
        try:
            del rules[i]
            del rules_st[i]
        except:
            pass

    if add_rule == True:
        rules.append(new_rule)
        rules_st.append(fail_or_surv)

    return rules, rules_st
"""
'''
# FIXME: NOT USED ANYMORE
def get_comp_st_for_next_bnb(up, down, rules, rules_st):
    """
    up: dict
    down: dict
    rules: list
    rules_st: list
    """

    assert isinstance(up, dict), f'up must be a dict: {type(up)}'
    assert isinstance(down, dict), f'down must be a dict: {type(down)}'

    # get the next component and its state for branching

    # rules: a list of rules (in dictionary)
    # rules_st: a list of rules' state (the same length as rules)

    idx_up, _ = get_compat_rules_old(up, rules, rules_st)
    idx_down, _ = get_compat_rules_old(down, rules, rules_st)

    idx = set(idx_up + idx_down)
    c_rules = [rules[i] for i in idx]
    c_st = [rules_st[i] for i in idx]
    _len = [len(x) for x in c_rules]

    idx = sorted(range(len(_len)), key=lambda y: _len[y])
    c_rules = [c_rules[i] for i in idx]
    c_st = [c_st[i] for i in idx]

    comps_count = {}
    comp_bnb = None
    for r, r_st in zip(c_rules, c_st):

        comps = list(r.keys())

        count = [] # counts of components' appearance across rules
        for x in comps:

            if x not in comps_count:
                comps_count[x] = sum([x in y for y in c_rules])

            count.append(comps_count[x])

        # order components by their frequency in rules set
        idx = sorted(range(len(count)), key=lambda y: count[y])

        for j in idx[::-1]:
            comp = comps[j]
            comp_st = r[comp]

            if r_st == 's':
                if comp_st > down[comp]:
                    comp_bnb = comp
                    st_up = comp_st # this is always the upper branch's lower state
                    break

            else: # r_i_st == 'f'
                if comp_st < up[comp]:
                    comp_bnb = comp
                    st_up = comp_st + 1 # this is always the upper branch's lower state
                    break

        if comp_bnb:
            break

    # in case nothing has been selected from above
    if comp_bnb is None: # just randomly select from the components that have higher frequencies
        for r in rules:
            for k in r:
                if k not in comps_cnt:
                    comps_count[k] = sum([k in r for r in rules])

        for x in sorted(comps_count)[::-1]:
            if down[x] < up[x]:
                comp_bnb = x
                st_up = up[x]
                break

    return comp_bnb, st_up


# FIXME: NOT USED ANYMORE
def decomp_to_two_branches(br, comp, state):
    """
    br: an instance of branch
    comp: component name
    state: component state (integer)
    """
    assert isinstance(br, branch.Branch_old), f'br must be an instance of Branch_old: f{type(br)}'
    assert isinstance(comp, str), f'comp must be a string: f{type(comp)}'
    assert comp in br.names, f'comp must exist in br.names: {comp}'
    assert isinstance(state, int), f'state must be an integer: f{type(state)}'

    down = {x: y for x, y in zip(br.names, br.down)}
    up = {x: y for x, y in zip(br.names, br.up)}

    up[comp] = state - 1
    down[comp] = state

    brs = [branch.Branch_old(br.down, list(up.values()), names=br.names, is_complete=False),
           branch.Branch_old(list(down.values()), br.up, names=br.names, is_complete=False)]

    return brs


# FIXME: NOT USED ANYMORE
def init_branch_old(varis, rules, rules_st):
    """
    return initial branch based on given rules and states
    """
    down = {x: 0 for x in varis.keys()} # all components in the worst state
    up = {k: len(v.values) - 1 for k, v in varis.items()} # all components in the best state

    brs = [branch.Branch_old(list(down.values()), list(up.values()), is_complete=False, names=list(varis.keys()))]

    _, brs[0].up_state = get_compat_rules_old(up, rules, rules_st)
    _, brs[0].down_state = get_compat_rules_old(down, rules, rules_st)

    return brs


# FIXME: NOT USED ANYMORE
def core(brs, rules, rules_st, cst, stop_br):
    """
    brs: list of Branch instance
    rules_st:
    cst: changed or passed *FIXEME: cst: does not seem to be necessary
    stop_br: changed or passed

    """
    brs_new = []
    #print(f'brs entering core: {brs}')
    for i, br in enumerate(brs):

        up = {x: y for x, y in zip(br.names, br.up)}
        idx, _ = get_compat_rules_old(up, rules, rules_st)
        #print(f'up, idx: {up}, {idx}')
        if br.up_state == 'u' and len(idx) == 0:
            cst = br.up # perform analysis on this state
            stop_br = True
            break

        down = {x: y for x, y in zip(br.names, br.down)}
        idx, _ = get_compat_rules_old(down, rules, rules_st)

        #print(f'down, idx: {down}, {idx}')
        if br.down_state == 'u' and len(idx) == 0:
            cst = br.down # perform analysis on this state
            stop_br = True
            break

        if br.up_state == 's' and br.down_state == 'f':

            comp, st_up = get_comp_st_for_next_bnb(up, down, rules, rules_st)
            brs2 = decomp_to_two_branches(br, comp, st_up)

            for b in brs2:

                up = {x: y for x, y in zip(br.names, b.up)}
                idx, cst_up = get_compat_rules_old(up, rules, rules_st)

                if cst_up == 'u' and len(idx) == 0:
                    cst = b.up # perform analysis on this state
                    stop_br = True
                    break

                else:
                    b.up_state = cst_up

                down = {x: y for x, y in zip(br.names, b.down)}
                idx, cst_down = get_compat_rules_old(down, rules, rules_st)

                if cst_down == 'u' and len(idx) == 0:
                    cst = b.down # perform analysis on this state
                    stop_br = True
                    break

                else:
                    b.down_state = cst_down
                    brs_new.append(b)
                    if cst_down == cst_up:
                        b.is_complete = True

            if stop_br == True:
                break

        elif br.up_state != 'u' and br.up_state == br.down_state:
            brs_new.append(br)

        elif br.is_complete == True:
            brs_new.append(br)

        else:
            b.down_state = cst_down
            if cst_down == cst_up:
                b.is_complete = True

    if stop_br == False:
        brs = copy.deepcopy(brs_new)

    return brs, cst, stop_br


# FIXME: NOT USED ANYMORE
def get_brs_from_rules( rules, rules_st, varis, max_br ):

    brs = init_branch_old(varis, rules, rules_st)
    stop_br = False
    while any([not b.is_complete for b in brs]) and stop_br==False:
        brs_new = []
        for i, br in enumerate(brs):

            up = {x: y for x, y in zip(br.names, br.up)}
            down = {x: y for x, y in zip(br.names, br.down)}

            if br.is_complete == False:
                comp, st_up = get_comp_st_for_next_bnb(up, down, rules, rules_st)
                brs2 = decomp_to_two_branches(br, comp, st_up)

                for b in brs2:

                    up = {x: y for x, y in zip(br.names, b.up)}
                    _, cst_up = get_compat_rules_old(up, rules, rules_st)

                    down = {x: y for x, y in zip(br.names, b.down)}
                    _, cst_down = get_compat_rules_old(down, rules, rules_st)

                    b.up_state = cst_up
                    b.down_state = cst_down

                    if cst_up == cst_down:
                        b.is_complete = True

                    brs_new.append(b)

            else:
                brs_new.append(br)


            if len(brs_new) + (len(brs)-(i+1)) >= max_br:
                brs_rem = copy.deepcopy( brs[(i+1):] )
                brs_new += brs_rem
                stop_br = True
                break

        brs = copy.deepcopy(brs_new)

    return brs_new


# FIXME: NOT USED ANYMORE
def get_sys_rules(cst, sys_fun, rules, rules_st, varis):
    """
    cst:

    """
    cst = {x: y for x, y in zip(varis.keys(), cst)}
    #no_sf += 1
    sys_val, sys_st, min_comps_st = sys_fun(cst)
    sys_res = pd.DataFrame({'sys_val': [sys_val], 'comps_st': [cst], 'comps_st_min': [min_comps_st]})

    if min_comps_st:
        rule = min_comps_st
    else:
        if sys_st == 'surv':
            rule = {k: v for k, v in cst.items() if v} # the rule is the same as up_dict_i but includes only components whose state is greater than the worst one (i.e. 0)
        else:
            rule = {k: v for k, v in cst.items() if v < len(varis[k].B[0]) - 1} # the rule is the same as up_dict_i but includes only components whose state is less than the best one

    rules, rules_st = add_rule(rules, rules_st, rule, sys_st)

    return sys_res, rules, rules_st


# FIXME: NOT USED ANYMORE
def do_gen_bnb(sys_fun, varis, max_br, output_path=Path(sys.argv[0]).parent, key=None, flag=False):
    ### MAIN FUNCTION ####
    """
    Input:
    sys_fun: system function that takes in a component vector state and returns system function value, system state, and (optional: if unavailable, "None" can be returned) minimially component state to fulfill the obtained system function value.
    varis: A dictionary of "Variable"s. All component events need to be defined with its "B matrix".
    max_br: max. number of branches (the algorithm stops when this number is met)
    """

    assert isinstance(varis, dict), f'varis must be a dict: {type(varis)}'

    # Initialisation
    #no_sf = 0 # number of system function runs so far
    sys_res = pd.DataFrame(data={'sys_val': [], 'comps_st': [], 'comps_st_min': []}) # system function results 
    rules = [] # a list of known rules
    rules_st = [] # a list of known rules' states
    no_iter =  0
    ok = True
    brs = []
    cst = []

    while ok and len(brs) < max_br:

        no_iter += 1
        ###############
        print(f'[Iteration {no_iter}]..')
        print(f'The # of found non-dominated rules: {len(rules)}')
        #print('System function runs: ', no_sf) # Redundant with iteration number
        print(f'The # of branches: {len(brs)}')
        print('---')
        ###############

        ## Start from the total event ##
        brs = init_branch_old(varis, rules, rules_st)
        #print(f'brat at {no_iter}: {brs}')
        stop_br = False

        while ok:

            brs, cst, stop_br = core(brs, rules, rules_st, cst, stop_br)

            if stop_br:
                break
            else:
                ok = any([not b.is_complete for b in brs])

        # update rules, rules_st
        sys_res_, rules, rules_st = get_sys_rules(cst, sys_fun, rules, rules_st, varis)
        #print(f'sys_res_: {sys_res_.values[0]}')
        #print(f'rules: {rules}')
        #print(f'rules_st: {rules_st}')
        #print(f'brs: {brs}')
        sys_res = pd.concat([sys_res, sys_res_], ignore_index=True)


    brs = get_brs_from_rules( rules, rules_st, varis, max_br )

    ###############
    print('[Algorithm completed.]')
    print('The # of found non-dominated rules: ', len(rules))
    print('System function runs: ', no_iter)
    print('The total # of branches: ', len(brs))
    print('The # of incomplete branches: ', sum([not b.is_complete for b in brs]))
    ###############

    if flag:
        output_file = output_path.joinpath(f'brs_{key}.pk')
        with open(output_file, 'wb') as fout:
            pickle.dump(brs, fout)
        print(f'{output_file} is saved')

    return no_iter, rules, rules_st, brs, sys_res
'''

'''
# FIXME: NOT USED ANYMORE
def get_c_from_br_old(br, varis, st_br_to_cs):
    """
    return updated varis and state
    br: a single branch
    varis: a dictionary of variables
    st_br_to_cs: a dictionary that maps state in br to state in C matrix of a system event
    """

    cst = np.zeros(len(br.names) + 1, dtype=int) # (system, compponents)

    if br.is_complete == True:
        cst[0] = st_br_to_cs[br.down_state]
    else:
        cst[0] = st_br_to_cs['u']

    for i, x in enumerate(br.names):
        down = br.down[i]
        up = br.up[i]

        if up > down:
            varis[x], st = variable.get_composite_state(varis[x], list(range(down, up + 1)))
        else:
            st = up

        cst[i + 1] = st

    return varis, cst

# FIXME: NOT USED ANYMORE
def get_csys_from_brs(brs, varis, st_br_to_cs):

    c_sys = np.empty(shape=(0, len(brs[0].names) + 1), dtype=int)

    for br in brs:
        varis, c = get_c_from_br_old(br, varis, st_br_to_cs)
        c_sys = np.vstack([c_sys, c])

    return c_sys, varis

# FIXME: NOT USED ANYMORE
def get_compat_rules_old(cst, rules, rules_st):
    #cst: component vector state in dictionary
    #rules: list of rules
    #rules_st: list of rules' state (either 'surv' or 'fail') -- must have the same length as rules

    assert isinstance(cst, dict), f'cst should be a dict: {type(cst)}'
    assert isinstance(rules, list), f'rules should be a list: {type(rules)}'
    assert isinstance(rules_st, list), f'rules_st should be a list: {type(rultes_st)}'

    cr_inds = [] # compatible rules--stored by their indices
    no_surv = 0 # number of compatible survival rules
    no_fail = 0 # number of compatible failure rules
    for ind, r in enumerate(rules):
        if rules_st[ind] == 's':
            if all([cst[k] >= r[k] for k in r]): # the survival rule is satisfied
                cr_inds.append(ind)
                no_surv += 1

        else: # rules_st[ind] == 'fail'
            if all([cst[k] <= r[k] for k in r]): # the failure rule is compatible
                cr_inds.append(ind)
                no_fail += 1

    if no_surv == no_fail == 0:
        cst_state = 'u' # unknown
    else:
        if no_surv > no_fail:
            cst_state = 's'
        else:
            cst_state = 'f'

        if no_surv > 0 and no_fail > 0:
            warnings.warn("[get_compat_rules] Conflicting rules found. The given system is not coherent." )

    return cr_inds, cst_state
'''

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
