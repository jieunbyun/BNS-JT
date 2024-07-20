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


def run(varis, probs, sys_fun, max_sf, max_nb, pf_bnd_wr=0.0, surv_first=True, rules=None):

    """
    INPUTS:
    varis: a dictionary of variable#s
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

    if not rules:
        rules = {'s': [], 'f': []}

    sys_res = pd.DataFrame(data={'sys_val': [],
                                 'comp_st': [],
                                 'comp_st_min': []}) # system function results

    monitor, ctrl = init_monitor()

    while ctrl['no_sf'] < max_sf:

        start = time.time() # monitoring purpose

        brs, _ = decomp_depth_first(varis, rules, probs, max_nb)  # S2
        x_star = get_comp_st(brs, surv_first, varis, probs)  # S4-1

        if x_star == None:
            monitor['out_flag'] = 'complete'
            break

        elif len(brs) >= max_nb:
            monitor['out_flag'] = 'max_nb'
            break

        elif ctrl['pr_bu'] < ctrl['pf_low'] * pf_bnd_wr:
            monitor['out_flag'] = 'pf_bnd'
            break

        else:
            rule, sys_res_ = run_sys_fn(x_star, sys_fun, varis) # S4-2, S5

            rules = update_rule_set(rules, rule) # S6
            sys_res = pd.concat([sys_res, sys_res_], ignore_index=True)
            ctrl['no_sf'] += 1

            monitor['no_sf'].append(ctrl['no_sf'])
            monitor, ctrl = update_monitor(monitor, brs, rules, start) # S7

            if ctrl['no_sf'] % 20 == 0:
                print(f"[System function runs {ctrl['no_sf']}]..")
                display_msg(monitor)

        if ctrl['no_sf'] == max_sf:
            monitor['out_flag'] = 'max_sf'

    # NOTSURE???
    #brs, _ = decomp_depth_first(varis, rules, probs, max_nb)

    try:
        monitor, ctrl = update_monitor(monitor, brs, rules, start)

        print(f"*** Analysis completed with f_sys runs {ctrl['no_sf']}: out_flag = {monitor['out_flag']} ***")
        display_msg(monitor)

    except NameError: # analysis is terminated before the first system function run
        print(f'***Analysis terminated without any evaluation***')

    return brs, rules, sys_res, monitor

'''
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
'''

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


def get_compat_rules(br, rules):
    """
    lower: lower bound on component vector state in dictionary
           e.g., {'x1': 0, 'x2': 0 ... }
    upper: upper bound on component vector state in dictionary
           e.g., {'x1': 2, 'x2': 2 ... }
    rules: dict of rules
           e.g., {'s': [{'x1': 2, 'x2': 2}],
                  'f': [{'x1': 2, 'x2': 0}]}
    """
    #assert isinstance(lower, dict), f'lower should be a dict: {type(lower)}'
    #assert isinstance(upper, dict), f'upper should be a dict: {type(upper)}'
    assert isinstance(rules, dict), f'rules should be a dict: {type(rules)}'

    compat_rules = {'s': [], 'f': []}
    for rule in rules['s']:
        if all([br.up[k] >= v for k, v in rule.items()]):
            c_rule = {k: v for k, v in rule.items() if v > br.down[k]}
            if c_rule:
                compat_rules['s'].append(c_rule)

    for rule in rules['f']:
        if all([br.down[k] <= v for k, v in rule.items()]):
            c_rule = {k: v for k, v in rule.items() if v < br.up[k]}
            if c_rule:
                compat_rules['f'].append(c_rule)

    return compat_rules



def approx_joint_prob_compat_rule(br, rule, rule_st, probs):

    #assert isinstance(lower, dict), f'lower should be a dict: {type(lower)}'
    #assert isinstance(upper, dict), f'upper should be a dict: {type(upper)}'
    assert isinstance(rule, dict), f'rules should be a dict: {type(rules)}'

    p = 1.0
    if rule_st == 's':
        for x, v in rule.items():
            p *= sum([probs[x][i] for i in range(v, br.up[x] + 1)])

    elif rule_st == 'f':
        for x, v in rule.items():
            p *= sum([probs[x][i] for i in range(br.down[x], v + 1)])

    return p


def approx_branch_prob(down, up, probs):
    """

    """
    p = 1.0
    for k, v in down.items():
        p *= sum([probs[k][x] for x in range(v, up[k] + 1)])

    return p


def get_decomp_comp_using_probs(br, rules, probs):
    """
    lower: lower bound on component vector state in dictionary
           e.g., {'x1': 0, 'x2': 0 ... }
    upper: upper bound on component vector state in dictionary
           e.g., {'x1': 2, 'x2': 2 ... }
    rules: dict of rules
           e.g., {'s': [{'x1': 2, 'x2': 2}], 'f': [{'x3': 0}]}
    probs: dict
    """
    assert isinstance(br, branch.Branch), f'br should be an instance of Branch: {type(br)}'
    #assert isinstance(br.lower, dict), f'lower should be a dict: {type(lower)}'
    #assert isinstance(br.upper, dict), f'upper should be a dict: {type(upper)}'
    assert isinstance(rules, dict), f'rules should be a dict: {type(rules)}'

    # get an order of component by their frequency in rules
    rules_st = [(k, x) for k, rule in rules.items() for x in rule]
    comps = Counter(chain.from_iterable([x[1] for x in rules_st]))
    comps = [x[0] for x in comps.most_common()]

    # get an order R by P  (higher to lower)
    if len(rules_st) > 1:
        rules_st = sorted(rules_st, key=lambda x: approx_joint_prob_compat_rule(br, x[1], x[0], probs), reverse=True)

    for st, rule in rules_st:
        for c in comps:
            state = None
            if c in rule and st == 's':
                state = rule[c]

            elif c in rule and st == 'f':
                state = rule[c] + 1

            try:
                if br.down[c] < state and state <= br.up[c]:
                    return c, state

            except TypeError:
                pass


def get_decomp_comp_using_probs_old(lower, upper, rules, probs):
    """
    lower: lower bound on component vector state in dictionary
           e.g., {'x1': 0, 'x2': 0 ... }
    upper: upper bound on component vector state in dictionary
           e.g., {'x1': 2, 'x2': 2 ... }
    rules: dict of rules
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

    # S4-2: get system state given comp
    sys_val, sys_st, comp_st_min = sys_fun(comp)

    sys_res = pd.DataFrame({'sys_val': [sys_val], 'comp_st': [comp], 'comp_st_min': [comp_st_min]})

    if comp_st_min:
        rule = comp_st_min, sys_st

    else:
        if sys_st == 's':
            rule = {k: v for k, v in comp.items() if v}, sys_st # the rule is the same as up_dict but includes only components whose state is greater than the worst one (i.e. 0)
        else:
            rule = {k: v for k, v in comp.items() if v < len(varis[k].values) - 1}, sys_st # the rule is the same as up_dict but includes only components whose state is less than the best one

    return rule, sys_res


def get_new_branch(br, rules, probs, xd, xd_st, up_flag=True):
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

    return branch.Branch(down, up, down_st, up_st, p)
    """
    if down_st == up_st:
        if (~up_flag and down_st != 'u') or (up_flag and up_st != 'u'):
           cr_new = {'s':[], 'f':[]}
        else:
           cr_new = get_compat_rules(down, up, rules)
    else:
        cr_new = get_compat_rules(down, up, rules)
    return br_new
    """

def init_branch(varis, rules):
    """
    initialise a branch set (x_min, x_max, s(x_min), s(x_max), 1)
    """

    down = {x: 0 for x in varis.keys()}
    up = {k: len(v.values) - 1 for k, v in varis.items()}

    down_state = get_state(down, rules)
    up_state = get_state(up, rules)

    return [branch.Branch(down, up, down_state, up_state, 1.0)]


def decomp_depth_first(varis, rules, probs, max_nb):
    """
    depth-first decomposition of event space using given rules
    """

    brs = init_branch(varis, rules)  # D1
    crules = [get_compat_rules(brs[0], rules)]

    go = True
    while go:

        # D2: sort branches from higher to lower p 
        sorted_brs = sorted(zip(brs, crules), key=lambda x: x[0].p, reverse=True)
        brs, crules = [list(x) for x in zip(*sorted_brs)]

        brs_new = []
        crules_new = []

        for i, (br, cr) in enumerate(sorted_brs, 1):

            # specified branch or no compatible rule exists
            if ((br.down_state == br.up_state) and (br.down_state != 'u')) or (len(cr['f']) + len(cr['s']) == 0):
                brs_new.append(br)
                crules_new.append({'s':[], 'f':[]})

            else:
                # D6??
                xd, xd_st = get_decomp_comp_using_probs(br, cr, probs)

                # D3: evaluate sl and su
                for up_flag in [True, False]:
                    br_new = get_new_branch(br, rules, probs, xd, xd_st, up_flag)
                    crule_new = get_compat_rules(br_new, rules)
                    brs_new.append(br_new)
                    crules_new.append(crule_new)

                #n_br = len(brs_new) + len(brs) - i # the current number of branches

                if len(brs_new) + len(brs) > max_nb:

                    go = False

                    try:
                        brs_new += brs[i:]
                        crules_new += crules[i:]
                    except TypeError:
                        pass
                    else:
                        break

        brs = copy.deepcopy(brs_new)
        crules = copy.deepcopy(crules_new)

        if go and sum([len(r['f']) + len(r['s']) for r in crules]) == 0:
           go = False

    return brs, crules


def get_comp_st(brs, surv_first=True, varis=None, probs=None):
    """
    get a component vector state from branches(brs)
    'brs' is a list of branches obtained by depth-first decomposition
    """

    if surv_first:

        brs = sorted(brs, key=lambda x: x.p, reverse=True)

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

        x_star = None
        if brs_new:
            brs_new = sorted(brs_new, key=lambda x: x.p, reverse=True)
            if brs_new[0].up_state == 'u':
                x_star = brs_new[0].up
            elif brs_new[0].down_state == 'u':
                x_star = brs_new[0].down

    return x_star

