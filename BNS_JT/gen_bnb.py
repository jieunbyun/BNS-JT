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

from BNS_JT import variable, branch, brc


def init_branch(down, up, rules):

    down_state = brc.get_state(down, rules)
    up_state = brc.get_state(up, rules)

    return [branch.Branch(down, up, down_state, up_state, 1.0)]


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
    monitor, ctrl = brc.init_monitor()

    stop_br = False
    while ctrl['no_bu'] and (ctrl['no_sf'] < max_sf):

        start = time.time() # monitoring purpose

        #no_iter += 1

        if stop_br and ctrl['no_sf'] % 20 == 0:
            print(f"[System function runs {ctrl['no_sf']}]..")
            brc.display_msg(monitor)

        stop_br = False
        brs = sorted(brs, key=lambda x: x.p, reverse=True)

        for br in brs:

            if (br.down_state == br.up_state) and (br.down_state != 'u'):
                brs_new.append(br)

            else:
                c_rules = brc.get_compat_rules(br.down, br.up, rules)

                if (not c_rules['s']) or (not c_rules['f']):
                    if not c_rules['s']:
                        x_star = br.up
                    else:
                        x_star = br.down

                    # run system function
                    rule, sys_res_ = brc.run_sys_fn(x_star, sys_fun, varis)
                    monitor['no_sf'].append(ctrl['no_sf'] + 1)

                    rules = brc.update_rule_set(rules, rule)
                    sys_res = pd.concat([sys_res, sys_res_], ignore_index=True)

                    ########## FOR MONITORING ##################
                    monitor, ctrl = brc.update_monitor(monitor, brs, rules, start)

                    brs = init_branch(worst, best, rules)
                    brs_new = []
                    #no_iter = 0
                    # for loop exit
                    stop_br = True
                    break

                else:
                    xd, xd_st = brc.get_decomp_comp_using_probs(br.down, br.up, c_rules, probs)

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
            monitor['no_sf'].append(ctrl['no_sf'])
            monitor, ctrl = brc.update_monitor(monitor, brs, rules, start)

        if ctrl['no_sf'] >= max_sf:
            print(f'*** Terminated due to the # of system function runs: {no_sf} >= {max_sf}')

    ########## FOR MONITORING ##################
    monitor['no_sf'].append(ctrl['no_sf'])
    monitor, ctrl = brc.update_monitor(monitor, brs, rules, start)
    #########################################

    print(f'**Algorithm Terminated**')
    print(f"[System function runs {ctrl['no_sf']}]..")
    brc.display_msg(monitor)

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
                    xd, xd_st = brc.get_decomp_comp(br.down, br.up, c_rules)

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
    assert isinstance(rules, dict), f'rules should be a list: {type(rules)}'

    # get and order of x by their frequency in rules
    _rules = [x for rule in rules.values() for x in rule]
    comp = Counter(chain.from_iterable(_rules))
    comp = [x[0] for x in comp.most_common()]

    # get an order of R by cardinality
    a_rules = sorted(_rules, key=len)

    for rule in a_rules:
        for c in comp:
            if c in rule:
                if (rule in rules['s']) and (lower[c] < rule[c]) and (rule[c] <= upper[c]):
                    xd = c, rule[c]
                    break
                if (rule in rules['f']) and (lower[c] <= rule[c]) and (rule[c] < upper[c]):
                    xd = c, rule[c]+ 1
                    break
        else:
            continue
        break

    return xd


def get_br_new(br, rules, probs, xd, xd_st, up_flag=True):
    """

    """
    if up_flag:
        up = br.up.copy()
        up[xd] = xd_st - 1
        up_st = brc.get_state(up, rules)

        down = br.down
        down_st = br.down_state

    else:
        up = br.up
        up_st = br.up_state

        down = br.down.copy()
        down[xd] = xd_st
        down_st = brc.get_state(down, rules)

    p = brc.approx_branch_prob(down, up, probs)
    br_new = branch.Branch(down, up, down_st, up_st, p)

    if down_st == up_st:
        if (~up_flag and down_st != 'u') or (up_flag and up_st != 'u'):
           cr_new = {'s':[], 'f':[]}
        else:
           cr_new = brc.get_compat_rules(down, up, rules)
    else:
        cr_new = brc.get_compat_rules(down, up, rules)

    return br_new, cr_new


def get_init_brs_and_cr(varis, rules):

    worst = {x: 0 for x in varis.keys()}
    best = {k: len(v.values) - 1 for k, v in varis.items()}

    brs = init_branch(worst, best, rules)
    brs_cr = [get_compat_rules(brs[0].down, brs[0].up, rules)]

    return brs, brs_cr


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
