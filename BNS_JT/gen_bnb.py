import pandas as pd
import copy
from pathlib import Path
from collections import Counter
from itertools import chain
import warnings
import sys
import pickle
import numpy as np


from BNS_JT import variable, branch


def init_branch(varis, rules):
    """
    return initial branch based on given rules and states
    """
    down = {x: 0 for x in varis.keys()} # all components in the worst state
    up = {k: v.B.shape[1] - 1 for k, v in varis.items()} # all components in the best state

    down_state = get_state(down, rules)
    up_state = get_state(up, rules)

    return [branch.Branch(down, up, down_state, up_state)]


def proposed_branch_and_bound(sys_fun, varis, max_br, output_path=Path(sys.argv[0]).parent, key=None, flag=False):

    assert isinstance(varis, dict), f'varis must be a dict: {type(varis)}'

    # Initialisation
    no_sf = 0 # number of system function runs so far
    sys_res = pd.DataFrame(data={'sys_val': [], 'comp_st': [], 'comp_st_min': []}) # system function results 
    no_iter =  0
    ok = True

    rules = [] # a list of known rules
    brs = init_branch(varis, rules)
    brs_new = []

    while ok and len(brs) < max_br:

        no_iter += 1
        ###############
        print(f'[Iteration {no_iter}]..')
        print(f'The # of found non-dominated rules: {len(rules)}')
        print('System function runs: ', no_sf) 
        print(f'The # of branches: {len(brs)}')
        print('---')
        ###############
        stop_br = False

        for br in brs:

            if (br.down_state == br.up_state) and (br.down_state != 'u'):
                brs_new.append(br)

            else:
                c_rules = get_compat_rules(br.down, br.up, rules)
                c_states = [x[1] for x in c_rules]

                if ('s' not in c_states) or ('f' not in c_states):
                    if ('s' not in c_states):
                        x_star = br.up
                    else:
                        x_star = br.down

                    # run system function
                    sys_res_, rule = run_sys_fn(x_star, sys_fun, rules, varis)
                    no_sf += 1

                    rules = update_rule_set(rules, rule)
                    sys_res = pd.concat([sys_res, sys_res_], ignore_index=True)
                    brs = init_branch(varis, rules)
                    brs_new = []
                    # for loop exit
                    stop_br = True
                    break

                else:
                    xd = get_decomp_comp(br.down, br.up, c_rules)
                    # for upper
                    up = br.up.copy()
                    up[xd[0]] = xd[1] - 1
                    up_state = get_state(up, rules)
                    br_up = branch.Branch(br.down, up, br.down_state, up_state)
                    brs_new.append(br_up)

                    # for lower
                    down = br.down.copy()
                    down[xd[0]] = xd[1]
                    down_state = get_state(down, rules)
                    br_down = branch.Branch(down, br.up, down_state, br.up_state)
                    brs_new.append(br_down)

        # for the for loop
        if stop_br == False:
            brs = brs_new
            brs_new = []

        ok = any([(b.up_state == 'u') or (b.down_state == 'u') or (b.down_state != b.up_state) for b in brs])  # exit for loop

    return brs, rules, sys_res


def get_cmat_from_br(br, varis, st_br_to_cs):
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
            varis[x], st = variable.get_composite_state(varis[x], list(range(down, up + 1)))
        else:
            st = up

        cst[i + 1] = st

    return varis, cst


def get_state(comp, rules):
    """
    comp: component state vector in dictionary
           e.g., {'x1': 0, 'x2': 0 ... }
    rules: list of rules
           e.g., {({'x1': 2, 'x2': 2}, 's')}
    """
    assert isinstance(comp, dict), f'comp should be a dict: {type(comp)}'
    assert isinstance(rules, list), f'rules should be a list: {type(rules)}'

    no_s = 0
    no_f = 0

    for rule in rules:
        # the survival rule is satisfied
        if rule[1] == 's' and all([comp[k] >= v for k, v in rule[0].items()]):
            no_s += 1

        # the failure rule is compatible
        elif rule[1] == 'f' and all([comp[k] <= v for k, v in rule[0].items()]):
            no_f += 1

    # no compatible rules. the state remains unknown
    if no_s == no_f == 0:
        state = 'u'
    elif no_s >= no_f:
        state = 's'
    else:
        state = 'f'

    if no_s > 0 and no_f > 0:
        warnings.warn("[get_state] Conflicting rules found. The given system is not coherent.")

    return state


def get_compat_rules(lower, upper, rules):
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

    x_new = new_rule[0].keys()

    if new_rule[1] == 's':

        for rule in [rule for rule in rules if rule[1] == 's']:
            x_rule = rule[0].keys()

            if set(x_new).issubset(x_rule) and all([rule[0][k] >= v for k, v in new_rule[0].items()]):
                rules.remove(rule)

            elif set(x_rule).issubset(x_new) and all([new_rule[0][k] >= v for k, v in rule[0].items()]):
                add_rule = False
                break

    elif new_rule[1] == 'f':

        for rule in [rule for rule in rules if rule[1] == 'f']:
            x_rule = rule[0].keys()

            if set(x_new).issubset(x_rule) and all([rule[0][k] <= v for k, v in new_rule[0].items()]):
                rules.remove(rule)

            elif set(x_rule).issubset(x_new) and all([new_rule[0][k] <= v for k, v in rule[0].items()]):
                add_rule = False
                break

    if add_rule:
        rules.append(new_rule)

    return rules


def run_sys_fn(comp, sys_fun, rules, varis):
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
            rule = {k: v for k, v in comp.items() if v < len(varis[k].B[0]) - 1}, sys_st # the rule is the same as up_dict_i but includes only components whose state is less than the best one

    return sys_res, rule



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


def init_branch_old(varis, rules, rules_st):
    """
    return initial branch based on given rules and states
    """
    down = {x: 0 for x in varis.keys()} # all components in the worst state
    up = {k: v.B.shape[1] - 1 for k, v in varis.items()} # all components in the best state

    brs = [branch.Branch_old(list(down.values()), list(up.values()), is_complete=False, names=list(varis.keys()))]

    _, brs[0].up_state = get_compat_rules_old(up, rules, rules_st)
    _, brs[0].down_state = get_compat_rules_old(down, rules, rules_st)

    return brs


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


def get_c_from_br(br, varis, st_br_to_cs):
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


def get_csys_from_brs(brs, varis, st_br_to_cs):
    """


    """
    c_sys = np.empty(shape=(0, len(brs[0].names) + 1), dtype=int)

    for br in brs:
        varis, c = get_c_from_br(br, varis, st_br_to_cs)
        c_sys = np.vstack([c_sys, c])

    return c_sys, varis


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

