import pandas as pd
import copy
from BNS_JT import variable, branch
import warnings
import numpy as np


def get_compat_rules(cst, rules, rules_st):
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
        if rules_st[ind] == 'surv':
            if all([cst[k] >= r[k] for k in r]): # the survival rule is satisfied
                cr_inds.append(ind)
                no_surv += 1

        else: # rules_st[ind] == 'fail'
            if all([cst[k] <= r[k] for k in r]): # the failure rule is compatible
                cr_inds.append(ind)
                no_fail += 1

    if no_surv == no_fail == 0:
        cst_state = 'unk' # unknown
    else:
        if no_surv > no_fail:
            cst_state = 'surv'
        else:
            cst_state = 'fail'

        if no_surv > 0 and no_fail > 0:
            warnings.warn("[get_compat_rules] Conflicting rules found. The given system is not coherent." )

    return cr_inds, cst_state


def add_rule(rules, rules_st, new_rule, fail_or_surv):
    """
    rules: list of rules
    rules_st: list of rules' state
    new_rule: dict
    fail_or_surv:
    """
    assert isinstance(new_rule, dict), f'rule should be a dict: {type(new_rule)}'

    # Update a rules list by removing dominated rules and adding a new rule
    rmv_inds = []
    add_rule = True

    for i, rule in enumerate(rules):

        if all([k in rule for k in new_rule.keys()]):# does all keys in rule 1 exist for rule_i?

            if fail_or_surv == rules_st[i] == 'surv':

                if all([rule[k] >= v for k, v in new_rule.items()]): # this rule is dominated by the new rule
                    rmv_inds.append(i)

                elif all([k in new_rule for k in rule.keys()]) and all([new_rule[k] >= v for k,v in rule.items()]):
                    add_rule = False
                    break # the new rule is dominated by an existing one. no further investigation required (assuming that a given list is a set of non-dominated rules)

            elif fail_or_surv == rules_st[i] == 'fail':

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

    idx_up, _ = get_compat_rules(up, rules, rules_st)
    idx_down, _ = get_compat_rules(down, rules, rules_st)

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

        #c_i_sort_ind = [j[0] for j in sorted(enumerate(comps_i_cnt), key=lambda x:x[1])] # order components by their frequency in rules set
        idx = sorted(range(len(count)), key=lambda y: count[y])# order components by their frequency in rules set
        for j in idx[::-1]:
            comp = comps[j]
            comp_st = r[comp]

            if r_st == 'surv':
                if comp_st > down[comp]:
                    comp_bnb = comp
                    st_up = comp_st # this is always the upper branch's lower state
                    break

            else: # r_i_st == 'fail'
                if comp_st < up[comp]:
                    comp_bnb = comp
                    st_up = comp_st + 1 # this is always the upper branch's lower state
                    break

        if comp_bnb:
            break

    # in case nothing has been selected from above
    comps_cnt = {}
    if comp_bnb is None: # just randomly select from the components that have higher frequencies
        for r in rules:
            for k in r:
                if k not in comps_cnt:
                    # FIXME: x? --> FIXED: to k
                    comps_cnt[k] = sum([k in r for r in rules])

        cnt_sort_xs = sorted(comps_cnt)
        for x in cnt_sort_xs[::-1]:
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
    assert isinstance(br, branch.Branch), f'br must be an instance of Branch: f{type(br)}'
    assert isinstance(comp, str), f'comp must be a string: f{type(comp)}'
    assert comp in br.names, f'comp must exist in br.names: {comp}'
    assert isinstance(state, int), f'state must be an integer: f{type(state)}'

    down = {x: y for x, y in zip(br.names, br.down)}
    up = {x: y for x, y in zip(br.names, br.up)}

    up[comp] = state - 1
    down[comp] = state

    brs = [branch.Branch(br.down, list(up.values()), names=br.names, is_complete=False),
           branch.Branch(list(down.values()), br.up, names=br.names, is_complete=False)]

    return brs


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


def core(brs, rules, rules_st, cst, stop_br):
    """
    brs: list of Branch instance
    rules_st:
    cst: changed or passed
    stop_br: changed or passed

    """
    brs_new = []

    for i, br in enumerate(brs):

        up = {x: y for x, y in zip(br.names, br.up)}
        idx, _ = get_compat_rules(up, rules, rules_st)

        if br.up_state == 'unk' and len(idx) == 0:
            cst = br.up # perform analysis on this state
            stop_br = True
            break

        down = {x: y for x, y in zip(br.names, br.down)}
        idx, _ = get_compat_rules(down, rules, rules_st)

        if br.down_state == 'unk' and len(idx) == 0:
            cst = br.down # perform analysis on this state
            stop_br = True
            break

        if br.up_state == 'surv' and br.down_state == 'fail':

            comp, st_up = get_comp_st_for_next_bnb(up, down, rules, rules_st)
            brs2 = decomp_to_two_branches(br, comp, st_up)

            for b in brs2:

                up = {x: y for x, y in zip(br.names, b.up)}
                idx, cst_up = get_compat_rules(up, rules, rules_st)

                if cst_up == 'unk' and len(idx) == 0:
                    cst = b.up # perform analysis on this state
                    stop_br = True
                    break

                else:
                    b.up_state = cst_up

                down = {x: y for x, y in zip(br.names, b.down)}
                idx, cst_down = get_compat_rules(down, rules, rules_st)

                if cst_down == 'unk' and len(idx) == 0:
                    cst = b.down # perform analysis on this state
                    stop_br = True
                    break

                else:
                    b.down_state = cst_down

                    if cst_down == cst_up:
                        b.is_complete = True

                    brs_new.append(b)

            if stop_br == True:
                break

        elif br.up_state != 'unk' and br.up_state == br.down_state:
            brs_new.append(br)

        elif br.is_complete == True:
            brs_new.append(br)

        else:
            b.down_state = cst_down
            if cst_down == cst_up:
                b.is_complete = True

        if stop_br == False:
            brs = copy.deepcopy(brs_new)
        else:
            break

    return brs_new, cst, stop_br


def init_brs(varis, rules, rules_st):

    down = {x: 0 for x in varis.keys()} # all components in the worst state
    up = {k: v.B.shape[1] - 1 for k, v in varis.items()} # all components in the best state

    brs = [branch.Branch(list(down.values()), list(up.values()), is_complete=False, names=list(varis.keys()))]

    _, brs[0].up_state = get_compat_rules(up, rules, rules_st)
    _, brs[0].down_state = get_compat_rules(down, rules, rules_st)

    return brs


def do_gen_bnb(sys_fun, varis, max_br):
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
    flag = True
    brs = []
    cst = []

    while flag and len(brs) < max_br:

        no_iter += 1
        ###############
        print(f'[Iteration {no_iter}]..')
        print(f'The # of found non-dominated rules: {len(rules)}')
        #print('System function runs: ', no_sf) # Redundant with iteration number
        print(f'The # of branches: {len(brs)}')
        print('---')
        ###############

        ## Start from the total event ##
        brs = init_brs(varis, rules, rules_st)
        stop_br = False

        while flag:

            brs, cst, stop_br = core(brs, rules, rules_st, cst, stop_br)

            if stop_br:
                break
            else:
                flag = any([not b.is_complete for b in brs])

        # update rules, rules_st
        sys_res_, rules, rules_st = get_sys_rules(cst, sys_fun, rules, rules_st, varis)
        print(f'go next iteration: {sys_res_["sys_val"].values[0]}')
        sys_res = pd.concat([sys_res, sys_res_], ignore_index=True)

    ###############
    print('[Algorithm completed.]')
    print('The # of found non-dominated rules: ', len(rules))
    print('System function runs: ', no_iter)
    print('The total # of branches: ', len(brs))
    print('The # of incomplete branches: ', sum([not b.is_complete for b in brs]))
    ###############

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
        cst[0] = st_br_to_cs['unk']

    for i, x in enumerate(br.names):
        down = br.down[i]
        up = br.up[i]

        if up > down:
            varis[x], st = get_composite_state(varis[x], list(range(down, up + 1)))
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


def get_composite_state(vari, states):
    """
    # Input: vari-one Variable object, st_list: list of states (starting from zero)
    # TODO: states start from 0 in Cpm and from 1 in B&B -- will be fixed later so that all start from 0
    """

    b = [1 if x in states else 0 for x in range(len(vari.B[0]))]

    comp_st = np.where((vari.B == b).all(axis=1))[0]

    if len(comp_st) > 0:
        cst = comp_st[0]

    else:
        vari.B = np.vstack((vari.B, b))
        cst = len(vari.B) - 1 # zero-based index

    return vari, cst
