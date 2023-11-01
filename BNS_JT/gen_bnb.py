import pandas as pd
import copy
from BNS_JT import variable, branch
import warnings


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

    if no_surv == 0 and no_fail == 0:
        cst_state = 'unk' # unknown
    else:
        if no_surv > no_fail:
            cst_state = 'surv'
        else:
            cst_state = 'fail'

        if no_surv > 0 and no_fail > 0:
            warnings.warn("[get_compat_rules] Conflicting rules found. The given system is not coherent." )

    return cr_inds, cst_state


def add_rule(rules, rules_st, rule1, fail_or_surv):
    """
    rules: list of rules
    rules_st: list of rules' state
    rule1: dict
    fail_or_surv:
    """
    assert isinstance(rule1, dict), f'rule should be a dict: {type(rule1)}'

    # Update a rules list by removing dominated rules and adding a new rule
    r_rmv_inds = []
    add_rule1 = True

    for i, rule in enumerate(rules):

        if all([k in rule for k in rule1.keys()]): # does all keys in rule 1 exist for rule_i?

            if fail_or_surv == 'surv' and rules_st[i] == 'surv':
                if all([rule[k] >= v for k,v in rule1.items()]): # this rule is dominated by the new rule
                    r_rmv_inds += [i]
                elif all([k in rule1 for k in rule.keys()]) and all([rule1[k] >= v for k,v in rule.items()]):
                    add_rule1 = False
                    break # the new rule is dominated by an existing one. no further investigation required (assuming that a given list is a set of non-dominated rules)

            elif fail_or_surv == 'fail' and rules_st[i] == 'fail':
                if all([rule[k] <= v for k,v in rule1.items()]): # this rule is dominated by the new rule
                    r_rmv_inds += [i]
                elif all([k in rule1 for k in rule.keys()]) and all([rule1[k] <= v for k,v in rule.items()]):
                    add_rule1 = False
                    break # the new rule is dominated by an existing one. no further investigation required (assuming that a given list is a set of non-dominated rules)

    rules_new = copy.deepcopy(rules)
    rules_st_new = copy.deepcopy(rules_st)

    for i in r_rmv_inds[::-1]:
        try:
            del rules_new[i]
            del rules_st_new[i]
        except:
            pass

    if add_rule1 == True:
        rules_new += [rule1]
        rules_st_new += [fail_or_surv]

    return rules_new, rules_st_new


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

    cr_inds_up, _ = get_compat_rules(up, rules, rules_st)
    cr_inds_down, _ = get_compat_rules(down, rules, rules_st)

    cr_inds = set(cr_inds_up + cr_inds_down)
    c_rules = [rules[i] for i in cr_inds]
    c_rules_st = [rules_st[i] for i in cr_inds]

    r_len = [len(x) for x in c_rules]
    r_len_sort_ind = [i[0] for i in sorted(enumerate(r_len), key=lambda x:x[1])]

    comps_cnt = {}
    comp_bnb = None
    for i in r_len_sort_ind:
        r_i = c_rules[i]
        r_i_st = c_rules_st[i]
        comps_i = [k for k in r_i]

        comps_i_cnt = [] # counts of components' appearance across rules
        for x in comps_i:
            if x not in comps_cnt:
                x_cnt = sum([x in r for r in c_rules])
                comps_cnt[x] = x_cnt
            else:
                x_cnt = comps_cnt[x]

            comps_i_cnt.append(x_cnt)

        c_i_sort_ind = [j[0] for j in sorted(enumerate(comps_i_cnt), key=lambda x:x[1])] # order components by their frequency in rules set
        for j in c_i_sort_ind[::-1]:
            x_ij = comps_i[j]
            x_ij_st = r_i[x_ij]

            if r_i_st == 'surv':
                if x_ij_st > down[x_ij]:
                    comp_bnb = x_ij
                    st_bnb_up = x_ij_st # this is always the upper branch's lower state
                    break

            else: # r_i_st == 'fail'
                if x_ij_st < up[x_ij]:
                    comp_bnb = x_ij
                    st_bnb_up = x_ij_st + 1 # this is always the upper branch's lower state
                    break

        if comp_bnb is not None:
            break

    # in case nothing has been selected from above
    if comp_bnb is None: # just randomly select from the components that have higher frequencies
        for r in rules:
            for k in r:
                if k not in comps_cnt:
                    comps_cnt[k] = sum([x in r for r in rules])

        cnt_sort_xs = sorted(comps_cnt)
        for x in cnt_sort_xs[::-1]:
            if down[x] < up[x]:
                comp_bnb = x
                st_bnb_up = up[x]
                break

    return comp_bnb, st_bnb_up


def decomp_to_two_branches(br, comp_bnb, st_bnb_up):
    """
    br: a branch
    comp_bnb:
    st_bnb_up:
    """
    down = {y:x for x, y in zip(br.down, br.names)}
    up = {y:x for x, y in zip(br.up, br.names)}

    up_bl = copy.deepcopy(up) # the branch on the lower side
    up_bl[comp_bnb] = st_bnb_up - 1

    down_bu = copy.deepcopy(down) # the branch on the upper side
    down_bu[comp_bnb] = st_bnb_up

    up_bl = [up_bl[x] for x in br.names]
    down_bu = [down_bu[x] for x in br.names]
    new_brs = [branch.Branch(br.down, up_bl, names=br.names, is_complete=False),
               branch.Branch(down_bu, br.up, names=br.names, is_complete=False)]

    return new_brs


def get_sys_rules(cst, sys_fun, rules, rules_st, varis):
    """
    cst:

    """
    cst = {y:x for x, y in zip(cst, varis.keys())}
    #no_sf += 1
    sys_val, sys_st, min_comps_st = sys_fun(cst)
    sys_res = pd.DataFrame({'sys_val': [sys_val], 'comps_st': [cst], 'comps_st_min': [min_comps_st]})
    #sys_res = pd.concat([sys_res,
    #                    pd.DataFrame({'sys_val': [sys_val], 'comps_st': [cst], 'comps_st_min': [min_comps_st]})],
    #                    ignore_index = True)

    if min_comps_st is not None:
        r_new = min_comps_st
    else:
        if sys_st == 'surv':
            r_new = {k:v for k,v in cst.items() if v > 1} # the rule is the same as up_dict_i but includes only components whose state is greater than the worst one (i.e. 1)
        else:
            r_new = {k:v for k,v in cst.items() if v < len(varis[k].B[0])} # the rule is the same as up_dict_i but includes only components whose state is less than the best one

    rules, rules_st = add_rule(rules, rules_st, r_new, sys_st)

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

        up = {y:x for x, y in zip(br.up, br.names)}
        down = {y:x for x, y in zip(br.down, br.names)}

        cr_inds_up, up_st = get_compat_rules(up, rules, rules_st)
        cr_inds_down, down_st = get_compat_rules(down, rules, rules_st)

        if br.up_state == 'unk' and len(cr_inds_up) == 0:
            cst = br.up # perform analysis on this state
            stop_br = True
            break

        elif br.down_state == 'unk' and len(cr_inds_down) == 0:
            cst = br.down # perform analysis on this state
            stop_br = True
            break

        elif br.up_state == 'surv' and br.down_state == 'fail':
            comp_bnb, st_bnb_up = get_comp_st_for_next_bnb(up, down, rules, rules_st)
            brs_new_i = decomp_to_two_branches(br, comp_bnb, st_bnb_up)

            for b in brs_new_i:
                up = {y:x for x, y in zip(b.up, br.names)}
                cr_inds1, cst_state_up = get_compat_rules(up, rules, rules_st)

                if cst_state_up == 'unk' and len(cr_inds1) == 0:
                    cst = b.up # perform analysis on this state
                    stop_br = True
                    break

                else:
                    b.up_state = cst_state_up

                down = {y:x for x, y in zip(b.down, br.names)}
                cr_inds1, cst_state_down = get_compat_rules(down, rules, rules_st)

                if cst_state_down == 'unk' and len(cr_inds1) == 0:
                    cst = b.down # perform analysis on this state
                    stop_br = True
                    break

                else:
                    b.down_state = cst_state_down
                    if cst_state_down == cst_state_up:
                        b.is_complete = True

                    brs_new.append(b)

            #if stop_br == True:
            #    break

        elif br.up_state != 'unk' and br.up_state == br.down_state:
            brs_new.append(br)

        elif br.is_complete == True:
            brs_new.append(br)

        else:
            b.down_state = cst_state_down
            if cst_state_down == cst_state_up:
                b.is_complete = True

    return brs_new, cst, stop_br


def init_brs(varis, rules, rules_st):

    down = {x:1 for x in varis.keys()} # all components in the worst state
    up = {k:v.B.shape[1] for k, v in varis.items()} # all components in the best state

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
        brs = init_brs(varis, rules, rules_st)
        stop_br = False

        while ok:

            brs, cst, stop_br = core(brs, rules, rules_st, cst, stop_br)

            if stop_br:
                break
            else:
                ok = any([not b.is_complete for b in brs])

        # update rules, rules_st
        sys_res_, rules, rules_st = get_sys_rules(cst, sys_fun, rules, rules_st, varis)

        sys_res = pd.concat([sys_res, sys_res_], ignore_index=True)

    ###############
    print('[Algorithm completed.]')
    print('The # of found non-dominated rules: ', len(rules))
    print('System function runs: ', no_iter)
    print('The total # of branches: ', len(brs))
    print('The # of incomplete branches: ', sum([not b.is_complete for b in brs]))
    ###############

    return no_iter, rules, rules_st, brs, sys_res

