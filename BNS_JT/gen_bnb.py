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


def add_rule(rules, rules_st, rule_new, fail_or_surv):
    """
    rules: list of rules
    rules_st: list of rules' state
    rule_new: dict
    fail_or_surv:
    """
    assert isinstance(rule_new, dict), f'rule should be a dict: {type(rule_new)}'

    # Update a rules list by removing dominated rules and adding a new rule
    rmv_inds = []
    add_rule = True

    for i, rule in enumerate(rules):

        if all([k in rule for k in rule_new.keys()]):# does all keys in rule 1 exist for rule_i?

            if fail_or_surv == rules_st[i] == 'surv':

                if all([rule[k] >= v for k, v in rule_new.items()]): # this rule is dominated by the new rule
                    rmv_inds.append(i)

                elif all([k in rule_new for k in rule.keys()]) and all([rule_new[k] >= v for k,v in rule.items()]):
                    add_rule = False
                    break # the new rule is dominated by an existing one. no further investigation required (assuming that a given list is a set of non-dominated rules)

            elif fail_or_surv == rules_st[i] == 'fail':

                if all([rule[k] <= v for k,v in rule_new.items()]): # this rule is dominated by the new rule
                    rmv_inds.append(i)

                elif all([k in rule_new for k in rule.keys()]) and all([rule_new[k] <= v for k,v in rule.items()]):
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
        rules.append(rule_new)
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

    c_rules = [rules[i] for i in set(idx_up + idx_down)]
    _len = [len(x) for x in c_rules]

    idx = sorted(range(len(_len)), key=lambda y: _len[y])
    c_rules = [c_rules[i] for i in idx]
    c_st = [rules_st[i] for i in idx]

    comps_cnt = {}
    comp_bnb = None
    for r, r_st in zip(c_rules, c_st):

        comps = list(r.keys())

        _comps_cnt = [] # counts of components' appearance across rules
        for x in comps:
            if x not in comps_cnt:
                x_cnt = sum([x in y for y in c_rules])
                comps_cnt[x] = x_cnt
            else:
                x_cnt = comps_cnt[x]

            _comps_cnt.append(x_cnt)

        #c_i_sort_ind = [j[0] for j in sorted(enumerate(comps_i_cnt), key=lambda x:x[1])] # order components by their frequency in rules set
        c_ind = sorted(range(len(_comps_cnt)), key=lambda y: _comps_cnt[y])# order components by their frequency in rules set
        for j in c_ind[::-1]:
            comp = comps[j]
            comp_st = r[comp]

            if r_st == 'surv':
                if comp_st > down[comp]:
                    comp_bnb = comp
                    st_bnb_up = comp_st # this is always the upper branch's lower state
                    break

            else: # r_i_st == 'fail'
                if comp_st < up[comp]:
                    comp_bnb = comp
                    st_bnb_up = comp_st + 1 # this is always the upper branch's lower state
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

    #up_bl = copy.deepcopy(up) # the branch on the lower side
    up[comp] = state - 1

    #down = copy.deepcopy(down) # the branch on the upper side
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
            rule = {k: v for k, v in cst.items() if v > 0} # the rule is the same as up_dict_i but includes only components whose state is greater than the worst one (i.e. 0)
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
        print(f'up: {up}, down, {down}')
        if br.down_state == 'unk' and len(idx) == 0:
            cst = br.down # perform analysis on this state
            stop_br = True
            break

        if br.up_state == 'surv' and br.down_state == 'fail':

            comp_bnb, st_bnb_up = get_comp_st_for_next_bnb(up, down, rules, rules_st)
            brs2 = decomp_to_two_branches(br, comp_bnb, st_bnb_up)

            for b in brs2:

                up = {x: y for x, y in zip(br.names, b.up)}
                idx, cst_state_up = get_compat_rules(up, rules, rules_st)

                if cst_state_up == 'unk' and len(idx) == 0:
                    cst = b.up # perform analysis on this state
                    stop_br = True
                    break

                else:
                    b.up_state = cst_state_up

                down = {x: y for x, y in zip(br.names, b.down)}
                idx, cst_state_down = get_compat_rules(down, rules, rules_st)

                if cst_state_down == 'unk' and len(idx) == 0:
                    cst = b.down # perform analysis on this state
                    stop_br = True
                    break

                else:
                    b.down_state = cst_state_down

                    if cst_state_down == cst_state_up:
                        b.is_complete = True

                    brs_new.append(b)

            if stop_br == True:
                break

        elif br.up_state != 'unk' and br.up_state == br.down_state:
            brs_new.append(br)

        elif br.is_complete == True:
            brs_new.append(br)

        #else:
        #    b.down_state = cst_state_down
        #    if cst_state_down == cst_state_up:
        #        b.is_complete = True

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
        print(f"""cst: {cst}
        rules: {rules}
        rules_st: {rules_st}
        brs: {brs}"""
        )
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
        print(f"""cst: {cst}
                rules: {rules}
                rules_st: {rules_st}
                brs: {brs}"""
                )

    ###############
    print('[Algorithm completed.]')
    print('The # of found non-dominated rules: ', len(rules))
    print('System function runs: ', no_iter)
    print('The total # of branches: ', len(brs))
    print('The # of incomplete branches: ', sum([not b.is_complete for b in brs]))
    ###############

    return no_iter, rules, rules_st, brs, sys_res

