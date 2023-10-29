import pandas as pd
import copy
from BNS_JT import variable, branch
import warnings

"""
def rule_dict_to_list(r_dict, r_st, no_comp, comps_name_list, worst_st = 1):

    # TODO: this function is unnecessary if branch's up and down are defined in Dictionary, instead of List.

    if r_st == 'surv':
        r_list = [worst_st] * no_comp

        for i in range(no_comp):
            name_i = comps_name_list[i]

            if name_i in r_dict:
                r_list[i] = r_dict[name_i]

    else: # r_st == 'fail'

        r_list = [best_st] * no_comp

        for i in range(no_comp):
            name_i = comps_name_list[i]

            if name_i in r_dict:
                r_list[i] = r_dict[name_i]

    return r_list

def comps_st_list_to_dict(states, comps_name):

    # function that converts list representation of a component vector state to dictionary
    _states = {comps_name[i]: states[i] for i in range(len(states))}

    return _states

def comps_st_dict_to_list(st_dict, comps_name_list):

    no_comp = len(comps_name_list)
    st_list = [None] * no_comp
    for i in range(no_comp):
        name_i = comps_name_list[i]
        st_list[ i ] = st_dict[name_i]

    return st_list

def comps_st_dict_to_list(states, comps_name):

    assert isinstance(states, dict)
    assert isinstance(comps_name, list)

    return [states.get(x) for x in comps_name]

"""

def get_compat_rules(cst, rules, rules_st):
    #cst: component vector state in dictionary
    #rules: list of rules in dictionary
    #rules_st: list of rules' state (either 'surv' or 'fail') -- must have the same length as rules

    assert isinstance(cst, dict), f'cst should be a dict: {type(cst)}'
    assert isinstance(rules, list), f'rules should be a dict: {type(rules)}'
    assert isinstance(rules_st, list), f'rules_st should be a list: {type(rultes_st)}'

    cr_inds = [] # compatible rules--stored by their indices
    no_surv = 0 # number of compatible survival rules
    no_fail = 0 # number of compatible failure rules
    for ind, r in enumerate(rules):
        if rules_st[ind] == 'surv':
            if all([cst_dict[k] >= r[k] for k in r]): # the survival rule is satisfied
                cr_inds.append(ind )
                no_surv+=1

        else: # rules_st[ind] == 'fail'
            if all([cst_dict[k] <= r[k] for k in r]): # the failure rule is compatible
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


def add_a_new_rule(rules, rules_st, rule1, fail_or_surv):
    """
    rules:
    rules_st:
    rule1:
    fail_or_surv:
    """
    # Update a rules list by removing dominated rules and adding a new rule
    r_rmv_inds = []
    add_rule1 = True

    for i, r_i in enumerate(rules):

        if all([True if k in r_i else False for k in rule1.keys()]): # does all keys in rule 1 exist for rule_i?

            if fail_or_surv == 'surv' and rules_st[i] == 'surv':
                if all([True if r_i[k] >= v else False for k,v in rule1.items()]): # this rule is dominated by the new rule
                    r_rmv_inds += [i]
                elif all([True if k in rule1 else False for k in r_i.keys()]) and all([True if rule1[k] >= v else False for k,v in r_i.items()]):
                    add_rule1 = False
                    break # the new rule is dominated by an existing one. no further investigation required (assuming that a given list is a set of non-dominated rules)

            elif fail_or_surv == 'fail' and rules_st[i] == 'fail':
                if all([True if r_i[k] <= v else False for k,v in rule1.items()]): # this rule is dominated by the new rule
                    r_rmv_inds += [i]
                elif all([True if k in rule1 else False for k in r_i.keys()]) and all([True if rule1[k] <= v else False for k,v in r_i.items()]):
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
    up:
    down:
    rules:
    rules_st:
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
                x_cnt = sum([1 if x in r else 0 for r in c_rules])
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
                    comps_cnt[k] = sum([1 if x in r else 0 for r in rules])

        cnt_sort_xs = sorted(comps_cnt)
        for x in cnt_sort_xs[::-1]:
            if down[x] < up[x]:
                comp_bnb = x
                st_bnb_up = up[x]
                break

    return comp_bnb, st_bnb_up


def decomp_to_two_branches(br1, comp_bnb, st_bnb_up, comps_name):
    """
    br1:
    comp_bnb:
    st_bnb_up:
    comps_name:
    """
    down = {y:x for x, y in zip(br1.down, comps_name)}
    up = {y:x for x, y in zip(br1.up, comps_name)}

    up_bl = copy.deepcopy(up) # the branch on the lower side
    up_bl[comp_bnb] = st_bnb_up - 1

    down_bu = copy.deepcopy(down) # the branch on the upper side
    down_bu[comp_bnb] = st_bnb_up

    up_bl = [up_bl[x] for x in comps_name]
    down_bu = [down_bu[x] for x in comps_name]
    new_brs = [branch.Branch(br1.down, up_bl, is_complete=False),
               branch.Branch(down_bu, br1.up, is_complete=False)]

    return new_brs



def do_gen_bnb(sys_fun, varis, comps_name, max_br):

    ### MAIN FUNCTION ####
    """
    Input:
    sys_fun: system function that takes in a component vector state and returns system function value, system state, and (optional: if unavailable, "None" can be returned) minimially component state to fulfill the obtained system function value.
    varis: A dictionary of "Variable"s. All component events need to be defined with its "B matrix".
    comps_name: A list of components' names. The name must be consistent with the names used in "varis".
    max_br: max. number of branches (the algorithm stops when this number is met)
    """

    # Initialisation
    no_sf = 0 # number of system function runs so far
    sys_res = pd.DataFrame(data={'sys_val': [], 'comps_st': [], 'comps_st_min': []}) # system function results 

    rules = [] # a list of known rules
    rules_st = [] # a list of known rules' states

    no_iter =  0
    _sum = 1
    brs = [branch.Branch([], [], is_complete=False)] # dummy branch to start the while loop
    while _sum > 0 and len(brs) < max_br:

        no_iter += 1
        ###############
        print('[Iteration ', no_iter, ']..')
        print('The # of found non-dominated rules: ', len(rules))
        #print('System function runs: ', no_sf) # Redundant with iteration number
        print('The # of branches: ', len(brs))
        print('---')
        ###############

        ## Start from the total event ##
        down = [1] * len(comps_name) # all components in the worst state
        up = [len(varis[x].B[0]) for x in comps_name] # all components in the best state

        brs = [branch.Branch(down, up, is_complete=False)]

        up_dict = {y:x for x, y in zip(up, comps_name)}
        cr_inds1, cst_state_up = get_compat_rules(up_dict, rules, rules_st)
        brs[0].up_state = cst_state_up

        down_dict = {y:x for x, y in zip(down, comps_name)}
        cr_inds1, cst_state1_down = get_compat_rules(down_dict, rules, rules_st)
        brs[0].down_state = cst_state1_down
        ###############################

        _sum = sum([1 if b.is_complete == False else 0 for b in brs])

        stop_br = False
        while _sum > 0 and len(brs) < max_br:

            brs_new = []

            for i, br_i in enumerate(brs):

                up_dict = {y:x for x, y in zip(br_i.up, comps_name)}
                down_dict = {y:x for x, y in zip(br_i.down, comps_name)}

                cr_inds_up_i, up_st = get_compat_rules(up_dict, rules, rules_st)
                cr_inds_down_i, down_st = get_compat_rules(down_dict, rules, rules_st)

                if br_i.up_state == 'unk' and len(cr_inds_up_i) == 0:
                    cst_list = br_i.up # perform analysis on this state
                    stop_br = True
                    break

                elif br_i.down_state == 'unk' and len(cr_inds_down_i) == 0:
                    cst_list = br_i.down # perform analysis on this state
                    stop_br = True
                    break

                elif br_i.up_state == 'surv' and br_i.down_state == 'fail' :

                    comp_bnb, st_bnb_up = get_comp_st_for_next_bnb(up_dict, down_dict, rules, rules_st)
                    brs_new_i = decomp_to_two_branches(br_i, comp_bnb, st_bnb_up, comps_name)

                    for b in brs_new_i:
                        up_dict = {y:x for x, y in zip(b.up, comps_name)}
                        cr_inds1, cst_state_up = get_compat_rules(up_dict, rules, rules_st)

                        if cst_state_up == 'unk' and len(cr_inds1) == 0:
                            cst_list = b.up # perform analysis on this state
                            stop_br = True
                            break

                        else:
                            b.up_state = cst_state_up

                        down_dict = {y:x for x, y in zip(b.down, comps_name)}
                        cr_inds1, cst_state_down = get_compat_rules(down_dict, rules, rules_st)

                        if cst_state_down == 'unk' and len(cr_inds1) == 0:
                            cst_list = b.down # perform analysis on this state
                            stop_br = True
                            break

                        else:
                            b.down_state = cst_state_down
                            if cst_state_down == cst_state_up:
                                b.is_complete = True

                            brs_new.append(b)

                    if stop_br == True:
                        break

                elif br_i.up_state != 'unk' and br_i.up_state == br_i.down_state:
                    brs_new.append(br_i)

                elif br_i.is_complete == True:
                    brs_new.append(br_i)

                else:
                    b.down_state = cst_state_down
                    if cst_state_down == cst_state_up:
                        b.is_complete = True

            if stop_br == False:
                brs = copy.deepcopy(brs_new)
            else:
                break

            _sum = sum([1 if b.is_complete == False else 0 for b in brs])

        cst_dict = {y:x for x, y in zip(cst_list, comps_name)}

        no_sf += 1
        sys_val1, sys_st1, min_comps_st1 = sys_fun(cst_dict)
        sys_res = pd.concat([sys_res,
                            pd.DataFrame({'sys_val': [sys_val1], 'comps_st': [cst_dict], 'comps_st_min': [min_comps_st1]})],
                            ignore_index = True)

        if sys_st1 == 'surv':
            if min_comps_st1 is not None:
                rs_new = min_comps_st1
            else:
                rs_new = {k:v for k,v in cst_dict.items() if v > 1} # the rule is the same as up_dict_i but includes only components whose state is greater than the worst one (i.e. 1)

            rules, rules_st = add_a_new_rule(rules, rules_st, rs_new, 'surv')

        else: # sys_st_i == 'fail'
            if min_comps_st1 is not None:
                rf_new = min_comps_st1
            else:
                rf_new = {k:v for k,v in cst_dict.items() if v < len(varis[k].B[0])} # the rule is the same as up_dict_i but includes only components whose state is less than the best one

            rules, rules_st = add_a_new_rule(rules, rules_st, rf_new, 'fail')

    ###############
    print('[Algorithm completed.]')
    print('The # of found non-dominated rules: ', len(rules))
    print('System function runs: ', no_sf)
    print('The total # of branches: ', len(brs))
    print('The # of incomplete branches: ', sum([1 if b.is_complete == False else 0 for b in brs ]))
    ###############

    return no_sf, rules, rules_st, brs, sys_res

