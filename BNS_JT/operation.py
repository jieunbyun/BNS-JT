import numpy as np
import textwrap
import copy
import collections
import warnings
import random
from scipy.stats import norm, beta
import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path

#from BNS_JT.utils import all_equal
from BNS_JT import variable
from BNS_JT import utils
from BNS_JT import cpm


def condition(cpms, cnd_vars, cnd_states):
    """
    Returns a list of instance of Cpm

    Parameters
    ----------
    cpms: a dict or list of instances of Cpm
    cnd_vars: a list of variables to be conditioned
    cnd_states: a list of the states to be conditioned
    """

    assert isinstance(cpms, (cpm.Cpm, list, dict)), 'invalid cpms'

    keys = None
    if isinstance(cpms, cpm.Cpm):
        cpms = [cpms]
    elif isinstance(cpms, dict):
        keys = list(cpms.keys())
        cpms = list(cpms.values())

    assert isinstance(cnd_vars, (list, np.ndarray)), 'invalid cnd_vars'

    if isinstance(cnd_vars, np.ndarray):
        cnd_vars = cnd_vars.tolist()

    if cnd_vars and isinstance(cnd_vars[0], str):
        cnd_vars = get_variables(cpms, cnd_vars)

    assert isinstance(cnd_states, (list, np.ndarray)), 'invalid cnd_vars'

    if isinstance(cnd_states, np.ndarray):
        cnd_states = cnd_states.tolist()

    if cnd_states and isinstance(cnd_states[0], str):
        cnd_states = [x.values.index(y) for x, y in zip(cnd_vars, cnd_states)]

    Mc = [Mx.condition(cnd_vars, cnd_states) for Mx in cpms]

    if keys:
        return {k: M for k, M in zip(keys, Mc)}
    else:
        return Mc


def get_var_idx(varis, names):
    """
    INPUT:
    varis: a list of variables
    names: a list of variable names
    OUTPUT:
    idx: a list of column indices of v_names
    """

    idx = []
    for v in names:

        ind = [i for i, k in enumerate(varis) if k.name == v]
        assert len(ind) == 1, f'Each input variable must appear exactly once in M.variables: {v} appears {len(idx)} times.'
        idx.append(ind[0])

    return idx


def get_prod_idx(cpms, varis):
    """
    Returns index for product operation

    Parameters
    ----------
    cpms:
    varis:
    """

    assert isinstance(cpms, (list,  dict)), 'cpms should be a list or dict'
    if isinstance(cpms, dict):
        cpms = cpms.values()

    assert isinstance(varis, (list,  dict)), 'varis should be a list or dict'
    if isinstance(varis, dict):
        varis = varis.values()

    idx = []
    for x in cpms:
        val = set(x.variables[x.no_child:]).difference(varis)
        idx.append(not val)

    try:
        # take integer from the list
        return np.where(idx)[0][0]

    except IndexError as e:
        print(f'CPMs include undefined parent node: {idx}')


def get_sample_order(cpms):

    if isinstance(cpms, dict):

        cpms = list(cpms.values())

    ncpms = len(cpms)
    cpms_ = copy.deepcopy(cpms)
    cpms_idx = list(range(ncpms))

    sample_order = []
    sample_vars = []
    var_add_order = []

    for i in range(ncpms):

        cpm_prod_idx = get_prod_idx(cpms_, sample_vars)

        if cpm_prod_idx is not None:
            sample_order.append(cpms_idx[cpm_prod_idx])
            cpm_prod = cpms_[cpm_prod_idx]

            vars_prod = cpm_prod.variables[:cpm_prod.no_child]

            if set(sample_vars).intersection(vars_prod):
                print('Given Cpms must not have common child nodes')
            else:
                [sample_vars.append(x) for x in vars_prod]

            try:
                var_add_order = np.append(
                    var_add_order,
                    i*np.ones(len(vars_prod)))
            except NameError:
                var_add_order = i*np.ones(len(vars_prod))

            cpms_.pop(cpm_prod_idx)
            cpms_idx.pop(cpm_prod_idx)

    return sample_order, sample_vars, var_add_order


def mcs_product(cpms, nsample, is_scalar=True):
    """
    Returns an instance of Cpm by MC based product operation

    Parameters
    ----------
    cpms: a list of instances of Cpm
    nsample: number of samples
    is_scalar: True if a prob is given as a scalar (all multiplied into one number); False if given as a list for each sampled variables
    """
    sample_order, sample_vars, var_add_order = get_sample_order(cpms)

    nvars = len(sample_vars)
    C_prod = np.zeros((nsample, nvars), dtype=int)

    if is_scalar:
        q_prod = np.zeros((nsample, 1))
    else:
        q_prod = np.zeros((nsample, nvars))

    sample_idx_prod = np.arange(nsample)

    for i in sample_idx_prod:

        sample, sample_prob = single_sample(cpms, sample_order, sample_vars, var_add_order, [i], is_scalar)
        C_prod[i,:] = sample
        if is_scalar:
            q_prod[i] = sample_prob
        else:
            q_prod[i,:] = sample_prob

    if is_scalar:
        q_prod = q_prod[:,::-1]

    return cpm.Cpm(variables=sample_vars[::-1],
               no_child=nvars,
               C = [],
               p = [],
               Cs=C_prod[:, ::-1],
               q=q_prod,
               sample_idx=sample_idx_prod)


def single_sample(cpms, sample_order, sample_vars, var_add_order, sample_idx, is_scalar=True):
    """
    sample from cpms

    parameters:
        cpms: list or dict
        sample_order: list-like
        sample_vars: list-like
        var_add_order: list-like
        sample_idx: list
        is_scalar: True if a prob is given as a scalar (all multiplied into one number); False if given as a list for each sampled variables
    """
    assert isinstance(sample_vars, list), 'should be a list'

    if isinstance(var_add_order, list):
        var_add_order = np.array(var_add_order)

    if isinstance(cpms, dict):
        cpms = cpms.values()

    sample = np.zeros(len(sample_vars), dtype=int)

    for i, (j, M) in enumerate(zip(sample_order, cpms)):

        #FIXME
        cnd_vars = [x for x, y in zip(sample_vars, sample) if y > 0]
        cnd_states = sample[sample > 0]

        M = M.condition(cnd_vars=cnd_vars,
                        cnd_states=cnd_states)

        #if (sample_idx == [1]) and any(M.p.sum(axis=0) != 1):
        #    print('Given probability vector does not sum to 1')

        weight = M.p.flatten()/M.p.sum(axis=0)
        irow = np.random.choice(range(len(M.p)), size=1, p=weight)[0]

        if is_scalar:
            try:
                sample_prob += np.log(M.p[[irow]])
            except NameError:
                sample_prob = np.log(M.p[[irow]])
        else:
            try:
                sample_prob = np.append(
                    sample_prob,
                    np.log(M.p[[irow]]))
            except NameError:
                sample_prob = np.array(np.log(M.p[[irow]]))

        idx = M.C[irow, :M.no_child]
        sample[var_add_order == i] = idx

    try:
        sample_prob = np.exp(sample_prob)
    except NameError:
        return sample, None
    else:
        return sample, sample_prob


def rejection_sampling_sys(cpms, sys_name, sys_fun, nsamp_cov, sys_st_monitor = None, known_prob=0.0, sys_st_prob = 0.0, rand_seed = None):
    """
    Perform rejection sampling on cpms w.r.t. given C
    INPUT:
    - cpms: a list/dictionary of cpms (including system event)
    - sys_name: the key in cpms that represents a system event (a rejection sampling is performed to avoid Csys)
    - nsamp_cov: either number of samples (if integer) or target c.o.v. (if float value)
    - sys_fun: a function (input: comp_st as dictionary and output: sys_val, sys_st)
    - isRejected: True if instances in C be rejected; False if instances not in C be.
    - sys_st_monitor: an integer representing system state to be reference to compute c.o.v. (only necessary when nsamp_cov indicates c.o.v.)
    - known_prob: a float representing already known probability (i.e. those represented by cpms[sys_name].C)
    - sys_st_prob: a float representing already known probility of sys_st
    - rand_seed: a scalar representing random ssed

    OUTPUT:
    - cpms: a list;dictionary of cpms with samples added
    - result: a dictionary including the summary of sampling result
    """
    if rand_seed:
        random.seed(rand_seed)

    assert isinstance(nsamp_cov, (float, int)), 'nsamp_cov must be either an integer (considered number of samples) or a float value (target c.o.v.)'
    if isinstance(nsamp_cov,int):
        is_nsamp = True
        stop = 0 # stop criterion: number of samples
    elif isinstance(nsamp_cov,float):
        is_nsamp = False
        stop = nsamp_cov + 1 # current c.o.v.

    comp_vars = cpms[sys_name].variables[cpms[sys_name].no_child:]
    comp_names = [x.name for x in comp_vars]
    C_reject = cpms[sys_name].C[:,cpms[sys_name].no_child:]

    cpms_no_sys = {}
    for k, m in cpms.items():
        if ( k != sys_name ) and ( set([v.name for v in m.variables[:m.no_child]]) & set([v.name for v in cpms[sys_name].variables]) ):
            cpms_no_sys[k] = m

    sample_order, sample_vars, var_add_order = get_sample_order(cpms_no_sys)
    sample_vars_str = [x.name for x in sample_vars]
    try:
        comp_vars_loc = [sample_vars_str.index(x) for x in comp_names]
    except ValueError:
        comp_vars_loc = []
    else:
        cpms_v_idxs_ = {k: get_var_ind(sample_vars, [y.name for y in x.variables[:x.no_child]])
            for k, x in cpms_no_sys.items()}
        cpms_v_idxs = {k: [cpms_v_idxs_[y.name][0]
            for y in x.variables] for k, x in cpms_no_sys.items()}

    nsamp_tot = 0 # total number of samples (including rejected ones)
    nsamp = 0 # accepted samples
    nfail = 0 # number of occurrences in system state to be monitored
    sys_vals = []
    samples = np.empty((0, len(sample_vars)), dtype=int)
    samples_sys = np.empty((0, 1), dtype=int)
    sample_probs = np.empty((0, len(sample_vars)), dtype=float)

    pf, cov = 0.0, 1.0
    while (is_nsamp and stop < nsamp_cov) or (not is_nsamp and stop > nsamp_cov):

        nsamp_tot += 1

        sample, sample_prob = single_sample(cpms_no_sys, sample_order, sample_vars, var_add_order, [nsamp], is_scalar=False)

        sample_comp = sample[comp_vars_loc]
        is_cpt = cpm.iscompatible(C_reject, comp_vars, comp_names, sample_comp)

        if (~is_cpt).all():

            comp_st = {x:sample_comp[i] for i,x in enumerate(comp_names)}
            sys_val, sys_st = sys_fun(comp_st)

            samples = np.vstack((samples, sample))
            samples_sys = np.vstack((samples_sys, [sys_st]))
            sample_probs = np.vstack((sample_probs, sample_prob))

            nsamp += 1
            if is_nsamp:
                stop += 1
            else:
                if sys_val == sys_st_monitor:
                    nfail +=1

                if nsamp > 9:
                    prior = 0.01
                    a, b = prior + nfail, prior + (nsamp-nfail) # Bayesian estimation assuming beta conjucate distribution
                    pf_s = a / (a+b)
                    var_s = a*b / (a+b)**2 / (a+b+1)
                    std_s = np.sqrt(var_s)

                    pf = sys_st_prob + (1 - known_prob) *pf_s
                    std = (1 - known_prob) * std_s

                    cov = std/pf
                    stop = cov
                else:

                    stop = nsamp_cov + 1 # do not stop the sampling until a few samples are secured

            sys_vals.append(sys_val)

        if nsamp_tot % 1000 == 0: # For monitoring
            print(f'[No. of samples] (total, accept): {nsamp_tot, nsamp}')
            print(f'(pf, c.o.v.): {pf, cov}')


    #Result
    ## Allocate samples to CPMs
    for k, M in cpms_no_sys.items():

        col_loc = cpms_v_idxs[k]
        col_loc_c = cpms_v_idxs[k][:M.no_child]

        Cs = samples[:,col_loc]
        q = sample_probs[:,col_loc_c]

        M2 = cpm.Cpm(variables=M.variables,
                 no_child = M.no_child,
                 C = M.C,
                 p = M.p,
                 Cs = Cs,
                 q = q,
                 sample_idx = np.arange(nsamp))

        cpms[k] = M2

    Cs_sys = np.hstack((samples_sys,samples[:,comp_vars_loc]))
    M = cpms[sys_name]
    M2 = cpm.Cpm(variables=M.variables,
             no_child = M.no_child,
             C = M.C,
             p = M.p,
             Cs = Cs_sys,
             q = np.ones((nsamp,1)), # assuming a deterministic system function
             sample_idx = np.arange(nsamp))

    cpms[sys_name] = M2

    result = {'pf': pf,
              'cov': cov,
              'nsamp_tot': nsamp_tot,
              'nsamp': nsamp,
              'sys_vals': sys_vals}

    return cpms, result


def isinscope(idx, cpms):
    """
    return list of boolean
    idx: list of index
    cpms: list or dict of CPMs
    """
    assert isinstance(idx, list), 'idx should be a list'

    if isinstance(cpms, dict):
        cpms = cpms.values()

    isin = np.zeros((len(cpms), 1), dtype=bool)

    for i in idx:

        flag = np.array([cpm.ismember([i], M.variables)[0] for M in cpms])
        isin = isin | flag

    return isin


def variable_elim(cpms, var_elim, prod=True):
    """
    cpms: list or dict of cpms

    var_elim_order:
    """
    if isinstance(cpms, dict):
        cpms = list(cpms.values())
    else:
        cpms = copy.deepcopy(cpms)

    assert isinstance(var_elim, list), 'var_elim should be a list of variables'

    for _var in var_elim:

        isin = isinscope([_var], cpms)

        sel = [y for x, y in zip(isin, cpms) if x]
        mult = cpm.product(sel)
        mult = mult.sum([_var])

        cpms = [y for x, y in zip(isin, cpms) if x == False]
        cpms.insert(0, mult)

    if prod:
        cpms = cpm.product(cpms)

    return cpms


def variable_elim_cond(cpms, varis, cpms_cond):
    """
    [INPUT]
    cpms: list or dict of cpms for VE
    varis: a list of variables "names" to be eliminated by VE
    cpms_cond: list or dict of cpms for conditioning

    [OUTPUT]
    M: a cpm with varis and variables of cpms_cond eliminated.
    """
    if isinstance(cpms, dict):
        cpms = list(cpms.values())
    else:
        cpms = copy.deepcopy(cpms)

    if isinstance(cpms_cond, dict):
        cpms_cond = list(cpms_cond.values())
    else:
        cpms_cond = copy.deepcopy(cpms_cond)

    assert isinstance(varis, list), 'varis should be a list of variable names'

    c_names = list(set([v.name for x in cpms_cond for v in x.variables]))

    cpms_prod = cpm.product(cpms_cond)

    for cx in cpms_prod.C:

        _cpms_prod = cpms_prod.condition(cpms_prod.variables, cx)
        _cpms_cond = condition(cpms, cpms_prod.variables, cx)

        _cpm = variable_elim(_cpms_cond, varis).product(_cpms_prod)

        idx = _cpm.get_col_ind(c_names)

        try:
            idv
        except NameError:
            idv = [i for i in range(len(_cpm.variables)) if i not in idx]
        finally:
            if idv:
                cpm2 = cpm.Cpm([_cpm.variables[v] for v in idv], _cpm.no_child-len(idx), _cpm.C[:, idv], _cpm.p)

                try:
                    M = M.merge(cpm2)
                except NameError:
                    M = copy.deepcopy(cpm2)
            else:
                del idv

    try:
        M
    except NameError:
        M = None
    finally:
        return M


def get_variables(cpms, variables):

    assert isinstance(cpms, (dict, list)), f'M should be either dict or list but {type(M)}'
    assert isinstance(variables, list), f'variables should be a list but {type(variables)}'
    assert all(isinstance(x, str) for x in variables), f'variables should be a list of str but {type(variables[0])}'

    res = []
    remain = variables[:]

    if isinstance(cpms, dict):
        cpms = cpms.values()

    for M in cpms:
        names = [x.name for x in M.variables]
        i = 0
        while i < len(remain):
            if remain[i] in names:
                res.append(M.get_variables(remain[i]))
                remain.remove(remain[i])
                #i -= 1
            else:
                i += 1

    assert len(res) == len(variables), f'not all variables found in M: {set(variables).difference([x.name for x in res])}'

    return sorted(res, key=lambda x: variables.index(x.name))

"""#FIXME: UNUSED
def append(cpm1, cpm2):
    '''
    return a list of combined cpm1 and cpm2
    cpm1 should be a list or dict
    cpm2 should be a list or dict
    '''

    assert isinstance(cpm1, (list, dict)), 'cpm1 should be a list or dict'
    assert isinstance(cpm2, (list, dict)), 'cpm2 should be a list or dict'

    if isinstance(cpm1, dict):
        cpm1 = list(cpm1.values())

    if isinstance(cpm2, dict):
        cpm2 = list(cpm2.values())

    assert len(cpm1) == len(cpm2), 'Given CPMs have different lengths'
"""


def prod_Msys_and_Mcomps(Msys, Mcomps_list):
    """


    """

    # New CPM's scope
    assert all(x.no_child == 1 for x in Mcomps_list), 'All CPMs of component events must have one child node.'

    sys_vars_ch = Msys.variables[:Msys.no_child] # child variables
    sys_vars_par = [] # parent variables

    comp_names = [x.variables[0].name for x in Mcomps_list]
    mult_idxs = []
    for v in Msys.variables[Msys.no_child:]:

        if v.name in comp_names:
            idx = comp_names.index(v.name)
            sys_vars_ch += [v]
            mult_idxs += [idx]
        else:
            sys_vars_par += [v]

    cond_vars, cond_st = [], []
    if len(Mcomps_list[0].variables) > Mcomps_list[0].no_child: # component CPMs are conditional distributions.
        cond_vars = Mcomps_list[0].variables[Mcomps_list[0].no_child:]

        assert all(x.variables[1:] == cond_vars for x in Mcomps_list), "All CPMS of component events must have the same conditional variables."

        cond_st = Mcomps_list[0].C[0][Mcomps_list[0].no_child:]
        for m in Mcomps_list:
            assert all( c[1:]==cond_st for c in m.C ), "All instances of component events must be conditioned on the same states."


    if Msys.C.size:
        C = np.empty_like(Msys.C)

        for i, v in enumerate(sys_vars_ch):
            col_idx = Msys.get_col_ind([v.name])
            C[:,i] = np.squeeze( Msys.C[:,col_idx] )

        no_ch = len(sys_vars_ch)
        for i, v in enumerate(sys_vars_par):
            col_idx = Msys.get_col_ind([v.name])
            C[:,i+no_ch] = np.squeeze( Msys.C[:,col_idx] )

        if len(cond_vars) > 0:
            for v, s in zip(cond_vars, cond_st):
                if v not in sys_vars_par:
                    sys_vars_par += [v]

                    cond_tiles = np.tile([s], (C.shape[0], 1))
                    C = np.concatenate((C, cond_tiles), axis=1)

        p = Msys.p.copy()
        for i, idx in enumerate(mult_idxs):
            m_i = Mcomps_list[idx]

            c1 = [c[0] for c in m_i.C]
            for j in range(len(p)):
                st = C[j][Msys.no_child + i]
                p_st = 0.0
                for k in m_i.variables[0].B[st]:
                    p_st += m_i.p[c1.index(k)][0]

                p[j] = p[j] * p_st

    if Msys.Cs.size and all(M.Cs.size for M in Mcomps_list):
        cpms_noC = {}
        cpms_noC[Msys.variables[0]] = copy.deepcopy( Msys )
        for v in Msys.variables[Msys.no_child:]:
            if v not in cond_vars:
                m_x = next((m for m in Mcomps_list if m.variables[0].name == v.name), None)
                assert m_x is not None, f'There is no cpm found for component event {v}'
                cpms_noC[v.name] = copy.deepcopy(m_x)

        for k, v in cpms_noC.items():
            v.C, v.p = np.empty((0,len(v.variables))), np.empty((0,1))

        cpm_sys2 = cpm.product(cpms_noC)
        Cs = cpm_sys2.Cs
        q = cpm_sys2.q
        ps = cpm_sys2.ps
        sample_idx = cpm_sys2.sample_idx

        v_idx = cpm_sys2.get_col_ind([v.name for v in sys_vars_ch+sys_vars_par])
        Cs = Cs[:, v_idx]

    else:
        Cs = np.empty((0, len(Msys.variables)))
        q, ps, sample_idx = np.empty((0,1)), np.empty((0,1)), np.empty((0,1))

    return cpm.Cpm(sys_vars_ch+sys_vars_par, len(sys_vars_ch), C, p, Cs, q, ps, sample_idx)


def get_inf_vars(cpms, varis, ve_ord=None):

    """
    INPUT:
    - cpms: a list of CPMs
    - varis: a list of variable names, whose marginal distributions are of interest
    - ve_ord (optional): a list of variable names, representing a VE order. The output list of vars_inf is sorted accordingly.
    OUPUT:
    - varis_inf: a list of variable names
    """

    def get_ord_inf(x, alist):
        try:
            return alist.index(x)
        except ValueError:
            return len(alist)

    if isinstance(varis, str):
        varis = [varis]

    assert isinstance(varis, list), f'varis must be a list: {type(varis)}'

    varis_inf = [] # relevant variables for inference
    varis_cp = varis[:]

    while varis_cp:

        v = varis_cp[0]
        varis_cp.remove(v)
        varis_inf.append(v)

        scope = [x.name for x in cpms[v].variables[cpms[v].no_child:]] # Scope of v1 (child nodes do not have to be multiplied again)
        for p in scope:
            if p not in varis_inf and p not in varis_cp:
                varis_cp.append(p)

    if ve_ord is not None:
        varis_inf.sort(key=(lambda x: get_ord_inf(x, ve_ord)))

    return varis_inf


def cal_Msys_by_cond_VE(cpms, varis, cond_names, ve_order, sys_name):
    """
    INPUT:
    - cpms: a dictionary of cpms
    - varis: a dictionary of variables
    - cond_names: a list of variables to be conditioned
    - ve_order: a list of variables representing an order of variable elimination
    - sys_name: a system variable's name (NB not list!) **FUTHER RESEARCH REQUIRED: there is no efficient way yet to compute a joint distribution of more than one system event

    OUTPUT:
    - Msys: a cpm containing the marginal distribution of variable 'sys_name'
    """

    vars_inf = get_inf_vars(cpms, [sys_name], ve_order) # inference only ancestors of sys_name
    c_names = [x.name for x in cpms[sys_name].variables[cpms[sys_name].no_child:]]
    ve_names = [x for x in vars_inf if (x in ve_order) and (x not in cond_names) and (x not in c_names)]
    c_jnt_names = [x for x in c_names if x not in ve_order] # if non-empty, a joint distribution (sys, c_jnt) is computed
    c_elm_names = [x for x in c_names if x not in c_jnt_names]

    ve_vars = [varis[v] for v in ve_names if v != sys_name] # other variables

    cpms_inf = {v: cpms[v] for v in ve_names + c_names + c_jnt_names}
    cpms_inf[sys_name] = cpms[sys_name]
    cond_cpms = [cpms[v] for v in cond_names]

    M_cond = cpm.product(cond_cpms)
    n_crows = len(M_cond.C)

    for i in range(n_crows):
        #m1 = M_cond.get_subset([i])
        m1 = condition(M_cond, M_cond.variables, M_cond.C[i,:])
        m1 = m1[0]
        VE_cpms_m1 = condition(cpms_inf, m1.variables, m1.C[0])

        for x_name in c_jnt_names:
            VE_cpms_m1[sys_name] = VE_cpms_m1[sys_name].product(VE_cpms_m1[x_name])
            del VE_cpms_m1[x_name]

        VE_cpms_m1_no_sys = {k: v for k,v in VE_cpms_m1.items() if k != sys_name}
        cpms_comps = variable_elim(VE_cpms_m1_no_sys, ve_vars, prod=False)

        m_m1 = prod_Msys_and_Mcomps(VE_cpms_m1[sys_name], cpms_comps)
        m_m1 = m_m1.sum(c_elm_names)

        m_m1 = m_m1.product(m1)
        m_m1 = m_m1.sum(cond_names)

        if i < 1:
            Msys = copy.deepcopy(m_m1)
        else:
            Msys = Msys.merge(m_m1)

    return Msys


# quantify cpms for standard system types
def sys_max_val(name, vars_p, B_flag='store'):

    # Reference: variant of Algorithm MBN-quant-MSSP in Byun, J. E., & Song, J. (2021). Generalized matrix-based Bayesian network for multi-state systems. Reliability Engineering & System Safety, 211, 107468.
    # C_max is quantified for a system whose value is determined as the maximum values of variables in vars_p
    """
    INPUT:
    vars_p: a list of variables (the parent nodes of the node of interest)
    OUTPUT:
    cpm_new: a new CPM representing the system node
    var_new: a new variable representing the system node
    """

    def get_mv(var): # get minimum values
        return min(var.values)

    vars_p_s = copy.deepcopy(vars_p)
    vars_p_s.sort(key=get_mv) # sort variables to minimise # of C's rows

    C_new = np.empty(shape=(0, 1 + len(vars_p)), dtype='int32') # var: [X] + vars_p

    vals_new = []
    for i, p in enumerate(vars_p_s):

        vs_p = copy.deepcopy(vars_p_s[i].values)
        vs_p.sort()

        for v in vs_p:
            c_i = np.zeros(shape=(1, 1 + len(vars_p)), dtype='int32')

            j = vars_p.index(p)
            c_i[0][j + 1] = p.values.index(v)

            add = True
            for i2, p2 in enumerate(vars_p_s):
                if i != i2:
                    if i2 < i:
                        vs_i2 = {y for y, z in enumerate(p2.values) if z < v}

                    if i2 > i:
                        vs_i2 = {y for y, z in enumerate(p2.values) if z <= v}

                    if len(vs_i2) < 1:
                        add = False
                        break
                    else:
                        st_i2 = p2.B.index(vs_i2)

                        j2 = vars_p.index(p2)
                        c_i[0][j2 + 1] = st_i2

            if add:
                if v not in vals_new:
                    vals_new.append(v)
                c_i[0][0] = vals_new.index(v)
                C_new = np.vstack([C_new, c_i])

    vals_new.sort()

    var_new = variable.Variable(name=name, values=vals_new, B_flag='store')
    cpm_new = cpm.Cpm(variables=[var_new] + vars_p, no_child = 1,
                      C=C_new, p=np.ones(shape=(len(C_new), 1), dtype='float64'))

    return cpm_new, var_new


def sys_min_val(name, vars_p, B_flag='store'):

    # Reference: variant of Algorithm MBN-quant-MSSP in Byun, J. E., & Song, J. (2021). Generalized matrix-based Bayesian network for multi-state systems. Reliability Engineering & System Safety, 211, 107468.
    # C_max is quantified for a system whose value is determined as the minimum values of variables in vars_p
    """
    INPUT:
    vars_p: a list of variables (the parent nodes of the node of interest)
    OUTPUT:
    cpm_new: a new CPM representing the system node
    var_new: a new variable representing the system node
    """

    def get_mv(var): # get maximum values
        return max(var.values)

    vars_p_s = copy.deepcopy(vars_p)
    vars_p_s.sort(key=get_mv) # sort variables to minimise # of C's rows

    C_new = np.empty(shape=(0, 1 + len(vars_p)), dtype='int32') # var: [X] + vars_p

    vals_new = []
    for i, p in enumerate(vars_p_s):

        vs_p = copy.deepcopy(vars_p_s[i].values)
        vs_p.sort()

        for v in vs_p:
            c_i = np.zeros(shape=(1, 1 + len(vars_p)), dtype='int32')

            j = vars_p.index(p)
            c_i[0][j + 1] = p.values.index(v)

            add = True
            for i2, p2 in enumerate(vars_p_s):
                if i != i2:
                    if i2 < i:
                        vs_i2 = {y for y, z in enumerate(p2.values) if z > v}

                    if i2 > i:
                        vs_i2 = {y for y, z in enumerate(p2.values) if z >= v}


                    if len(vs_i2) < 1:
                        add = False
                        break
                    else:
                        st_i2 = p2.B.index(vs_i2)

                        j2 = vars_p.index(p2)
                        c_i[0][j2 + 1] = st_i2

            if add:
                if v not in vals_new:
                    vals_new.append(v)

                c_i[0][0] = vals_new.index(v)
                C_new = np.vstack([C_new, c_i])

    vals_new.sort()

    var_new = variable.Variable(name=name, values=vals_new, B_flag='store')
    cpm_new = cpm.Cpm(variables=[var_new] + vars_p, no_child = 1,
                      C=C_new, p=np.ones(shape=(len(C_new), 1), dtype='float64') )

    return cpm_new, var_new

def max_flow(comps_st, target_flow, od_pair, edges, varis): # maximum flow analysis

    """
    INPUT:
    comps_st: a dictionary of component states
    target_flow: an integer 
    od_pair: a tuple
    edges: a dictionary of edge connectivities
    varis: a dictionary of BNS_JT.variable

    OUTPUT:
    f_val: any quantity that indicates a level of system performance (for user information's sake)
    sys_st: either 's' or 'f'
    min_comps_st: a dictionary of minimal survival rules in case 's', None in case of 'f'

    Example:
    nodes = {'n1': (0, 0),
         'n2': (1, 1),
         'n3': (1, -1),
         'n4': (2, 0)}

    edges = {'e1': ['n1', 'n2'],
            'e2': ['n1', 'n3'],
            'e3': ['n2', 'n3'],
            'e4': ['n2', 'n4'],
            'e5': ['n3', 'n4']}

    od_pair=('n1','n4')

    varis = {}
    for k, v in edges.items():
        varis[k] = variable.Variable( name=k, values = [0, 1, 2]) # values: edge flow capacity
    """   

    G = nx.Graph()
    for k,x in comps_st.items():
        G.add_edge(edges[k][0], edges[k][1], capacity=varis[k].values[x])
    G.add_edge(od_pair[1], 'new_t', capacity=target_flow)

    f_val, f_dict = nx.maximum_flow(G,od_pair[0], 'new_t', capacity='capacity', flow_func=shortest_augmenting_path)

    if f_val >= target_flow:
        sys_st = 's'

        min_comps_st = {}
        for k, x in comps_st.items():
            k_flow = max([f_dict[edges[k][0]][edges[k][1]], f_dict[edges[k][1]][edges[k][0]]])
            if k_flow > 0: # the edge plays a role in an associated survival rule
                index = next((i for i,x in enumerate(varis[k].values) if x >= k_flow), None)
                min_comps_st[k] = index

    else:
        sys_st = 'f'
        min_comps_st = None

    return f_val, sys_st, min_comps_st
