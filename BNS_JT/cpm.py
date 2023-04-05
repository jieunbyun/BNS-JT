import numpy as np
import textwrap
import copy
import collections


class Cpm(object):
    """
    Defines the conditional probability matrix (cf., CPT)

    Parameters
    ----------
    variables: array_like
        list of int or string (any hashable python object)
    no_child: int
        number of child nodes
    C: array_like
        event matrix
    p: array_like
        probability vector (n x 1)
    q: array_like
        sampling weight vector for continuous r.v. (n x 1)
    sample_idx: array_like
        sample index vector (n x 1)

    Examples
    --------

    Cpm(varibles, no_child, C, p, q, sample_idx)
    """

    def __init__(self, variables, no_child, C, p, q=[], sample_idx=[]):

        if isinstance(variables, list):
            self.variables = np.array(variables)
        else:
            self.variables = variables

        self.no_child = no_child

        self.C = C

        if isinstance(p, list):
            self.p = np.array(p)[:, np.newaxis]
        else:
            self.p = p

        if isinstance(q, list):
            self.q = np.array(q)[:, np.newaxis]
        else:
            self.q = q

        if isinstance(sample_idx, list):
            self.sample_idx = np.array(sample_idx)[:, np.newaxis]
        else:
            self.sample_idx = sample_idx

        assert len(self.variables), 'variables must be a numeric vector'

        assert all(isinstance(x, (int, np.int32, np.int64)) for x in self.variables), 'variables must be a numeric vector'

        assert isinstance(self.no_child, (int, np.int32, np.int64)), 'no_child must be a numeric scalar'
        assert self.no_child <= len(self.variables), 'no_child must be less than or equal to the number of variables'

        assert isinstance(self.C, np.ndarray), 'Event matrix C must be a numeric matrix'
        assert self.C.dtype in (np.dtype('int64'), np.dtype('int32')), 'Event matrix C must be a numeric matrix'
        if self.C.ndim == 1:
            self.C.shape = (len(self.C), 1)
        else:
            assert self.C.shape[1] == len(self.variables), 'C must have the same number of columns with that of variables'

        assert isinstance(self.p, np.ndarray), 'p must be a numeric vector'
        all(isinstance(y, (float, np.float32, np.float64, int, np.int32, np.int64)) for y in self.p), 'p must be a numeric vector'

        if self.p.ndim == 1:
            self.p.shape = (len(self.p), 1)

        if any(self.q):
            assert isinstance(self.q, np.ndarray), 'q must be a numeric vector'
            all(isinstance(y, (float, np.float32, np.float64, int, np.int32, np.int64)) for y in self.q), 'p must be a numeric vector'

            assert len(self.q) == self.C.shape[0], 'q must have the same length with the number of rows in C'

            if self.q.ndim == 1:
                self.q.shape = (len(self.q), 1)

        else:
            assert len(self.p) == self.C.shape[0], 'p must have the same length with the number of rows in C'

        if any(self.sample_idx):
            assert len(self.sample_idx) == self.C.shape[0], 'sample_idx must have the same length with the number of rows in C'

            if self.sample_idx.ndim == 1:
                self.sample_idx.shape = (len(self.sample_idx), 1)

    def __repr__(self):
        return textwrap.dedent(f'''\
{self.__class__.__name__}(variables={self.variables}, no_child={self.no_child}, C={self.C}, p={self.p}''')

    def get_subset(self, row_idx, flag=True):
        """
        Returns the subset of Cpm

        Parameters
        ----------
        row_idx: array like
        flag: boolean
            default True, 0 if exclude row_idx
        """

        assert flag in (0, 1)

        if flag:
            assert set(row_idx).issubset(range(self.C.shape[0]))
        else:
            # select row excluding the row_index
            row_idx = np.setdiff1d(range(self.C.shape[0]), row_idx)

        if any(self.p):
            p_sub = self.p[row_idx]
        else:
            p_sub = []

        if any(self.q):
            q_sub = self.q[row_idx]
        else:
            q_sub = []

        if any(self.sample_idx):
            sample_idx_sub = self.sample_idx[row_idx]
        else:
            sample_idx_sub = []

        return Cpm(variables=self.variables,
                   no_child=self.no_child,
                   C=self.C[row_idx,:],
                   p=p_sub,
                   q=q_sub,
                   sample_idx=sample_idx_sub)

    def iscompatible(self, M, var=[]):
        """
        Returns a boolean list (n,)

        Parameters
        ----------
        M: instance of Cpm for compatibility check
        var: list or instance of Variable (default: [])
        """

        assert M.C.shape[0] == 1, 'C must be a single row'

        idx = ismember(M.variables, self.variables)
        check_vars = get_value_given_condn(M.variables, idx)
        check_states = get_value_given_condn(M.C[0], idx)
        idx = get_value_given_condn(idx, idx)

        C = self.C[:, idx].copy()

        if any(self.sample_idx) and any(M.sample_idx):
            is_cmp = (self.sample_idx == M.sample_idx)
        else:
            is_cmp = np.ones(shape=C.shape[0], dtype=bool)

        for i, (variable, state) in enumerate(zip(check_vars, check_states)):

            if any(var):
                B = var[variable].B
            else:
                B = np.eye(np.max(C[:, i]))

            x1 = [B[k - 1, :] for k in C[is_cmp, i]]
            x2 = B[state - 1,: ]
            check = (np.sum(x1 * x2, axis=1) >0)

            is_cmp[np.where(is_cmp > 0)[0][:len(check)]] = check

        return is_cmp

    def sum(self, variables, flag=True):
        """
        Returns Sum over CPMs.

        Parameters
        ----------
        variables: variables
        flag: boolean
            1 (default) - sum out variables, 0 - leave only variables
        """

        if flag and any(set(self.variables[self.no_child:]).intersection(variables)):
            print('Parent nodes are NOT summed up')

        if flag:
            vars_rem, vars_rem_idx = setdiff(self.variables[:self.no_child], variables)
        else:
            # FIXME
            vars_rem_idx = ismember(variables, self.variables[:self.no_child])
            vars_rem_idx = get_value_given_condn(vars_rem_idx, vars_rem_idx)
            vars_rem_idx = np.sort(vars_rem_idx)
            vars_rem = self.variables[vars_rem_idx]

        no_child_sum = len(vars_rem)

        if any(self.variables[self.no_child:]):
            vars_rem = np.append(vars_rem, self.variables[self.no_child:])
            vars_rem_idx = np.append(vars_rem_idx, range(self.no_child, len(self.variables)))

        M = Cpm(variables=self.variables[vars_rem_idx],
                C=self.C[:, vars_rem_idx],
                no_child=len(vars_rem_idx),
                p=self.p,
                q=self.q,
                sample_idx=self.sample_idx)

        while M.C.any():

            Mc = M.get_subset([0]) # need to change to 0 
            is_cmp = M.iscompatible(Mc)

            val = M.C[[0]]
            try:
                Csum = np.append(Csum, val, axis=0)
            except NameError:
                Csum = val

            if any(M.p):
                val = np.array([np.sum(M.p[is_cmp])])
                try:
                    psum = np.append(psum, val, axis=0)
                except NameError:
                    psum = val

            if any(M.q):
                val = M.q[0]
                try:
                    qsum = np.append(qsum, val, axis=0)
                except NameError:
                    qsum = val

            if any(M.sample_idx):
                val = M.sample_idx[0]
                try:
                    sample_idx_sum = np.append(sample_idx_sum, val, axis=0)
                except NameError:
                    sample_idx_sum = val

            M = M.get_subset(np.where(is_cmp)[0], flag=0)

        Ms = Cpm(variables=vars_rem,
                 no_child=no_child_sum,
                 C=Csum,
                 p=psum)

        try:
            Ms.q = qsum
        except NameError:
            pass

        try:
            Ms.sample_idx = sample_idx_sum
        except NameError:
            pass

        return Ms

    def product(self, M, var):
        """
        Returns
        M: instance of Cpm
        var: a list of instances of Variable
        """

        assert isinstance(M, Cpm), f'M should be an instance of Cpm'

        if self.C.shape[1] > M.C.shape[1]:
            return M.product(self, var)

        check = set(self.variables[:self.no_child]).intersection(M.variables[:M.no_child])
        assert not bool(check), 'PMFs must not have common child nodes'

        if any(self.p):
            if not any(M.p):
                M.p = np.ones(shape=(M.C.shape[0], 1))
        else:
            if any(M.p):
                self.p = np.ones(shape=(self.C.shape[0], 1))

        if any(self.q):
            if not any(M.q):
                M.q = np.ones(shape=(M.C.shape[0], 1))
        else:
            if any(M.q):
                self.q = np.ones(shape=(self.C.shape[0], 1))

        if self.C.any():
            # FIXME: defined but not used
            #com_vars = list(set(self.variables).intersection(M.variables))

            idx_vars = ismember(self.variables, M.variables)
            com_vars = get_value_given_condn(self.variables, idx_vars)

            for i in range(self.C.shape[0]):

                c1 = get_value_given_condn(self.C[i, :], idx_vars)
                c1_not_com = self.C[i, flip(idx_vars)]

                if self.sample_idx.any():
                    sample_idx = self.sample_idx[i]
                else:
                    sample_idx = []

                [Mc], var = condition([M],
                                        cnd_vars=com_vars,
                                        cnd_states=c1,
                                        var=var,
                                        sample_idx=sample_idx)

                _prod = np.append(Mc.C, np.tile(c1_not_com, (Mc.C.shape[0], 1)), axis=1)

                try:
                    Cprod = np.append(Cprod, _prod, axis=0)
                except NameError:
                    Cprod = _prod

                if any(self.p):
                    _prod = get_prod(Mc.p, self.p[i])

                try:
                    pprod = np.append(pprod, _prod, axis=0)
                except NameError:
                    pprod = _prod

                if any(self.q):
                    _prod = get_prod(Mc.q, self.q[i])

                try:
                    qprod = np.append(qprod, _prod, axis=0)
                except NameError:
                    qprod = _prod

                if any(sample_idx):
                    _prod = np.tile(sample_idx, (Mc.C.shape[0], 1))

                    try:
                        sample_idx_prod = np.append(sample_idx_prod, _prod, axis=0)
                    except NameError:
                        sample_idx_prod = _prod

                elif any(Mc.sample_idx):

                    try:
                        sample_idx_prod = np.append(sample_idx_prod, Mc.sample_idx, axis=0)
                    except NameError:
                        sample_idx_prod = Mc.sample_idx



            prod_vars = np.append(M.variables, get_value_given_condn(self.variables, flip(idx_vars)))

            new_child = np.append(self.variables[:self.no_child], M.variables[:M.no_child])
            new_child = np.sort(new_child)

            new_parent = np.append(self.variables[self.no_child:], M.variables[M.no_child:])
            new_parent = list(set(new_parent).difference(new_child))
            if new_parent:
                new_vars = np.concatenate((new_child, new_parent), axis=0)
            else:
                new_vars = new_child

            idx_vars = ismember(new_vars, prod_vars)

            #print(new_vars)
            #print(new_child)
            #print(Cprod[:, idx_vars])
            #print(pprod)
            Mprod = Cpm(variables=new_vars,
                        no_child = len(new_child),
                        C = Cprod[:, idx_vars],
                        p = pprod)

            if any(qprod):
                Mprod.q = qprod

            # FIXME
            #if any(sample_idx_prod):
            #    Mprod.sample_idx = sample_idx_prod

            Mprod.sort()

        else:
            Mprod = M

        return  Mprod, var


    def sort(self):

        if any(self.sample_idx):
            idx = argsort(self.sample_idx)
        else:
            idx = argsort(list(map(tuple, self.C[:, ::-1])))

        self.C = self.C[idx, :]

        if self.p.any():
            self.p = self.p[idx]

        if self.q.any():
            self.q = self.q[idx]

        if self.sample_idx.any():
            self.sample_idx = self.sample_idx[idx]


def argsort(seq):

    #http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
    return sorted(range(len(seq)), key=seq.__getitem__)


def ismember(A, B):
    """
    A: vector
    B: list
    return the list (same length as A) of index of the first matching elment in B or False for non-matching element
    """


    if isinstance(A, np.ndarray) and (A.ndim > 1):

        assert A.shape[1] == np.array(B).shape[1]

        res = []
        for x in A:
            v = np.where((np.array(B) == x).all(axis=1))[0]
            if len(v):
                res.append(v[0])
            else:
                res.append(False)
    else:

        if isinstance(B, np.ndarray) and (B.ndim > 1):
            assert len(A) == B.shape[1]

        res  = [np.where(np.array(B) == x)[0].min()
                if x in B else False for x in A]

    return res

def setdiff(A, B):
    """
    matlab setdiff equivalent
    """
    C = list(set(A).difference(B))
    ia = [list(A).index(x) for x in C]
    return C, ia


def get_value_given_condn(A, condn):

    if isinstance(A, np.ndarray) and A.ndim==2 and A.shape[1] == len(condn):
        A = A.T
        val = np.array([x for (i, x) in zip(condn, A) if i is not False])
        if val.any():
            val = val.reshape(-1, A.shape[1]).T
    else:
        assert len(A) == len(condn), f'len of {A} is not equal to len of {condn}'
        val = [x for (i, x) in zip(condn, A) if i is not False]

    return val


def iscompatible(C, variables, check_vars, check_states, var):
    """
    Returns a boolean list

    Parameters
    ----------
    C: np.ndarray
    variables: array_like
    check_vars: array_like
    check_sates: array_like
    var: can be dict or list, collection of instance of Variable
    """

    idx = ismember(check_vars, variables)
    check_vars = get_value_given_condn(check_vars, idx)
    check_states = get_value_given_condn(check_states, idx)
    idx = get_value_given_condn(idx, idx)

    C = C[:, idx].copy()

    is_cmp = np.ones(shape=C.shape[0], dtype=bool)

    for i, (variable, state) in enumerate(zip(check_vars, check_states)):

        B = var[variable].B

        x1 = [B[k - 1, :] for k in C[is_cmp, i]]
        try:
            x2 = B[state - 1, :]
        except IndexError:
            print('IndexError: {state}')
        check = (np.sum(x1 * x2, axis=1) > 0)

        is_cmp[np.where(is_cmp > 0)[0][:len(check)]] = check

    return is_cmp


def flip(idx):
    """
    boolean flipped
    Any int including 0 will be flipped False
    """
    return [True if x is False else False for x in idx]


def condition(M, cnd_vars, cnd_states, var, sample_idx=[]):
    """
    Returns an array of conditioned cliques and an array of variable

    Parameters
    ----------
    M: a list or dictionary of instances of Cpm
    cnd_vars: an array of indices of the variables to be conditioned
    cnd_states: an array of the states to be conditioned
    var: an array of Variable
    sample_idx:
    """
    if isinstance(M, Cpm):
        M = [M]

    if isinstance(cnd_vars, list):
        cnd_vars = np.array(cnd_vars)

    if isinstance(cnd_states, list):
        cnd_states = np.array(cnd_states)

    assert isinstance(sample_idx, list), 'sample_idx should be a list'

    Mc = copy.deepcopy(M)
    for Mx in Mc:

        is_cmp = iscompatible(Mx.C, Mx.variables, cnd_vars, cnd_states, var)
        #print(f'is_cmp: {is_cmp}')
        # FIXME
        #if any(sample_idx) and any(Mx.sample_idx):
        #    is_cmp = is_cmp & ( M.sample_idx == sample_idx )
        C = Mx.C[is_cmp, :].copy()
        idx_cnd = ismember(cnd_vars, Mx.variables)
        idx_vars = ismember(Mx.variables, cnd_vars)
        #print(idx_cnd, idx_vars)

        Ccond = np.zeros_like(C)
        not_idx_vars = flip(idx_vars)
        Ccond[:, not_idx_vars] = get_value_given_condn(C, not_idx_vars)
        #print(f'Ccond: {Ccond}')
        #print(f'before: {cnd_vars}, {idx_cnd}')
        cnd_vars = get_value_given_condn(cnd_vars, idx_cnd)
        cnd_states = get_value_given_condn(cnd_states, idx_cnd)
        idx_cnd = get_value_given_condn(idx_cnd, idx_cnd)

        #print(cnd_vars, cnd_states, idx_cnd)
        for cnd_var, state, idx in zip(cnd_vars, cnd_states, idx_cnd):
            #print(cnd_var, state, idx)
            B = var[cnd_var].B.copy()

            if B.any():
                C1 = C[:, idx].copy() - 1
                check = B[C1, :] * B[state - 1,:]
                var[cnd_var].B = add_new_states(check, B)
                Ccond[:, idx] = [x + 1 for x in ismember(check, B)]
                #print(f'C1: {C1}')
                #print(f'check: {check}')

        Mx.C = Ccond.copy()

        if any(Mx.p):
            Mx.p = Mx.p[is_cmp]

        if any(Mx.q):
            Mx.q = Mx.q[is_cmp]

        if any(Mx.sample_idx):
            Mx.sample_idx = Mx.sample_idx[is_cmp]

    return Mc, var


def add_new_states(states, B):
    """

    """

    check = flip(ismember(states, B))
    new_state = states[check, :]

    #FIXME 
    #newState = unique(newState,'rows')    
    if any(new_state):
        B = np.append(B, new_state, axis=1)

    return B


def prod_cpms(cpms, var):
    """

    """
    assert isinstance(cpms, (list,  collections.abc.ValuesView)), 'cpms should be a list'

    prod = cpms[0]
    for c in cpms[1:]:
        prod, var = prod.product(c, var)
    return prod, var


def get_prod(A, B):
    """
    A: matrix
    B: matrix
    """
    assert A.shape[1] == B.shape[0]

    prod_sign = np.sign(A * B)
    prod_val = np.exp(np.log(np.abs(A)) + np.log(np.abs(B)))
    return prod_sign * prod_val


def get_prod_idx(cpms, varis):
    """
    Returns index for product operation

    Parameters
    ----------
    cpms:
    varis:
    """

    assert isinstance(cpms, (list,  collections.abc.ValuesView)), 'cpms should be a list'

    idx = []
    for cpm in cpms:
        val = cpm.variables[cpm.no_child:].tolist()
        val = not any(set(val).difference(varis))
        idx.append(val)

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

    for i in range(ncpms):

        cpm_prod_idx = get_prod_idx(cpms_, sample_vars)

        sample_order.append(cpms_idx[cpm_prod_idx])
        cpm_prod = cpms_[cpm_prod_idx]

        vars_prod = cpm_prod.variables[:cpm_prod.no_child].tolist()

        if any(set(sample_vars).intersection(vars_prod)):
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


def mcs_product(cpms, nsample, varis):
    """
    Returns an instance of Cpm by MC based product operation

    Parameters
    ----------
    cpms: a list of instances of Cpm
    nsample: number of samples
    varis: variables
    """
    sample_order, sample_vars, var_add_order = get_sample_order(cpms)

    nvars = len(sample_vars)
    C_prod = np.zeros((nsample, nvars), dtype=int)
    q_prod = np.zeros((nsample, 1))
    sample_idx_prod = np.arange(nsample)

    for i in sample_idx_prod:

        sample, sample_prob = single_sample(cpms, sample_order, sample_vars, var_add_order, varis, [i])
        C_prod[i,:] = sample
        q_prod[i] = sample_prob

    return Cpm(variables=sample_vars[::-1],
               no_child=nvars,
               C=C_prod[:, ::-1],
               p=[],
               q=q_prod,
               sample_idx=sample_idx_prod)


def single_sample(cpms, sample_order, sample_vars, var_add_order, varis, sample_idx):
    """
    sample from cpms

    parameters:
        cpms: list-like
        sample_order: list-like
        sample_vars: list-like
        var_add_order: list-like
        varis:
        sample_idx: list

    """
    if isinstance(sample_vars, list):
        sample_vars = np.array(sample_vars)

    if isinstance(var_add_order, list):
        var_add_order = np.array(var_add_order)

    sample = np.zeros(len(sample_vars), dtype=int)
    sample_prob = 0.0

    for i, j in enumerate(sample_order):

        cnd_vars = sample_vars[sample > 0]
        cnd_states = sample[sample > 0]

        [cpm], _ = condition(
                    M=cpms[j],
                    cnd_vars=cnd_vars,
                    cnd_states=cnd_states,
                    var=varis,
                    sample_idx=sample_idx)

        if (sample_idx == [1]) and any(cpm.p.sum(axis=0) != 1):
            print('Given probability vector does not sum to 1')

        weight = cpm.p.flatten()/cpm.p.sum(axis=0)
        irow = np.random.choice(range(len(cpm.p)), size=1, p=weight)[0]
        sample_prob += np.log(cpm.p[[irow]])
        idx = cpm.C[irow, :cpm.no_child]
        try:
            sample[var_add_order == i] = idx
        except TypeError:
            print(f'i: {i}')
            print(f'cpm.no_child: {cpm.no_child}')
            print(f'idx: {idx}')

    sample_prob = np.exp(sample_prob)

    return sample, sample_prob

def isinscope(idx, Ms):
    """
    return list of boolean
    idx: list of index
    Ms: list of CPMs
    """
    isin = np.zeros((len(Ms), 1), dtype=bool)
    variables = [M.variables for M in Ms]
    for i in idx:
        flag = [[False] if ismember([i], x)[0] is False else [True] for x in variables]
        isin = isin | np.array(flag)
    return isin
