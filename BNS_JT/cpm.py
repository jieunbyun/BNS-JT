import numpy as np
import textwrap
import copy
import collections
import warnings
import random
from scipy.stats import norm, beta

#from BNS_JT.utils import all_equal
from BNS_JT.variable import Variable
from BNS_JT import utils

class Cpm(object):
    """
    Defines the conditional probability matrix (cf., CPT)

    Parameters
    ----------
    variables: list
        list of instances of Variable
    no_child: int
        number of child nodes
    C: array_like
        event matrix (referencing row of Variable)
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

    def __init__(self, variables, no_child, C=[], p=[], Cs=[], q=[], ps=[], sample_idx=[]):

        self.variables = variables
        self.no_child = no_child
        self.C = C
        self.p = p
        self.Cs = Cs
        self.q = q
        self.ps = ps
        self.sample_idx = sample_idx

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, value):
        assert isinstance(value, list), 'variables must be a list of Variable'
        assert all([isinstance(x, Variable) for x in value]), 'variables must be a list of Variable'
        self._variables = value

    @property
    def no_child(self):
        return self._no_child

    @no_child.setter
    def no_child(self,value):
        assert isinstance(value, (int, np.int32, np.int64)), 'no_child must be a numeric scalar'
        assert value <= len(self._variables), 'no_child must be less than or equal to the number of variables'
        self._no_child = value

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, value):
        if isinstance(value,list):
            value = np.array(value, dtype=np.int32)

        if value.size:
            assert value.dtype in (np.dtype('int64'), np.dtype('int32')), f'Event matrix C must be a numeric matrix: {value}'

            if value.ndim == 1:
                value.shape = (len(value), 1)
            else:
                assert value.shape[1] == len(self._variables), 'C must have the same number of columns as that of variables'

            max_C = np.max(value, axis=0, initial=0)
            max_var = [2**len(x.values)-1 for x in self._variables]
            assert all(max_C <= max_var), f'check C matrix: {max_C} vs {max_var}'

        self._C = value

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        if isinstance(value, list):
            value = np.array(value)[:, np.newaxis]

        if value.ndim == 1:
            value.shape = (len(value), 1)

        if value.size:
            assert len(value) == self._C.shape[0], 'p must have the same length as the number of rows in C'

            all(isinstance(y, (float, np.float32, np.float64, int, np.int32, np.int64)) for y in value), 'p must be a numeric vector'

        self._p = value

    @property
    def Cs(self):
        return self._Cs

    @Cs.setter
    def Cs(self, value):
        if isinstance(value, list):
            value = np.array(value, dtype=np.int32)

        if value.size:
            if value.ndim == 1: # event matrix for samples
                value.shape = (len(value), 1)
            else:
                assert value.shape[1] == len(self._variables), 'Cs must have the same number of columns as the number of variables'

            max_Cs = np.max(value, axis=0, initial=0)
            max_var_basic = [len(x.values) for x in self.variables]
            assert all(max_Cs <= max_var_basic), f'check Cs matrix: {max_Cs} vs {max_var_basic}'

        self._Cs = value

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value):
        if isinstance(value, list):
            value = np.array(value)[:, np.newaxis]

        if value.ndim == 1:
            value.shape = (len(value), 1)

        if value.size and self._Cs.size:
            assert len(value) == self._Cs.shape[0], 'q must have the same length as the number of rows in Cs'

        if value.size:
            all(isinstance(y, (float, np.float32, np.float64, int, np.int32, np.int64)) for y in value), 'q must be a numeric vector'

        self._q = value

    @property
    def ps(self):
        return self._ps

    @ps.setter
    def ps(self, value):
        if isinstance(value, list):
            value = np.array(value)[:, np.newaxis]

        if value.ndim == 1:
            value.shape = (len(value), 1)

        if value.size and self._Cs.size:
            assert len(value) == self._Cs.shape[0], 'ps must have the same length as the number of rows in Cs'

            all(isinstance(y, (float, np.float32, np.float64, int, np.int32, np.int64)) for y in value), 'p must be a numeric vector'

        self._ps = value

    @property
    def sample_idx(self):
        return self._sample_idx

    @sample_idx.setter
    def sample_idx(self, value):
        if isinstance(value, list):
            value = np.array(value)[:, np.newaxis]

        if value.ndim == 1:
            value.shape = (len(value), 1)

        if value.size and self._Cs.size:
            assert len(value) == self._Cs.shape[0], 'sample_idx must have the same length as the number of rows in Cs'

            all(isinstance(y, (float, np.float32, np.float64, int, np.int32, np.int64)) for y in value), 'p must be a numeric vector'

        self._sample_idx = value

    def __repr__(self):
        _variable = [x.name for x in self.variables]
        return textwrap.dedent(f'''\
{self.__class__.__name__}(variables={_variable}, no_child={self.no_child}, C={self.C}, p={self.p}''')


    def get_variables(self, item):

        if isinstance(item, str):
            return [x for x in self.variables if x.name == item][0]
        elif isinstance(item, list):
            return [self.get_variables(y) for y in item]

    def get_subset(self, row_idx, flag=True, isC=True):
        """
        Returns the subset of Cpm

        Parameters
        ----------
        row_idx: array like
        flag: boolean
            default True, 0 if exclude row_idx
        isCs: boolean
            if True, C and p are reduced; if False, Cs, q, ps, sample_idx are.
        """

        assert flag in (0, 1)
        assert isC in (0, 1)

        if isC:
            if flag:
                assert set(row_idx).issubset(range(self.C.shape[0]))
            else:
                # select row excluding the row_index
                row_idx, _ = setdiff(range(self.C.shape[0]), row_idx)

            if self.p.size:
                p_sub = self.p[row_idx]
            else:
                p_sub = []

            Mnew = Cpm(variables=self.variables,
                       no_child=self.no_child,
                       C=self.C[row_idx,:],
                       p=p_sub,
                       Cs = self.Cs,
                       q = self.q,
                       ps = self.ps,
                       sample_idx = self.sample_idx)

        else:
            if flag:
                assert set(row_idx).issubset(range(self.Cs.shape[0]))
            else:
                # select row excluding the row_index
                row_idx, _ = setdiff(range(self.Cs.shape[0]), row_idx)

            if self.q.size:
                q_sub = self.q[row_idx]
            else:
                q_sub = []

            if self.ps.size:
                ps_sub = self.ps[row_idx]
            else:
                ps_sub = []

            if self.sample_idx.size:
                si_sub = self.sample_idx[row_idx]
            else:
                si_sub = []

            Mnew = Cpm(variables=self.variables,
                       no_child=self.no_child,
                       Cs=self.Cs[row_idx,:],
                       q=q_sub,
                       ps=ps_sub,
                       sample_idx = si_sub)

        return Mnew


    def iscompatible(self, M, flag=True):
        """
        Returns a boolean list (n,)

        Parameters
        ----------
        M: instance of Cpm for compatibility check
        flag: True if composite state considered
        """

        assert ( (M.C.shape[0] == 1) and (not M.Cs.size) ) or ( (M.Cs.shape[0] == 1) and (not M.C.size) ), 'C/Cs must be a single row'

        if M.C.size:
            _, idx = ismember(M.variables, self.variables)
            check_vars = get_value_given_condn(M.variables, idx)
            check_states = get_value_given_condn(M.C[0], idx)
            idx = get_value_given_condn(idx, idx)

            C = self.C[:, idx].copy()

            is_cmp = np.ones(shape=C.shape[0], dtype=bool)

            for i, (variable, state) in enumerate(zip(check_vars, check_states)):

                if flag:
                    B = variable.B
                else:
                    B = [{i} for i in range(np.max(C[:, i]) + 1)]

                try:
                    x1 = [B[int(k)] for k in C[is_cmp, i]]
                except TypeError:
                    x1 = [variable.B_fly(int(k)) for k in C[is_cmp, i]]
                    check = [bool(variable.B_fly(state).intersection(x)) for x in x1]
                else:
                    check = [bool(B[state].intersection(x)) for x in x1]

                is_cmp[np.where(is_cmp > 0)[0][:len(check)]] = check

            #FIXME
            """if self.Cs.size:
                _, idx = ismember(M.variables, self.variables[:self.no_child])
                check_vars = get_value_given_condn(M.variables, idx)
                check_states = get_value_given_condn(M.C[0], idx)
                idx = get_value_given_condn(idx, idx)

                Cs = self.Cs[:, idx].copy()

                is_cmp_Cs = np.ones(shape=C.shape[0], dtype=bool)

                for i, (variable, state) in enumerate(zip(check_vars, check_states)):

                    if flag:
                        B = variable.B
                    else:
                        B = [{i} for i in range(np.max(Cs[:, i]) + 1)]

                    x1 = [B[int(k)] for k in Cs[is_cmp_Cs, i]]
                    check = [bool(B[state].intersection(x)) for x in x1]

                    is_cmp_Cs[np.where(is_cmp_Cs > 0)[0][:len(check)]] = check

                is_cmp = {'C': is_cmp_C, 'Cs': is_cmp_Cs}
            else:
                is_cmp = is_cmp_C"""

        """if M.Cs.size: # Cs is not compared with other Cs but only with C.
            _, idx = ismember(M.variables, self.variables)
            check_vars = get_value_given_condn(M.variables, idx)
            check_states = get_value_given_condn(M.C[0], idx)
            idx = get_value_given_condn(idx, idx)

            C = self.C[:, idx].copy()

            is_cmp_C = np.ones(shape=C.shape[0], dtype=bool)

            for i, (variable, state) in enumerate(zip(check_vars, check_states)):

                if flag:
                    B = variable.B
                else:
                    B = [{i} for i in range(np.max(C[:, i]) + 1)]

                x1 = [B[int(k)] for k in C[is_cmp_C, i]]
                check = [bool(B[state].intersection(x)) for x in x1]

                is_cmp_C[np.where(is_cmp_C > 0)[0][:len(check)]] = check

            is_cmp = is_cmp_C"""

        return is_cmp

    # FIXME: NOT_USED
    def get_col_ind(self, v_names):
        """
        INPUT:
        v_names: a list of variable names
        OUTPUT:
        v_idxs: a list of column indices of v_names
        """

        v_idxs = []
        for v in v_names:
            idx = [i for (i,k) in enumerate(self.variables) if k.name == v]

            assert len(idx) == 1, f'Each input variable must appear exactly once in M.variables: {v} appears {len(idx)} times.'

            v_idxs.append(idx[0])

        return v_idxs


    def sum(self, variables, flag=True):
        """
        Returns instance of Cpm with based on Sum over CPMs.

        Parameters
        ----------
        variables: list of variables or names of variables
        flag: boolean
            1 (default) - sum out variables, 0 - leave only variables
        """

        assert isinstance(variables, list), 'variables should be a list'
        if variables and isinstance(variables[0], str):
            variables = self.get_variables(variables)

        if flag and set(self.variables[self.no_child:]).intersection(variables):
            print('Parent nodes are NOT summed up')

        if flag:
            vars_rem, vars_rem_idx = setdiff(self.variables[:self.no_child], variables)

        else:
            # FIXME
            _, vars_rem_idx = ismember(variables, self.variables[:self.no_child])
            vars_rem_idx = get_value_given_condn(vars_rem_idx, vars_rem_idx)
            vars_rem_idx = sorted(vars_rem_idx)
            vars_rem = [self.variables[x] for x in vars_rem_idx]

        no_child_sum = len(vars_rem)

        if self.variables[self.no_child:]:
            vars_rem += self.variables[self.no_child:]
            vars_rem_idx += list(range(self.no_child, len(self.variables)))

        _variables = [self.variables[i] for i in vars_rem_idx]

        M = Cpm(variables=_variables,
                C=self.C[:, vars_rem_idx],
                no_child=len(vars_rem_idx),
                p=self.p,
                q=self.q,
                sample_idx=self.sample_idx)

        Csum, psum, qsum, sample_idx_sum = [], [], [], []

        while M.C.size:

            Mc = M.get_subset([0]) # need to change to 0 
            is_cmp = M.iscompatible(Mc, flag=False)

            Csum.append(M.C[[0]])

            if M.p.size:
                psum.append(np.sum(M.p[is_cmp]))

            if M.q.size:
                qsum.append(M.q[0])

            if M.sample_idx.size:
                sample_idx_sum.append(M.sample_idx[0])

            M = M.get_subset(np.where(is_cmp)[0], flag=0)

        Ms = Cpm(variables=vars_rem,
                 no_child=no_child_sum,
                 C=np.reshape(Csum, (-1, M.C.shape[1])),
                 p=np.reshape(psum, (-1, 1)))
        
        if self.Cs.size:
            lia, res = ismember(vars_rem, self.variables)

            Cs = np.empty((len(self.Cs), len(vars_rem)), dtype=np.int32)
            for i,r in enumerate(res):
                Cs[:,i] = self.Cs[:,r]

            Ms.Cs = Cs.copy()
            Ms.q = self.q.copy()
            Ms.ps = self.ps.copy()
            Ms.sample_idx = self.sample_idx.copy()

        return Ms


    def product(self, M):
        """
        Returns an instance of Cpm
        M: instance of Cpm
        var: a dict of instances of Variable
        """

        assert isinstance(M, Cpm), f'M should be an instance of Cpm'

        if self.C.shape[1] > M.C.shape[1]:
            return M.product(self)

        first = [x.name for x in self.variables[:self.no_child]]
        second = [x.name for x in M.variables[:M.no_child]]
        check = set(first).intersection(second)
        assert not bool(check), f'PMFs must not have common child nodes: {first}, {second}'

        if self.p.size:
            if not M.p.size:
                M.p = np.ones(shape=(M.C.shape[0], 1))
        else:
            if M.p.size:
                self.p = np.ones(shape=(self.C.shape[0], 1))

        """if self.q.size:
            if not M.q.size:
                M.q = np.ones(shape=(M.Cs.shape[0], 1))
        else:
            if M.q.size:
                self.q = np.ones(shape=(self.Cs.shape[0], 1))"""

        if self.C.size or self.Cs.size:
            idx_vars, _ = ismember(self.variables, M.variables)
            com_vars = get_value_given_condn(self.variables, idx_vars)
            prod_vars = M.variables + get_value_given_condn(self.variables, flip(idx_vars))

            new_child = self.variables[:self.no_child] + M.variables[:M.no_child]
            ##FIXME: sort required?
            new_child = sorted(new_child)

            new_parent = self.variables[self.no_child:] + M.variables[M.no_child:]
            new_parent, _ = setdiff(new_parent, new_child)

            if new_parent:
                new_vars = new_child + new_parent
            else:
                new_vars = new_child
            _, idx_vars2 = ismember(new_vars, prod_vars)

            Mprod = Cpm(variables=new_vars,
                        no_child = len(new_child))
        else:
            Mprod = M
            

        if self.C.size:
            # FIXME: defined but not used
            #com_vars = list(set(self.variables).intersection(M.variables))

            Cprod, pprod = [], []
            for i in range(self.C.shape[0]):

                c1 = get_value_given_condn(self.C[i, :], idx_vars)
                c1_not_com = self.C[i, flip(idx_vars)]

                [Mc] = condition([M],
                                 cnd_vars=com_vars,
                                 cnd_states=c1)

                _cprod = np.append(Mc.C, np.tile(c1_not_com, (Mc.C.shape[0], 1)), axis=1)

                Cprod.append(_cprod)

                if self.p.size:
                    pprod.append(get_prod(Mc.p, self.p[i]))

            Cprod = np.concatenate(Cprod, axis=0)
            Cprod = Cprod[:, idx_vars2].astype(int)
            pprod = np.concatenate(pprod, axis=0)

        else:
            Cprod = np.empty((0,len(idx_vars2)), dtype=int)
            pprod = np.empty((0,1), dtype=float)

        Mprod.C = Cprod
        Mprod.p = pprod
        Mprod.sort()
            

        if self.Cs.size and M.Cs.size:
            Csprod, qprod, psprod, sample_idx_prod = [], [], [], []

            self.q = np.prod(self.q, axis=1)
            M.q = np.prod(M.q, axis = 1)
            if self.ps.size:
                self.ps = np.prod(self.ps, axis=1)
            else:
                self.ps = self.q.copy()
            if M.ps.size:
                M.ps = np.prod(M.ps, axis=1)
            else:
                M.ps = M.q.copy()

            for i in range(self.Cs.shape[0]):

                #c1 = get_value_given_condn(self.Cs[i, :], idx_vars)
                c1_not_com = self.Cs[i, flip(idx_vars)]

                M_idx = np.where(M.sample_idx==self.sample_idx[i][0])[0]
                c2 = M.Cs[M_idx,:][0]

                _csprod = np.concatenate((c2, c1_not_com), axis = 0 )
                Csprod.append([_csprod])

                qprod.append(get_prod(M.q[M_idx], self.q[i])) 
                psprod.append(get_prod(M.ps[M_idx], self.ps[i]))

            Csprod = np.concatenate(Csprod, axis=0)
            psprod = np.concatenate(psprod, axis=0)
            qprod = np.concatenate(qprod, axis=0)

            Mprod.Cs = Csprod[:, idx_vars2].astype(int)
            Mprod.q = qprod
            Mprod.ps = psprod
            Mprod.sample_idx = self.sample_idx.copy()

        return  Mprod


    def sort(self):

        if self.sample_idx.size:
            idx = argsort(self.sample_idx)
        else:
            idx = argsort(list(map(tuple, self.C[:, ::-1])))

        self.C = self.C[idx, :]

        if self.p.size:
            self.p = self.p[idx]

        if self.q.size:
            self.q = self.q[idx]

        if self.sample_idx.size:
            self.sample_idx = self.sample_idx[idx]


def argsort(seq):

    #http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
    return sorted(range(len(seq)), key=seq.__getitem__)


def ismember(A, B):
    """
    A: vector
    B: list
    return
     Lia: logical true and false where data in A is found in B
     Lib: the list (same length as A) of index of the first matching elment in B or False for non-matching element

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

    elif isinstance(A, list) and isinstance(B, list):

        res  = [B.index(x) if x in B else False for x in A]

    else:

        if isinstance(B, np.ndarray) and (B.ndim > 1):
            assert len(A) == B.shape[1]

        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            res  = [np.where(np.array(B)==x)[0].min()
                 if x in B else False for x in A]

    lia = [False if x is False else True for x in res]

    return lia, res


def setdiff(first, second):
    """
    matlab setdiff equivalent
    """
    second = set(second)
    first = list(dict.fromkeys(first))
    val = [item for item in first if item not in second]
    iv = [first.index(x) for x in val]
    return val, iv


def get_value_given_condn(A, condn):

    if isinstance(A, np.ndarray) and A.ndim==2 and A.shape[1] == len(condn):
        A = A.T
        val = np.array([x for (i, x) in zip(condn, A) if i is not False])
        if val.size:
            val = val.reshape(-1, A.shape[1]).T
    else:
        assert len(A) == len(condn), f'len of {A} is not equal to len of {condn}'
        val = [x for (i, x) in zip(condn, A) if i is not False]

    return val


def iscompatible(C, variables, check_vars, check_states):
    """
    Returns a boolean list

    Parameters
    ----------
    C: np.ndarray
    variables: array_like
    check_vars: array_like list of Variable or string
    check_sates: array_like list of index or string
    var: dict of instance of Variable
    """
    if check_vars and isinstance(check_vars[0], str):
        check_vars = [x for y in check_vars for x in variables if x.name == y]
        
    _, idx = ismember(check_vars, variables)
    check_vars = get_value_given_condn(check_vars, idx)
    check_states = get_value_given_condn(check_states, idx)
    idx = get_value_given_condn(idx, idx)

    C = C[:, idx].copy()

    is_cmp = np.ones(shape=C.shape[0], dtype=bool)

    for i, (variable, state) in enumerate(zip(check_vars, check_states)):

        if isinstance(state, str):
            state = variable.values.index(state)

        try:
            B = variable.B
        except NameError:
            print(f'{variable} is not defined')

        try:
            x1 = [B[int(k)] for k in C[is_cmp, i]]
        except TypeError:
            x1 = [variable.B_fly(int(k)) for k in C[is_cmp, i]]

        try:
            check = [bool(B[state].intersection(x)) for x in x1]
        except IndexError:
            print('IndexError: {state}')
        except TypeError:
            check = [bool(variable.B_fly(state).intersection(x)) for x in x1]
            is_cmp[np.where(is_cmp > 0)[0][:len(check)]] = check
        else:
            is_cmp[np.where(is_cmp > 0)[0][:len(check)]] = check

    return is_cmp


def flip(idx):
    """
    boolean flipped
    Any int including 0 will be flipped False
    """
    return [True if x is False else False for x in idx]


def condition(M, cnd_vars, cnd_states, sample_idx=[]):
    """
    Returns a list of instance of Cpm

    Parameters
    ----------
    M: a dict or list of instances of Cpm
    cnd_vars: a list of variables to be conditioned
    cnd_states: a list of the states to be conditioned
    """

    assert isinstance(M, (Cpm, list, dict)), 'invalid M'

    if isinstance(M, Cpm):
        M = [M]
    elif isinstance(M, dict):
        keys = list(M.keys())
        M = list(M.values())

    assert isinstance(cnd_vars, (list, np.ndarray)), 'invalid cnd_vars'

    if isinstance(cnd_vars, np.ndarray):
        cnd_vars = cnd_vars.tolist()

    if cnd_vars and isinstance(cnd_vars[0], str):
        cnd_vars = get_variables_from_cpms(M, cnd_vars)

    assert isinstance(cnd_states, (list, np.ndarray)), 'invalid cnd_vars'

    if isinstance(cnd_states, np.ndarray):
        cnd_states = cnd_states.tolist()

    if cnd_states and isinstance(cnd_states[0], str):
        cnd_states = [x.values.index(y) for x, y in zip(cnd_vars, cnd_states)]

    Mc = copy.deepcopy(M)
    for Mx in Mc:

        is_cmp = iscompatible(Mx.C, Mx.variables, cnd_vars, cnd_states)
        # FIXME
        #if any(sample_idx) and any(Mx.sample_idx):
        #    is_cmp = is_cmp & ( M.sample_idx == sample_idx )
        C = Mx.C[is_cmp, :].copy()
        _, idx_cnd = ismember(cnd_vars, Mx.variables)
        _, idx_vars = ismember(Mx.variables, cnd_vars)

        Ccond = np.zeros_like(C)
        not_idx_vars = flip(idx_vars)

        if C.size:
            Ccond[:, not_idx_vars] = get_value_given_condn(C, not_idx_vars)

        cnd_vars_m = get_value_given_condn(cnd_vars, idx_cnd)
        cnd_states_m = get_value_given_condn(cnd_states, idx_cnd)
        idx_cnd = get_value_given_condn(idx_cnd, idx_cnd)

        for cnd_var, state, idx in zip(cnd_vars_m, cnd_states_m, idx_cnd):
            try:
                B = cnd_var.B.copy()
            except NameError:
                print(f'{cnd_var} is not defined')
            else:
                C1 = C[:, idx].copy().astype(int)
                #check = [bool(B[state].intersection(x)) for x in B[C1]]
                check = [B[x].intersection(B[state]) for x in C1]
                #B = add_new_states(check, B)
                #cnd_var = Variable(name=cnd_var.name,
                #                   #B=add_new_states(check, B),
                #                   values=cnd_var.values)
                Ccond[:, idx] = [x for x in ismember(check, B)[1]]

        Mx.C = Ccond.copy()

        if Mx.p.size:
            Mx.p = Mx.p[is_cmp]

        if Mx.Cs.size: # NB: ps is not properly updated if the corresponding instance is not in C.
            _, res = ismember(Mx.variables, cnd_vars)
            if any(not isinstance(r,bool) for r in res[:Mx.no_child]): # conditioned variables belong to child nodes
                ps = []
                for c, q in zip(Mx.Cs, Mx.q):
                    var_states = [cnd_states[r] if not isinstance(r,bool) else c[i] for i,r in enumerate(res)]
                    pr = get_prob(Mx, Mx.variables, var_states)
                    ps.append(pr*q[0])

                Mx.ps = ps # the conditioned variables' samples are removed.

            elif all(isinstance(r,bool) and r==False for r in res): # conditioned variables are not in the scope.
                Mx.ps = Mx.q.copy()

            else: # conditioned variables are in parent nodes.
                ps = []
                for c in Mx.Cs:
                    var_states = [cnd_states[r] if not isinstance(r,bool) else c[i] for i,r in enumerate(res)]
                    pr = get_prob(Mx, Mx.variables, var_states)
                    ps.append(pr)

                Mx.ps = ps

    try:
        return {k: M for k,M in zip(keys,Mc)}
    except NameError:
        return Mc


#FIXME: UNUSED
def add_new_states(states, B):
    """

    """
    _, check = ismember(states, B)
    check = flip(check)
    new_state = [states[i] for i in check if i]

    #FIXME 
    #newState = unique(newState,'rows')    
    if new_state:
        [B.append(x) for x in new_state]

    return B


def prod_cpms(cpms):
    """
    return an instance of Cpm

    cpms: a list or dict of instances of Cpm
    """
    assert isinstance(cpms, (list,  dict)), 'cpms should be a list or dict'

    if isinstance(cpms, dict):
        cpms = list(cpms.values())

    assert cpms, f'{cpms} is empty list'

    prod = cpms[0]
    for c in cpms[1:]:
        prod = prod.product(c)

    return prod


def get_prod(A, B):
    """
    A: matrix
    B: matrix
    """
    if len(A.shape) < 2:
        A = np.reshape(A, (A.shape[0], 1))
    if len(B.shape) < 1:
        B=np.reshape(B,(1,))

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

    assert isinstance(cpms, (list,  dict)), 'cpms should be a list or dict'

    if isinstance(cpms, dict):
        cpms = cpms.values()

    idx = []
    for cpm in cpms:
        val = cpm.variables[cpm.no_child:]
        val = not set(val).difference(varis)
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

    return Cpm(variables=sample_vars[::-1],
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

        [M] = condition(
                    M=M,
                    cnd_vars=cnd_vars,
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

    sample_prob = np.exp(sample_prob)

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
    comp_vars_loc = [sample_vars_str.index(x) for x in comp_names]

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

    pf, cov = 0, 1
    while (is_nsamp and stop < nsamp_cov) or (not is_nsamp and stop > nsamp_cov):

        nsamp_tot += 1

        sample, sample_prob = single_sample(cpms_no_sys, sample_order, sample_vars, var_add_order, [nsamp], is_scalar=False)

        sample_comp = sample[comp_vars_loc]
        is_cpt = iscompatible(C_reject, comp_vars, comp_names, sample_comp)

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

        M2 = Cpm(variables=M.variables,
                 no_child = M.no_child,
                 C = M.C,
                 p = M.p,
                 Cs = Cs,
                 q = q,
                 sample_idx = np.arange(nsamp))

        cpms[k] = M2

    Cs_sys = np.hstack((samples_sys,samples[:,comp_vars_loc]))
    M = cpms[sys_name]
    M2 = Cpm(variables=M.variables,
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


def isinscope(idx, Ms):
    """
    return list of boolean
    idx: list of index
    Ms: list or dict of CPMs
    """
    assert isinstance(idx, list), 'idx should be a list'

    if isinstance(Ms, dict):
        Ms = Ms.values()

    isin = np.zeros((len(Ms), 1), dtype=bool)

    for i in idx:

        flag = np.array([ismember([i], M.variables)[0] for M in Ms])
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
        mult = prod_cpms(sel)
        mult = mult.sum([_var])

        cpms = [y for x, y in zip(isin, cpms) if x == False]
        cpms.insert(0, mult)

    if prod:
        cpms = prod_cpms(cpms)

    return cpms


def get_prob(M, var_inds, var_states, flag=True):

    assert isinstance(M, Cpm), 'Given CPM must be a single CPM'
    assert isinstance(var_inds, list), 'var_inds should be a list'
    if var_inds and isinstance(var_inds[0], str):
        var_inds = M.get_variables(var_inds)

    assert isinstance(var_states, (list, np.ndarray)), 'var_states should be an array'

    assert len(var_inds) == len(var_states), f'"var_inds" {var_inds} and "var_states" {var_states} must have the same length.'

    assert flag in (0, 1), 'Operation flag must be either 1 (keeping given row indices default) or 0 (deleting given indices)'

    _var_states = []
    for i, x in enumerate(var_states):
        if isinstance(x, str):
            assert x in var_inds[i].values, f'{x} not in {var_inds[i].values}'
            _var_states.append(var_inds[i].values.index(x))

    if _var_states:
        var_states = _var_states[:]

    if isinstance(var_states, list):
        var_states = np.array(var_states)
        var_states = np.reshape(var_states, (1, -1))

    Mcompare = Cpm(variables=var_inds,
                   no_child=len(var_inds),
                   C=var_states,
                   p=np.empty(shape=(var_states.shape[0], 1)))
    is_compat = M.iscompatible(Mcompare, flag=True)
    idx = np.where(is_compat)[0]
    Msubset = M.get_subset(idx, flag )
    prob = Msubset.p.sum()

    return prob

def get_prob_bnd(M, var_inds, var_states, flag=True, cvar_inds=None, cvar_states=None, cflag=True):

    prob1 = get_prob( M, var_inds, var_states, flag )
    prob_unk = 1.0 - np.sum(M.p) # Unknown probs
    prob1_bnd = [prob1, prob1 + prob_unk]

    if cvar_inds is None:
        prob_bnd = prob1_bnd
    else:
        prob2 =  get_prob( M, cvar_inds, cvar_states, cflag )
        prob2_bnd = [prob2, prob2 + prob_unk]

        prob_bnd = [prob1_bnd[0] / prob2_bnd[1], prob1_bnd[1] / prob2_bnd[0]]

    prob_bnd[1] = min( 1, prob_bnd[1] )
    return prob_bnd



def get_prob_and_cov(M, var_inds, var_states, method='MLE', flag=True, nsample_repeat = 0, conf_p=0.95):

    assert isinstance(nsample_repeat, int), 'nsample_repeat must be a nonnegative integer, representing if samples are repeated (to calculate c.o.v.)'

    prob_C = get_prob(M, var_inds, var_states, flag)

    if not nsample_repeat:
        n_round = 1
        nsamp = len(M.Cs)
    else:
        assert len(M.Cs) % nsample_repeat == 0, 'Given number of samples is not divided by given nsample_repeat'
        n_round = int(len(M.Cs) / nsample_repeat)
        nsamp = nsample_repeat

    prob_Cs = 0
    var = 0
    for i in range(n_round):
        row_range = range(i*nsamp, (i + 1)*nsamp)
        is_cmp = iscompatible(M.Cs[row_range,:], M.variables, var_inds, var_states)

        try:
            w = M.ps[row_range] / M.q[row_range]
        except IndexError:
            w = np.ones_like(M.q) # if ps is empty, assume ps is the same as q

        w_ori = w.copy() # weight before compatibility check
        if flag:
            w[~is_cmp] = 0
        else:
            w[is_cmp] = 0

        if method=='MLE':
            mean = w.sum() / nsamp
            prob_Cs += mean

            if np.allclose(w_ori, w_ori[0], atol=1e-4): # this is MCS
                var1 = np.square( w_ori[0] ) * (1-mean) * mean / nsamp
                var += var1[0]
            else:
                var1 = np.square( ( w - mean ) / nsamp )
                var += var1.sum()

        elif method=='Bayesian':
            neff = len(w_ori)*w_ori.mean()**2 / (sum(x**2 for x in w_ori)/len(w_ori)) # effective sample size
            w_eff = w / w_ori.sum() *neff
            nTrue = w_eff.sum()

            try:
                a, b = a + nTrue, b + (neff[0] - nTrue)
            except NameError:
                prior = 0.01
                a, b = prior + nTrue, prior + (neff[0] - nTrue) 

    if method == 'MLE':
        prob = prob_C + (1 - M.p.sum()) * prob_Cs
        cov = (1 - M.p.sum()) * np.sqrt(var) / prob

        # confidence interval
        z = norm.pdf( 1 - (1 - conf_p)*0.5 ) # for both tails
        prob_Cs_int = prob_Cs + z * np.sqrt(var) * np.array([-1, 1])
        cint = prob_C + (1 - M.p.sum()) * prob_Cs_int

    elif method == 'Bayesian':

        mean = a / (a + b)
        var = a*b / (a+b)**2 / (a+b+1)

        prob = prob_C + (1 - M.p.sum()) * mean
        cov = (1 - M.p.sum()) * np.sqrt(var) / prob

        low = beta.ppf( 0.5*(1-conf_p), a, b )
        up = beta.ppf( 1 - 0.5*(1-conf_p), a, b )
        cint = prob_C + (1 - M.p.sum()) * np.array([low, up])

    return prob, cov, cint

def get_prob_and_cov_cond(M, var_inds, var_states, cvar_inds, cvar_states, nsample_repeat = 0, conf_p=0.95):
    # Assuming beta distribution (i.e. Bayeisan inference)

    assert isinstance(nsample_repeat, int), 'nsample_repeat must be a nonnegative integer, representing if samples are repeated (to calculate c.o.v.)'

    if not nsample_repeat:
        n_round = 1
        nsamp = len(M.Cs)
    else:
        assert len(M.Cs) % nsample_repeat == 0, 'Given number of samples is not divided by given nsample_repeat'
        n_round = int(len(M.Cs) / nsample_repeat)
        nsamp = nsample_repeat

    for i in range(n_round):
        row_range = range(i*nsamp, (i + 1)*nsamp)

        try:
            w_ori = M.ps[row_range] / M.q[row_range] # weight before compatibility check
        except IndexError:
            w_ori = np.ones_like(M.q) # if ps is empty, assume ps is the same as q

        is_cmp1 = iscompatible(M.Cs[row_range,:], M.variables, var_inds, var_states)
        w1 = w_ori.copy() 
        w1[~is_cmp1] = 0

        is_cmp2 = iscompatible(M.Cs[row_range,:], M.variables, cvar_inds, cvar_states)
        w2 = w_ori.copy() 
        w2[~is_cmp2] = 0       

        neff = len(w_ori)*w_ori.mean()**2 / (sum(x**2 for x in w_ori)/len(w_ori)) # effective sample size
        
        w1_eff = w1 / w_ori.sum() *neff
        nTrue1 = w1_eff.sum()

        w2_eff = w2 / w_ori.sum() *neff
        nTrue2 = w2_eff.sum()

        try:
            a1, b1 = a1 + nTrue1, b1 + (neff[0] - nTrue1)
            a2, b2 = a2 + (nTrue2-nTrue1), b2 + (neff[0] - (nTrue2-nTrue1))
        except NameError:
            prior = 0.01
            a1, b1 = prior + nTrue1, prior + (neff[0] - nTrue1) 
            a2, b2 = prior + (nTrue2-nTrue1), prior + (neff[0] - (nTrue2-nTrue1)) 

    prob_C = get_prob(M, var_inds, var_states)
    prob_C_c = get_prob(M, cvar_inds, cvar_states)

    prob, std, cint = utils.get_rat_dist( prob_C, prob_C_c - prob_C, 1 - M.p.sum(), a1, a2, b1, b2, conf_p )

    cov = std/prob

    return prob, cov, cint

def get_variables_from_cpms(M, variables):

    res = []
    remain = variables[:]

    if isinstance(M, dict):
        M = M.values()

    for Mx in M:
        names = [x.name for x in Mx.variables]
        i = 0
        while i < len(remain):
            if remain[i] in names:
                res.append(Mx.get_variables(remain[i]))
                remain.remove(remain[i])
                #i -= 1
            else:
                i += 1
    assert len(res) == len(variables), f'not all variables found in M: {set(variables).difference([x.name for x in res])}'
    return sorted(res, key=lambda x: variables.index(x.name))

#FIXME: UNUSED
def append(cpm1, cpm2):
    """
    return a list of combined cpm1 and cpm2
    cpm1 should be a list or dict
    cpm2 should be a list or dict
    """

    assert isinstance(cpm1, (list, dict)), 'cpm1 should be a list or dict'
    assert isinstance(cpm2, (list, dict)), 'cpm2 should be a list or dict'

    if isinstance(cpm1, dict):
        cpm1 = list(cpm1.values())

    if isinstance(cpm2, dict):
        cpm2 = list(cpm2.values())

    assert len(cpm1) == len(cpm2), 'Given CPMs have different lengths'


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
        
        cpm_sys2 = prod_cpms(cpms_noC)
        Cs = cpm_sys2.Cs
        q = cpm_sys2.q
        ps = cpm_sys2.ps
        sample_idx = cpm_sys2.sample_idx

        v_idx = cpm_sys2.get_col_ind([v.name for v in sys_vars_ch+sys_vars_par])
        Cs = Cs[:, v_idx]
        
    else:
        Cs = np.empty((0, len(Msys.variables)))
        q, ps, sample_idx = np.empty((0,1)), np.empty((0,1)), np.empty((0,1))

    return Cpm(sys_vars_ch+sys_vars_par, len(sys_vars_ch), C, p, Cs, q, ps, sample_idx)


def get_inf_vars(vars_star, cpms, VE_ord=None):

    """
    INPUT:
    - vars_star: a list of variable names, whose marginal distributions are of interest
    - cpms: a list of CPMs
    - VE_ord (optional): a list of variable names, representing a VE order. The output list of vars_inf is sorted accordingly.
    OUPUT:
    - vars_inf: a list of variable names
    """

    vars_inf = [] # relevant variables for inference
    vars_inf_new = vars_star
    while len(vars_inf_new) > 0:
        v1 = copy.deepcopy( vars_inf_new[0] )
        vars_inf_new.remove(v1)
        vars_inf.append(v1)

        v1_sco = [x.name for x in cpms[v1].variables[cpms[v1].no_child:]] # Scope of v1 (child nodes do not have to be multiplied again)
        for p in v1_sco:
            if p not in vars_inf and p not in vars_inf_new:
                vars_inf_new.append(p)

    if VE_ord is not None:
        def get_ord_inf( x, VE_ord ):
            try:
                return VE_ord.index(x)
            except ValueError:
                return len(VE_ord)

        vars_inf.sort( key=(lambda x: get_ord_inf(x, VE_ord)) )

    return vars_inf

def merge_cpms( cpm1, cpm2 ):

    assert cpm1.variables == cpm2.variables, 'cpm1 and cpm2 must have the same scope'

    M_new = copy.deepcopy( cpm1 )
    C1_list = cpm1.C.tolist()
    for c1, p1 in zip( cpm2.C.tolist(), cpm2.p.tolist() ):
        if c1 in C1_list:
            idx = C1_list.index(c1)
            M_new.p[idx] += p1
        else:
            M_new.C = np.vstack( (M_new.C, c1) )
            M_new.p = np.vstack( (M_new.p, p1) )

    M_new.Cs = np.vstack((cpm1.Cs, cpm2.Cs))
    M_new.q = np.vstack((cpm1.q, cpm2.q))
    M_new.ps = np.vstack((cpm1.ps, cpm2.ps))
    M_new.sample_idx = np.vstack((cpm1.sample_idx, cpm2.sample_idx))

    return M_new


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

    vars_inf = get_inf_vars([sys_name], cpms, ve_order) # inference only ancestors of sys_name
    c_names = [x.name for x in cpms[sys_name].variables[cpms[sys_name].no_child:]]
    ve_names = [x for x in vars_inf if (x in ve_order) and (x not in cond_names) and (x not in c_names)]
    c_jnt_names = [x for x in c_names if x not in ve_order] # if non-empty, a joint distribution (sys, c_jnt) is computed
    c_elm_names = [x for x in c_names if x not in c_jnt_names]

    ve_vars = [varis[v] for v in ve_names if v != sys_name] # other variables

    cpms_inf = {v: cpms[v] for v in ve_names + c_names + c_jnt_names}
    cpms_inf[sys_name] = cpms[sys_name]
    cond_cpms = [cpms[v] for v in cond_names]

    M_cond = prod_cpms( cond_cpms )
    n_crows = len(M_cond.C)

    for i in range(n_crows):
        #m1 = M_cond.get_subset([i])
        m1 = condition(M_cond, M_cond.variables, M_cond.C[i,:])
        m1 = m1[0]
        VE_cpms_m1 = condition( cpms_inf, m1.variables, m1.C[0] )

        for x_name in c_jnt_names:
            VE_cpms_m1[sys_name] = VE_cpms_m1[sys_name].product(VE_cpms_m1[x_name])
            del VE_cpms_m1[x_name]

        VE_cpms_m1_no_sys = {k: v for k,v in VE_cpms_m1.items() if k!=sys_name}
        cpms_comps = variable_elim( VE_cpms_m1_no_sys, ve_vars, prod=False )

        m_m1 = prod_Msys_and_Mcomps( VE_cpms_m1[sys_name], cpms_comps )
        m_m1 = m_m1.sum(c_elm_names)

        m_m1 = m_m1.product(m1)
        m_m1 = m_m1.sum(cond_names)

        if i < 1:
            Msys = copy.deepcopy( m_m1 )
        else:
            Msys = merge_cpms( Msys, m_m1 )

    return Msys


def get_means(M1, v_names):

    """
    Get means of variables in v_names from CPM M1.
    INPUT:
    M1: A CPM
    v_names: a list of names
    OUTPUT:
    means: a list of means (the same order as v_names)
    """
    means = []
    for v in v_names:
        col_idx = [i for i in range(len(M1.variables)) if M1.variables[i].name == v]
        if len(col_idx) != 1:
            raise( v + " appears {:d} times in M1.variables. It must appear exactly ONCE.".format(len(col_idx)) )
        elif col_idx[0] > M1.no_child-1:
            raise( v + " must not be a parent node." )

        m = 0
        for c, p in zip(M1.C, M1.p):
            m += c[col_idx[0]] * p[0]
        means.append(m)

    return means

def get_var_ind(variables, v_names):
    """
    INPUT:
    variables: a list of variables
    v_names: a list of variable names
    OUTPUT:
    v_idxs: a list of column indices of v_names
    """

    v_idxs = []
    for v in v_names:
        idx = [i for (i,k) in enumerate(variables) if k.name == v]

        assert len(idx) == 1, f'Each input variable must appear exactly once in M.variables: {v} appears {len(idx)} times.'

        v_idxs.append(idx[0])

    return v_idxs

