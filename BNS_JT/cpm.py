import numpy as np
import textwrap
import copy
import warnings
from scipy.stats import norm, beta

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
    def no_child(self, value):
        assert isinstance(value, (int, np.int32, np.int64)), 'no_child must be a numeric scalar'
        assert value <= len(self.variables), f'no_child must be less than or equal to the number of variables: {value}, {len(self.variables)}'
        self._no_child = value

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, value):
        if isinstance(value, list):
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
        return textwrap.dedent(f'''\
{self.__class__.__name__}(variables={self.get_names()}, no_child={self.no_child}, C={self.C}, p={self.p}''')

    def get_variables(self, item):

        if isinstance(item, str):
            return [x for x in self.variables if x.name == item][0]
        elif isinstance(item, list):
            return [self.get_variables(y) for y in item]


    def get_names(self):
        return [x.name for x in self.variables]


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


    def get_means(self, names):
        """
        Get means of variables in names
        INPUT:
        names: a list of names
        OUTPUT:
        means: a list of means (the same order as names)
        """
        assert isinstance(names, list), 'names should be a list'
        assert len(set(names))==len(names), f'names has duplicates: {names}'

        idx = [self.get_names().index(x) for x in names]

        return [(self.C[:, i]*self.p[:, 0]).sum() for i in idx]


    def iscompatible(self, M, flag=True):
        """
        Returns a boolean list (n,)

        Parameters
        ----------
        M: instance of Cpm for compatibility check
        flag: True if composite state considered
        """

        assert isinstance(M, Cpm), f'M should be an instance of Cpm'

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


    def get_col_ind(self, names):
        """
        INPUT:
        names: a list of variable names
        OUTPUT:
        idx: a list of column indices of v_names
        """

        assert isinstance(names, list), f'names should be a list'

        assert len(set(names)) == len(names), f'names has duplicates: {names}'

        return [self.get_names().index(x) for x in names]


    def merge(self, M):

        assert isinstance(M, Cpm), f'M should be an instance of Cpm'
        assert self.variables == M.variables, 'must have the same scope and order'

        new_cpm = copy.copy(self)

        cs = self.C.tolist()

        for cx, px in zip(M.C.tolist(), M.p.tolist()):
            try:
                new_cpm.p[cs.index(cx)] += px
            except IndexError:
                new_cpm.C = np.vstack((new_cpm.C, cx))
                new_cpm.p = np.vstack((new_cpm.p, px))

        new_cpm.Cs = np.vstack((self.Cs, M.Cs))
        new_cpm.q = np.vstack((self.q, M.q))
        new_cpm.ps = np.vstack((self.ps, M.ps))
        new_cpm.sample_idx = np.vstack((self.sample_idx, M.sample_idx))

        return new_cpm


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


    def get_prob(self, var_inds, var_states, flag=True):

        assert isinstance(var_inds, list), 'var_inds should be a list'

        if var_inds and isinstance(var_inds[0], str):
            var_inds = self.get_variables(var_inds)

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

        Mc = Cpm(variables=var_inds,
                 no_child=len(var_inds),
                 C=var_states,
                 p=np.empty(shape=(var_states.shape[0], 1)))

        is_compat = self.iscompatible(Mc, flag=True)
        idx = np.where(is_compat)[0]
        Msubset = self.get_subset(idx, flag)

        return Msubset.p.sum()


    def condition(self, cnd_vars, cnd_states):
        """
        Returns a list of instance of Cpm

        Parameters
        ----------
        cnd_vars: a list of variables to be conditioned
        cnd_states: a list of the states to be conditioned
        """

        assert isinstance(cnd_vars, (list, np.ndarray)), 'invalid cnd_vars'

        if isinstance(cnd_vars, np.ndarray):
            cnd_vars = cnd_vars.tolist()

        if cnd_vars and isinstance(cnd_vars[0], str):
            cnd_vars = self.get_variables(cnd_vars)

        assert isinstance(cnd_states, (list, np.ndarray)), 'invalid cnd_vars'

        if isinstance(cnd_states, np.ndarray):
            cnd_states = cnd_states.tolist()

        if cnd_states and isinstance(cnd_states[0], str):
            cnd_states = [x.values.index(y) for x, y in zip(cnd_vars, cnd_states)]

        Mx = copy.deepcopy(self)

        is_cmp = iscompatible(Mx.C, Mx.variables, cnd_vars, cnd_states)

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
                raise(f'{cnd_var} is not defined')
            else:
                C1 = C[:, idx].copy().astype(int)
                check = [B[x].intersection(B[state]) for x in C1]
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
                    pr = Mx.get_prob(Mx.variables, var_states)
                    ps.append(pr*q[0])

                Mx.ps = ps # the conditioned variables' samples are removed.

            elif all(isinstance(r,bool) and r==False for r in res): # conditioned variables are not in the scope.
                Mx.ps = Mx.q.copy()

            else: # conditioned variables are in parent nodes.
                ps = []

                for c in Mx.Cs:
                    var_states = [cnd_states[r] if not isinstance(r,bool) else c[i] for i,r in enumerate(res)]
                    pr = Mx.get_prob(Mx.variables, var_states)
                    ps.append(pr)

                Mx.ps = ps

        return Mx


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

                Mc = M.condition(cnd_vars=com_vars, cnd_states=c1)

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


    def get_prob_bnd(self, var_inds, var_states, flag=True, cvar_inds=None, cvar_states=None, cflag=True):

        prob1 = self.get_prob(var_inds, var_states, flag)
        prob_unk = 1.0 - np.sum(self.p) # Unknown probs
        prob1_bnd = [prob1, prob1 + prob_unk]

        if cvar_inds:
           prob2 =  self.get_prob(cvar_inds, cvar_states, cflag)
           prob2_bnd = [prob2, prob2 + prob_unk]
           prob_bnd = [prob1_bnd[0] / prob2_bnd[1], prob1_bnd[1] / prob2_bnd[0]]

        else:
           prob_bnd = prob1_bnd

        prob_bnd[1] = min(1, prob_bnd[1])

        return prob_bnd


    def get_prob_and_cov(self, var_inds, var_states, method='MLE', flag=True, nsample_repeat=0, conf_p=0.95):

        assert isinstance(nsample_repeat, int), 'nsample_repeat must be a nonnegative integer, representing if samples are repeated (to calculate c.o.v.)'

        prob_C = self.get_prob(var_inds, var_states, flag)

        if not nsample_repeat:
            n_round = 1
            nsamp = len(self.Cs)
        else:
            assert len(self.Cs) % nsample_repeat == 0, 'Given number of samples is not divided by given nsample_repeat'
            n_round = int(len(self.Cs) / nsample_repeat)
            nsamp = nsample_repeat

        prob_Cs = 0
        var = 0
        for i in range(n_round):

            row_range = range(i*nsamp, (i + 1)*nsamp)
            is_cmp = iscompatible(self.Cs[row_range,:], self.variables, var_inds, var_states)

            try:
                w = self.ps[row_range] / self.q[row_range]
            except IndexError:
                w = np.ones_like(self.q) # if ps is empty, assume ps is the same as q

            w_ori = w.copy() # weight before compatibility check
            if flag:
                w[~is_cmp] = 0
            else:
                w[is_cmp] = 0

            if method=='MLE':
                mean = w.sum() / nsamp
                prob_Cs += mean

                if np.allclose(w_ori, w_ori[0], atol=1e-4): # this is MCS
                    var1 = np.square(w_ori[0]) * (1 - mean) * mean / nsamp
                    var += var1[0]
                else:
                    var1 = np.square((w - mean) / nsamp)
                    var += var1.sum()

            elif method=='Bayesian':
                neff = len(w_ori)*w_ori.mean()**2 / (sum(x**2 for x in w_ori)/len(w_ori)) # effective sample size
                w_eff = w / w_ori.sum() *neff
                nTrue = w_eff.sum()

                # to avoid numerical errors
                if np.isnan(nTrue):
                    nTrue = 0.0
                if np.isnan(neff[0]):
                    neff[0] = 0.0

                try:
                    a, b = a + nTrue, b + (neff[0] - nTrue)
                except NameError:
                    prior = 0.01
                    a, b = prior + nTrue, prior + (neff[0] - nTrue)

        if method == 'MLE':
            prob = prob_C + (1 - self.p.sum()) * prob_Cs
            cov = (1 - self.p.sum()) * np.sqrt(var) / prob

            # confidence interval
            z = norm.pdf(1 - (1 - conf_p)*0.5) # for both tails
            prob_Cs_int = prob_Cs + z * np.sqrt(var) * np.array([-1, 1])
            cint = prob_C + (1 - self.p.sum()) * prob_Cs_int

        elif method == 'Bayesian':

            mean = a / (a + b)
            var = a*b / (a+b)**2 / (a+b+1)

            prob = prob_C + (1 - self.p.sum()) * mean
            cov = (1 - self.p.sum()) * np.sqrt(var) / prob

            low = beta.ppf(0.5*(1-conf_p), a, b)
            up = beta.ppf(1 - 0.5*(1-conf_p), a, b)
            cint = prob_C + (1 - self.p.sum()) * np.array([low, up])

        return prob, cov, cint


    def get_prob_and_cov_cond(self, var_inds, var_states, cvar_inds, cvar_states, nsample_repeat=0, conf_p=0.95):
        # Assuming beta distribution (i.e. Bayeisan inference)

        assert isinstance(nsample_repeat, int), 'nsample_repeat must be a nonnegative integer, representing if samples are repeated (to calculate c.o.v.)'

        if not nsample_repeat:
            n_round = 1
            nsamp = len(self.Cs)
        else:
            assert len(self.Cs) % nsample_repeat == 0, 'Given number of samples is not divided by given nsample_repeat'
            n_round = int(len(self.Cs) / nsample_repeat)
            nsamp = nsample_repeat

        for i in range(n_round):
            row_range = range(i*nsamp, (i + 1)*nsamp)

            try:
                w_ori = self.ps[row_range] / self.q[row_range] # weight before compatibility check
            except IndexError:
                w_ori = np.ones_like(self.q) # if ps is empty, assume ps is the same as q

            is_cmp1 = iscompatible(self.Cs[row_range,:], self.variables, var_inds, var_states)
            w1 = w_ori.copy()
            w1[~is_cmp1] = 0

            is_cmp2 = iscompatible(self.Cs[row_range,:], self.variables, cvar_inds, cvar_states)
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

        prob_C = self.get_prob(var_inds, var_states)
        prob_C_c = self.get_prob(cvar_inds, cvar_states)

        prob, std, cint = utils.get_rat_dist( prob_C, prob_C_c - prob_C, 1 - self.p.sum(), a1, a2, b1, b2, conf_p )

        cov = std/prob

        return prob, cov, cint


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


def product(cpms):
    """
    return an instance of Cpm

    cpms: a list or dict of instances of Cpm
    """
    assert isinstance(cpms, (list,  dict)), 'cpms should be a list or dict'

    if isinstance(cpms, dict):
        cpms = list(cpms.values())

    assert cpms, f'{cpms} is empty list'

    prod = cpms[0]
    for cx in cpms[1:]:
        prod = prod.product(cx)

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

