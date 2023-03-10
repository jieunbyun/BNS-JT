import numpy as np
import textwrap
import copy



class Cpm(object):
    '''
    class to define conditional probability matrix (cf., CPT)

        Parameters:
        variables: array_like
            index for variables
        numChild: int
            number of child nodes
        C: array_like
            event matrix
        p: array_like
            probability vector
        q: array_like
            sampling weight vector for continuous r.v.
        sampleIndex: array_like
            sample index vector

        Cpm(varibles, numChild, C, p, q, sampleIndex)
    '''
    def __init__(self, **kwargs):

        if isinstance(kwargs['variables'], list):
            self.variables = np.array(kwargs['variables'])
        else:
            self.variables = kwargs['variables']

        self.numChild = kwargs['numChild']
        self.C = kwargs['C']
        if isinstance(kwargs['p'], list):
            self.p = np.array(kwargs['p'])[:, np.newaxis]
        else:
            self.p = kwargs['p']
        self.q = kwargs.get('q', np.array([]))
        self.sampleIndex  = kwargs.get('sampleIndex', np.array([])) ## sample index (numbering) vector

        assert len(self.variables), 'variables must be a numeric vector'
        assert all(isinstance(x, (int, np.int32, np.int64)) for x in self.variables), 'variables must be a numeric vector'

        assert isinstance(self.numChild, (int, np.int32, np.int64)), 'numChild must be a numeric scalar'
        assert self.numChild <= len(self.variables), 'numChild must be less than or equal to the number of variables'

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
        assert len(self.p) == self.C.shape[0], 'p must have the same length with the number of rows in C'

        if any(self.q):
            assert len(self.q) == self.C.shape[0], 'q must have the same length with the number of rows in C'

        if any(self.sampleIndex):
            assert len(self.sampleIndex) == self.C.shape[0], 'sampleIndex must have the same length with the number of rows in C'

        '''
        elseif ~isempty(M.q) && ~isnumeric(M.q)
            errFlag = 1
            errMess ='Sampling probability vector q must be a numeric vector'
        elseif (~isempty(M.q)&&~isempty(M.C)) && (length(M.q)~=size(M.C,1))
        elseif (~isempty(M.sampleIndex)&&~isempty(M.C)) && (length(M.sampleIndex)~=size(M.C,1))
        '''

    def __repr__(self):
        return textwrap.dedent(f'''\
{self.__class__.__name__}(variables={self.variables}, numChild={self.numChild}, C={self.C}, p={self.p}''')

    def getCpmSubset(self, rowIndex, flagRow=1):
        '''
        M: instance of Cpm
        rowIndex: array like
        getFlag:
        '''

        assert flagRow in (0, 1)

        #if len(M)~= 1: error( 'Given CPM must be a single CPM array' )

        if not flagRow:
            rowIndex = np.setdiff1d(range(self.C.shape[0]), rowIndex)

        if any(self.p):
            pSubset = self.p[rowIndex]
        else:
            pSubset = np.array([])

        if any(self.q):
            qSubset = self.q[rowIndex]
        else:
            qSubset = np.array([])

        if any(self.sampleIndex):
            sampleIndSubset = self.sampleIndex( rowIndex )
        else:
            sampleIndSubset = np.array([])

        return Cpm(variables=self.variables,
                   numChild=self.numChild,
                   C=self.C[rowIndex,:],
                   p=pSubset,
                   q=qSubset,
                   sampleIndex=sampleIndSubset)

    def isCompatible(self, Mc, vInfo=[]):
        '''
        M: an instance of Cpm
        Mc: another instance of Cpm
        vInfo: list or dictionary of variables
        '''
        #if length(M) ~= 1, error( 'Given CPM must be a single array of Cpm' ) 
        #if size(Mc.C,1) ~= 1, error( 'Given CPM to compare must include only a single row' ) 

        idx = ismember(Mc.variables, self.variables)
        varis = get_value_given_condn(Mc.variables, idx)
        states = get_value_given_condn(Mc.C[0], idx)
        idx = get_value_given_condn(idx, idx)

        C = self.C[:, idx].copy()
        if any(Mc.sampleIndex) and any(self.sampleIndex):
            flag = ( self.sampleIndex == Mc.sampleIndex )[:, np.newaxis]
        else:
            flag = np.ones(shape=(C.shape[0], 1), dtype=bool)

        for i, (vari, state) in enumerate(zip(varis, states)):
            C1 = C[:, i][np.newaxis, :]
            if any(vInfo):
                B = vInfo[vari].B
            else:
                B = np.eye(np.max(C1))
            x1 = [B[k - 1, :] for k in C1[:, flag.flatten()]][0]
            x2 = B[state - 1,: ]
            check = (np.sum(x1 * x2, axis=1) >0)[:, np.newaxis]
            flag[np.where(flag > 0)[0][:len(check)]] = check

        return flag

    def sum(self, varis, flag=1):
        '''
        Sum over CPMs.
        Parameters:
        varis: variables
        flag: int
            1 (default) - sum out varis, 0 - leave only varis
        '''

        if flag and any(set(self.variables[self.numChild:]).intersection(varis)):
            print('Parent nodes are NOT summed up')

        if flag:
            varsRemain, varsRemainIdx = setdiff(self.variables[:self.numChild], varis)
        else:
            # FIXME
            varsRemainIdx = ismember(varis, self.variables[:self.numChild])
            varsRemainIdx = get_value_given_condn(varsRemainIdx, varsRemainIdx)
            varsRemainIdx = np.sort(varsRemainIdx)
            varsRemain = self.variables[varsRemainIdx]

        numChild = len(varsRemain)

        if any(self.variables[self.numChild:]):
            varsRemain = np.append(varsRemain, self.variables[self.numChild:])
            varsRemainIdx = np.append(varsRemainIdx, range(self.numChild, len(self.variables)))

        Mloop = Cpm(variables=self.variables[varsRemainIdx],
                    C=self.C[:, varsRemainIdx],
                    p=self.p,
                    q=self.q,
                    sampleIndex=self.sampleIndex,
                    numChild=len(varsRemainIdx))

        while Mloop.C.any():

            Mcompare = Mloop.getCpmSubset([0]) # need to change to 0 
            _flag = Mloop.isCompatible(Mcompare)

            val = Mloop.C[0, :][np.newaxis, :]
            try:
                Csum = np.append(Csum, val, axis=0)
            except NameError:
                Csum = val

            if any(Mloop.p):
                pval = np.array([np.sum(Mloop.p[_flag])])[:, np.newaxis]
                try:
                    psum = np.append(psum, pval, axis=0)
                except NameError:
                    psum = pval

            if any(Mloop.q):
                qval = Mloop.q[0]
                try:
                    qsum = np.append(qsum, qval, axis=0)
                except NameError:
                    qsum = qval

            if any(Mloop.sampleIndex):
                val = Mloop.sampleIndex[0]
                try:
                    samplesum = np.append(sampleIndsum, val, axis=0)
                except NameError:
                    sampleIndsum = val

            Mloop = Mloop.getCpmSubset(np.where(_flag)[0], flagRow=0)

        Ms = Cpm(variables=varsRemain, numChild=numChild, C=Csum, p=psum)

        try:
            Ms.q = qsum
        except NameError:
            pass

        try:
            Ms.sampleIndex = sampleIndsum
        except NameError:
            pass

        return Ms

    def product(self, M2, vInfo):
        '''
        M1: instance of Cpm
        M2: instance of Cpm
        vInfo:

        '''
        assert isinstance(M2, Cpm), f'M2 should be an instance of Cpm'

        if self.C.shape[1] > M2.C.shape[1]:
            return M2.product(self, vInfo)

        check = set(self.variables[:self.numChild]).intersection(M2.variables[:M2.numChild])
        assert not bool(check), 'PMFs must not have common child nodes'

        if any(self.p):
            if not any(M2.p):
                self.p = np.ones(self.C.shape[0])
        else:
            if any(M2.p):
                M2.p = np.ones(M2.C.shape[0])

        if any(self.q):
            if not any(M2.q):
                M2.q = np.ones(M2.C.shape[0])
        else:
            if any(M2.q):
                self.q = ones(self.C.shape[0])

        if self.C.any():
            # FIXME: defined but not used
            commonVars = list(set(self.variables).intersection(M2.variables))

            idxVarsM1 = ismember(self.variables, M2.variables)
            commonVars = get_value_given_condn(self.variables, idxVarsM1)

            for i in range(self.C.shape[0]):

                c1_ = get_value_given_condn(self.C[i, :], idxVarsM1)
                c1_notCommon = self.C[i, flip(idxVarsM1)]

                if self.sampleIndex.any():
                    sampleInd1 = self.sampleIndex[i]
                else:
                    sampleInd1 = []

                [[M2_], vInfo] = condition([M2], commonVars, c1_, vInfo, sampleInd1)
                _add = np.append(M2_.C, np.tile(c1_notCommon, (M2_.C.shape[0], 1)), axis=1)

                if i:
                    Cprod = np.append(Cprod, _add, axis=0)
                else:
                    Cprod = _add

                # FIXME
                #if any(sampleInd1):
                    #_add = repmat(sampleInd1, M2_.C.shape[0], 1)
                    #sampleIndProd = np.append(sampleIndProd, _add).reshape(M2_.C.shape[0], -1)

                #elif any(M2_.s):
                #    sampleIndProd = np.append(sampleIndPro, M2_.s).reshape(M2_s.shape[0], -1)

                if any(self.p):
                    _prod = get_sign_prod(M2_.p, self.p[i])

                if i:
                    pprod = np.append(pprod, _prod, axis=0)
                else:
                    pprod = _prod

                if any(self.q):
                    _prod = get_sign_prod(M2_.q, self.q[i])

                if i:
                    qprod = np.append(qprod, _prod, axis=0)
                else:
                    qprod = _prod

            Cprod_vars = np.append(M2.variables, get_value_given_condn(self.variables, flip(idxVarsM1)))

            newVarsChild = np.append(self.variables[:self.numChild], M2.variables[:M2.numChild])
            newVarsChild = np.sort(newVarsChild)

            newVarsParent = np.append(self.variables[self.numChild:], M2.variables[M2.numChild:])
            newVarsParent = list(set(newVarsParent).difference(newVarsChild))
            newVars = np.append(newVarsChild, newVarsParent, axis=0)

            idxVars = ismember(newVars, Cprod_vars)

            Mprod = Cpm(variables=newVars,
                        numChild = len(newVarsChild),
                        C = Cprod[:, idxVars],
                        p = pprod)

            if any(qprod):
                Mprod.q = qprod

            Mprod.sort()

        else:
            Mprod = M2

        return  Mprod, vInfo


    def sort(self):

        if any(self.sampleIndex):
            rowIdx = argsort(self.sampleIndex)
        else:
            rowIdx = argsort(list(map(tuple, self.C[:, ::-1])))

        self.C = self.C[rowIdx, :]

        if self.p.any():
            self.p = self.p[rowIdx]

        if self.q.any():
            self.q = self.q[rowIdx]

        if self.sampleIndex.any():
            self.sampleIndex = self.sampleIndex[rowIdx]


def argsort(seq):

    #http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
    return sorted(range(len(seq)), key=seq.__getitem__)


def ismember(A, B):
    #FIXME: shuld we return False
    return [np.where(np.array(B) == x)[0].min() if x in B else False for x in A]


def setdiff(A, B):
    '''
    matlab setdiff equivalent
    '''
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


def isCompatible(C, variables, varis, states, vInfo):
    '''
    C: np.ndarray
    variables: array_like
    vars: array_like
    sates: array_like
    vInfo: can be dict or list, collection of instance of Variable
    '''

    idx = ismember(varis, variables)
    varis = get_value_given_condn(varis, idx)
    states = get_value_given_condn(states, idx)
    idx = get_value_given_condn(idx, idx)

    C = C[:, idx].copy()
    flag = np.ones(shape=(C.shape[0], 1), dtype=bool)
    for i, (vari, state) in enumerate(zip(varis, states)):

        B = vInfo[vari].B
        C1 = C[:, i][np.newaxis, :]
        x1 = [B[k - 1, :] for k in C1[:, flag.flatten()]][0]
        x2 = B[state - 1, :]
        check = (np.sum(x1 * x2, axis=1) > 0)[:, np.newaxis]

        flag[np.where(flag > 0)[0][:len(check)]] = check

    return flag


def flip(idx):
    '''
    boolean flipped
    Any int including 0 will be flipped False
    '''
    return [True if x is False else False for x in idx]


def condition(M, varis, states, vars_, sampleInd=[]):
    '''
    M: a list or dictionary of instances of Cpm
    varis: array
    states: array
    vars_:
    sampleInd:
    '''
    if isinstance(varis, list):
        varis = np.array(varis)

    if isinstance(states, list):
        states = np.array(states)

    Mc = copy.deepcopy(M)
    for Mx in Mc:
        flag = isCompatible(Mx.C, Mx.variables, varis, states, vars_)
        # FIXME
        #if any(sampleInd) and any(Mx.sampleIndex):
        #    flag = flag & ( M.sampleIndex == sampleInd )
        C = Mx.C[flag.flatten(),:].copy()
        idxInCs = np.array(ismember(varis, Mx.variables))
        idxInvaris = ismember(Mx.variables, varis)

        Ccond = np.zeros_like(C)
        not_idxInvaris = flip(idxInvaris)
        Ccond[:, not_idxInvaris] = get_value_given_condn(C, not_idxInvaris)

        varis = varis[idxInCs >= 0].copy()
        states = states[idxInCs >= 0].copy()
        idxInCs = idxInCs[idxInCs >= 0].copy()

        for vari, state, idxInC in zip(varis, states, idxInCs):

            B = vars_[vari].B.copy()

            if B.any():
                C1 = C[:, idxInC].copy() - 1
                check = B[C1, :] * B[state - 1,:]
                vars_[vari].B = addNewStates(check, B)
                Ccond[:, idxInC] = [x + 1 for x in ismember(check, B)]

        Mx.C = Ccond.copy()
        if any(Mx.p):
            Mx.p = Mx.p[flag][:, np.newaxis]
        if any(Mx.q):
            Mx.q = Mx.q[flag][:, np.newaxis]
        if any(Mx.sampleIndex):
            Mx.sampleIndex = Mx.sampleIndex[flag][:, np.newaxis]

    return (Mc, vars_)


def addNewStates(states, B):
    check = flip(ismember(states, B))
    newState = states[check,:]

    #FIXME 
    #newState = unique(newState,'rows')    
    if any(newState):
        B = np.append(B, newState, axis=1)
    return B


def get_sign_prod(A, B):
    '''
    A: M2_.p
    B: M1.p[i]
    '''
    assert A.shape[1] == B.shape[0]
    prodSign = np.sign(A * B)
    prodVal = np.exp(np.log(np.abs(A)) + np.log(np.abs(B)))
    return prodSign * prodVal


