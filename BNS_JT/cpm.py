import numpy as np
import textwrap
import copy

# save away Python sum
_sum_ = sum


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
            index for sample

        Cpm(varibles, numChild, C, p)
    '''
    def __init__(self, **kwargs):

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

        assert len(self.p), 'p must be a numeric vector'
        all(isinstance(y, (float, np.float32, np.float64, int, np.int32, np.int64)) for y in self.p), 'p must be a numeric vector'

        if self.p.ndim == 1:
            self.p.shape = (len(self.p), 1)
        assert len(self.p) == self.C.shape[0], 'p must have the same length with the number of rows in C'

        '''
        elseif ~isempty(M.q) && ~isnumeric(M.q)
            errFlag = 1
            errMess ='Sampling probability vector q must be a numeric vector'
        elseif (~isempty(M.q)&&~isempty(M.C)) && (length(M.q)~=size(M.C,1))
            errFlag = 1
            errMess = 'q must have the same length with the number of rows in C'
        elseif (~isempty(M.sampleIndex)&&~isempty(M.C)) && (length(M.sampleIndex)~=size(M.C,1))
            errFlag = 1
            errMess = 'Sample index array must have the same length with the number of rows in C'
        '''

    def __repr__(self):
        return textwrap.dedent(f'''\
{self.__class__.__name__}(variables={self.variables}, numChild={self.numChild}, C={self.C}, p={self.p}''')

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
    '''
    FIXIT: shuld we return False
    '''
    return [np.where(np.array(B) == x)[0].min() if x in B else False for x in A]


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


def getCpmSubset(M, rowIndex, flagRow=1):
    '''
    M: instance of Cpm
    rowIndex: array like
    getFlag:
    '''

    assert flagRow in (0, 1)

    #if len(M)~= 1: error( 'Given CPM must be a single CPM array' )

    if not flagRow:
        rowIndex = np.setdiff1d(range(M.C.shape[0]), rowIndex)

    if any(M.p):
        pSubset = M.p[rowIndex]
    else:
        pSubset = []

    if any(M.q):
        qSubset = M.q[rowIndex]
    else:
        qSubset = []

    if any(M.sampleIndex):
        sampleIndSubset = M.sampleIndex( rowIndex )
    else:
        sampleIndSubset = []

    return Cpm(variables=M.variables, numChild=M.numChild,
               C=M.C[rowIndex,:], p=pSubset, q=qSubset,
               sampleIndex=sampleIndSubset)


def isCompatibleCpm(M, Mcompare, vInfo, isCompositeStateConsidered=1):
    '''
    M: an instance of Cpm
    Mcompare: another instance of Cpm
    vInfo: list or dictionary of variables
    isCompositeStateConsidered: True (default) or False
    '''
    #if length(M) ~= 1, error( 'Given CPM must be a single array of Cpm' ) 
    #if size(Mcompare.C,1) ~= 1, error( 'Given CPM to compare must include only a single row' ) 

    idx = ismember(Mcompare.variables, M.variables)
    varis = get_value_given_condn(Mcompare.variables, idx)
    states = get_value_given_condn(Mcompare.C[0], idx)
    idx = get_value_given_condn(idx, idx)

    C = M.C[:, idx].copy()
    if any(Mcompare.sampleIndex) and any(M.sampleIndex):
        flag = ( M.sampleIndex == Mcompare.sampleIndex )[:, np.newaxis]
    else:
        flag = np.ones(shape=(C.shape[0], 1), dtype=bool)

    for i, (vari, state) in enumerate(zip(varis, states)):
        C1 = C[:, i][np.newaxis, :]
        if isCompositeStateConsidered:
            B = vInfo[vari].B
        else:
            B = np.eye(np.max(C1))
        x1 = [B[k - 1, :] for k in C1[:, flag.flatten()]][0]
        x2 = B[state - 1,: ]
        check = (np.sum(x1 * x2, axis=1) >0)[:, np.newaxis]
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
        # FIXIT
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

    #FIXIT 
    #newState = unique(newState,'rows')    
    if any(newState):
        B = np.append(B, newState, axis=1)
    return B

"""
def get_varsRemain(M, sumVars, sumFlag):

    print(M.variables)
    tf = np.isin(M.variables[:M.numChild], sumVars)
    if sumFlag:
        varsRemain = M.variables[:M.numChild][~tf]
        varsRemainIdx, _ = np.where(~tf)
    else:
        varsRemain = M.variables[tf]

    return varsRemain
"""

"""
def sum(M, sumVars, sumFlag=1):
    '''
    Sum over CPMs.

    Parameters:
    M: instance of Cpm
    sumVars:
    sumFlag: int
        1 (default) - sum out sumVars, 0 - leave only sumVars
    '''
    assert isinstance(M, Cpm), 'M must be a single CPM'

    if sumFlag and any(set(M.variables[M.numChild:]).intersection(sumVars)):
        print('Parent nodes are NOT summed up')

    varsRemain, varsRemainIdx = compute_varsRemain(M, sumVars, sumFlag)
    numChild = length( varsRemain )

    if ~isempty( M.variables(M.numChild+1:]  )
        varsRemain = [varsRemain M.variables(M.numChild+1:] ]
        varsRemainIdx = [varsRemainIdx (M.numChild+1):length(M.variables)]
    end

    Mloop = Cpm( M.variables( varsRemainIdx ), length( varsRemainIdx ), M.C(:,varsRemainIdx), M.p, M.q, M.sampleIndex )
    if isempty(Mloop.C)
        Msum = Cpm(varsRemain, numChild, zeros(1,0), sum(M.p), [], [] ) 
    else
        Csum = [] psum = [] qsum = [] sampleIndSum = []
        while ~isempty(Mloop.C)
                
            Mcompare = getCpmSubset( Mloop, 1 )
            compatFlag = isComplatibleCpm( Mloop, Mcompare )
         
            Csum = [Csum Mloop.C(1,:)]
            
            if ~isempty(Mloop.p)
                psum = [psum sum(Mloop.p(compatFlag))]
            end
            
            if ~isempty(Mloop.q)
                if ~all( Mloop.q( compatFlag ) == Mloop.q(1) )
                    error( 'Compatible samples cannot have different weights' )
                else
                    qsum = [qsum Mloop.q(1)]
                end
            end
                
            if ~isempty( Mloop.sampleIndex )
                sampleIndSum = [sampleIndSum Mloop.sampleIndex(1)]
            end
            
            Mloop = getCpmSubset( Mloop, find( compatFlag ), 0 )

        end
        
        Msum = Cpm( varsRemain, numChild, Csum, psum, qsum, sampleIndSum )
    end
"""



def get_sign_prod(A, B):
    '''
    A: M2_.p
    B: M1.p[i]
    '''
    assert A.shape[1] == B.shape[0]
    prodSign = np.sign(A * B)
    prodVal = np.exp(np.log(np.abs(A)) + np.log(np.abs(B)))
    return prodSign * prodVal



def product(M1, M2, vInfo):
    '''
    M1: instance of Cpm
    M2: instance of Cpm
    vInfo:

    '''
    assert isinstance(M1, Cpm), f'M1 should be an instance of Cpm'
    assert isinstance(M2, Cpm), f'M2 should be an instance of Cpm'

    check = set(M1.variables[:M1.numChild]).intersection(M2.variables[:M2.numChild])
    assert not bool(check), 'PMFs must not have common child nodes'

    if any(M1.p):
        if not any(M2.p):
            M1.p = np.ones(M1.C.shape[0])
    else:
        if any(M2.p):
            M2.p = np.ones(M2.C.shape[0])

    if any(M1.q):
        if not any(M2.q):
            M2.q = np.ones(M2.C.shape[0])
    else:
        if any(M2.q):
            M1.q = ones(M1.C.shape[0])

    if M1.C.shape[1] > M2.C.shape[1]:
        M1_ = M1
        M1 = M2
        M2 = M1_

    if M1.C.any():
        # FIXIT: defined but not used
        commonVars = list(set(M1.variables).intersection(M2.variables))

        idxVarsM1 = ismember(M1.variables, M2.variables)
        commonVars = get_value_given_condn(M1.variables, idxVarsM1)

        for i in range(M1.C.shape[0]):

            c1_ = get_value_given_condn(M1.C[i, :], idxVarsM1)
            c1_notCommon = M1.C[i, flip(idxVarsM1)]

            if M1.sampleIndex.any():
                sampleInd1 = M1.sampleIndex[i]
            else:
                sampleInd1 = []

            [[M2_], vInfo] = condition([M2], commonVars, c1_, vInfo, sampleInd1)
            _add = np.append(M2_.C, np.tile(c1_notCommon, (M2_.C.shape[0], 1)), axis=1)

            if i:
                Cprod = np.append(Cprod, _add, axis=0)
            else:
                Cprod = _add

            # FIXIT
            #if any(sampleInd1):
                #_add = repmat(sampleInd1, M2_.C.shape[0], 1)
                #sampleIndProd = np.append(sampleIndProd, _add).reshape(M2_.C.shape[0], -1)

            #elif any(M2_.s):
            #    sampleIndProd = np.append(sampleIndPro, M2_.s).reshape(M2_s.shape[0], -1)

            if any(M1.p):
                _prod = get_sign_prod(M2_.p, M1.p[i])

            if i:
                pprod = np.append(pprod, _prod, axis=0)
            else:
                pprod = _prod

            if any(M1.q):
                _prod = get_sign_prod(M2_.q, M1.q[i])

            if i:
                qprod = np.append(qprod, _prod, axis=0)
            else:
                qprod = _prod

        Cprod_vars = np.append(M2.variables, get_value_given_condn(M1.variables, flip(idxVarsM1)))

        newVarsChild = np.append(M1.variables[:M1.numChild], M2.variables[:M2.numChild])
        newVarsChild = np.sort(newVarsChild)

        newVarsParent = np.append(M1.variables[M1.numChild:], M2.variables[M2.numChild:])
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

