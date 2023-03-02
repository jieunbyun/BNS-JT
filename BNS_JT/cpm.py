import numpy as np
import textwrap

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
        self.q = None
        self.sampleIndex  = None ## sample index (numbering) vector

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



def ismember(A, B):
    '''
    FIXIT: shuld we return False
    '''
    return [np.where(np.array(B) == x)[0].min() if x in B else False for x in A]


def get_value_given_condn(A, condn):
    assert len(A) == len(condn), f'len of {A} is not equal to len of {condn}'
    return [x for (i, x) in zip(condn, A) if i is not False]

def isCompatible(C, variables, checkVars, checkStates, vInfo):
    '''
    C: np.ndarray
    variables: array_like
    checkVars: array_like
    checkStates: array_like
    vInfo: can be dict or list, collection of instance of Variable
    '''

    idxInCheckVars = ismember(checkVars, variables)
    checkVars = get_value_given_condn(checkVars, idxInCheckVars)
    checkStates = get_value_given_condn(checkStates, idxInCheckVars)
    idxInCheckVars = get_value_given_condn(idxInCheckVars, idxInCheckVars)

    C1_common = C[:, idxInCheckVars]
    compatFlag = np.ones(shape=(C.shape[0], 1), dtype=bool)
    for i, (checkVar, checkState) in enumerate(zip(checkVars, checkStates)):

        B = vInfo[checkVar].B
        C1 = C1_common[:, i][np.newaxis, :]

        x1 = [B[k-1, :] for k in C1][0]
        x2 = B[checkState-1, :]
        compatCheck = (np.sum(x1 * x2, axis=1) > 0)[:, np.newaxis]
        compatFlag = np.logical_and(compatFlag, compatCheck)

    return compatFlag



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

    





"""
def product(M1, M2, vInfo):
    '''
    M1: instance of Cpm
    M2: instance of Cpm
    vInfo:

    '''

    assert bool(set(M1.variables[:M1.numChild]).intersection(
        M2.variables[:M2.numChild]), 'PMFs must not have common child nodes'

    if any(M1.p):
        if any(M2.p):
            M1.p = np.ones(M1.C.shape[0])
    else:
       if not any(M2.p):
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

    if any(M1.C):

        commonVars=set(M1.variables).intersection(M2.variables)

        flagCommonVarsInM1 = ismember(M1.variables,M2.variables)
        commonVars = M1.variables(flagCommonVarsInM1)

        C1 = M1.C
        p1 = M1.p
        q1 = M1.q

        sampleInd1 = M1.sampleIndex

        Cproduct = []
        pproduct = []
        qproduct = []
        sampleIndProduct = []

        for rr in 1:size(C1,1):
            c1_r = C1(rr, flagCommonVarsInM1)
            c1_notCommon_r = C1[rr, ~flagCommonVarsInM1]

            if any(sampleInd1):
                sampleInd1_r = sampleInd1(rr)
            else:
                sampleInd1_r = []

            [M2_r,vInfo] = condition(M2, commonVars, c1_r, vInfo, sampleInd1_r)
            Cproduct = [Cproduct M2_r.C repmat(c1_notCommon_r, size(M2_r.C,1), 1)]

            if any(sampleInd1_r):
                sampleIndProduct = [sampleIndProduct repmat(sampleInd1_r, size(M2_r.C,1), 1)]
            elseif ~isempty( M2_r.sampleIndex )
                sampleIndProduct = [sampleIndProduct M2_r.sampleIndex]

            if any(p1):
                pproductSign_r = sign( M2_r.p * p1(rr) )
                pproductVal_r = exp( log( abs(M2_r.p) )+log( abs(p1(rr)) ) )
                pproduct = [pproduct pproductSign_r .* pproductVal_r]
            end

            if any(q1):
                qproductSign_r = sign( M2_r.q * q1(rr) )
                qproductVal_r = exp( log( abs(M2_r.q) )+log( abs(q1(rr)) ) )
                qproduct = [qproduct qproductSign_r .* qproductVal_r]
            end

        end

        Cproduct_vars = [M2.variables M1.variables(~flagCommonVarsInM1)]

        newVarsChild = [M1.variables(1:M1.numChild) M2.variables(1:M2.numChild)]
        newVarsChild = sort(newVarsChild)
        newVarsParent = [M1.variables(M1.numChild+1:]  M2.variables(M2.numChild+1:] ]
        newVarsParent = setdiff(newVarsParent,newVarsChild)
        newVars = [newVarsChild newVarsParent]

        [~,idxVars] = ismember(newVars,Cproduct_vars)
        Cproduct = Cproduct(:,idxVars)

        Mproduct = Cpm
        Mproduct.variables = newVars
        Mproduct.numChild = length(newVarsChild)
        Mproduct.C = Cproduct Mproduct.p = pproduct Mproduct.q = qproduct
        Mproduct.sampleIndex = sampleIndProduct

        Mproduct = sort(Mproduct)
    else
        Mproduct = M2
    end
"""
