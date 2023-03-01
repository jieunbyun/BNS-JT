import numpy as np

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
            self.p = np.array(kwargs['p'])
        else:
            self.p = kwargs['p']
        self.q = None
        self.sampleIndex  = None ## sample index (numbering) vector

        assert len(self.variables), 'variables must be a numeric vector'
        assert all(isinstance(x, (int, np.int32, np.int64)) for x in self.variables), 'variables must be a numeric vector'

        assert isinstance(self.numChild, (int, np.int32, np.int64)), 'numChild must be a numeric scalar'
        assert self.numChild < len(self.variables), 'numChild must be greater than the number of variables'
        '''
        elseif (~isempty(M.numChild)&&~isempty(M.variables)) && (M.numChild>length(M.variables))
            errFlag = 1
            errMess =        '''

        assert isinstance(self.C, np.ndarray), 'Event matrix C must be a numeric matrix'
        assert self.C.dtype in (np.dtype('int64'), np.dtype('int32')), 'Event matrix C must be a numeric matrix'

        assert self.C.shape[1] == len(self.variables), 'C must have the same number of columns with that of variables'

        assert len(self.p), 'p must be a numeric vector'
        all(isinstance(y, (float, np.float32, np.float64, int, np.int32, np.int64)) for y in self.p), 'p must be a numeric vector'

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


"""
def isCompatible(C, variables, checkVars, checkStates, vInfo):
    '''
    C:
    variables:
    checkVars:
    checkStates:
    vInfo:
    '''

    [~, idxInCheckVars] = ismember(checkVars, variables);
    checkVars(~idxInCheckVars) = [];
    checkStates(~idxInCheckVars) = [];
    idxInCheckVars(~idxInCheckVars) = [];

    C1_common = [];
    for vv = 1:length(checkVars)
       C1_common = [C1_common C(:,idxInCheckVars(vv))];


    compatFlag = true( size(C,1),1 );
    for vv = 1:length(checkVars)
        checkVar_v = checkVars(vv);
        checkState_v = checkStates(vv);

        B_v = vInfo(checkVar_v).B;
        C1_v = C1_common(:,vv);

        compatCheck_v = B_v(C1_v(compatFlag),:) .* B_v( checkState_v,: );
        compatFlag(compatFlag) = ( sum(compatCheck_v,2)>0 );

    return CompatFlag
"""


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
