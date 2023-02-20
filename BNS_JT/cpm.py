import numpy as np

class Cpm(object):
    '''
    class to define conditional probability matrix (cf., CPT)
        Cpm(varibles, numChild, C, p)
        variables:
        numChild:
        C:
        p:
        q:
        sampleIndex:
    '''
    def __init__(self, **kwargs):

        self.variables = kwargs['variables']
        self.numChild = kwargs['numChild']
        # event matrix
        self.C = kwargs['C']
        # probability vector
        if isinstance(kwargs['p'], list):
            self.p = np.array(kwargs['p'])
        else:
            self.p = kwargs['p']
        # sampling weight vector
        self.q = None
        self.sampleIndex  = None #% sample index (numbering) vector

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
        newVarsParent = [M1.variables(M1.numChild+1:end) M2.variables(M2.numChild+1:end)]
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
