import numpy as np

from BNS_JT.cpm import Cpm
from BNS_JT.variable import Variable

def define_system():

    M = {}
    vars_ = {}

    M[1] = Cpm(variables=[1],
            numChild=1,
            C = np.array([1, 2]).T,
            p = np.array([0.9, 0.1]).T)

    M[2] = Cpm(variables=[2, 1],
            numChild=1,
            C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]),
            p = np.array([0.99, 0.01, 0.9, 0.1]).T)

    M[3] = Cpm(variables=[3, 1],
            numChild=1,
            C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]),
            p = np.array([0.95, 0.05, 0.85, 0.15]).T)

    M[4] = Cpm(variables=[4, 1],
            numChild=1,
            C = np.array([[1, 1], [2, 1], [1, 2], [2, 2]]),
            p = np.array([0.99, 0.01, 0.9, 0.1]).T)

    M[5] = Cpm(variables=[5, 2, 3, 4],
            numChild=1,
            C = np.array([[2, 3, 3, 2],
                         [1, 1, 3, 1],
                         [1, 2, 1, 1],
                         [2, 2, 2, 1]]),
            p = np.array([1, 1, 1, 1]).T)

    vars_[1] = Variable(B=np.eye(2),
                        value=['Mild', 'Severe'])

    vars_[2] = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]),
                        value=['Survive', 'Fail'])

    vars_[3] = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]),
                        value=['Survive', 'Fail'])

    vars_[4] = Variable(B=np.array([[1, 0], [0, 1], [1, 1]]),
                        value=['Survive', 'Fail'])

    vars_[5] = Variable(B=np.array([[1, 0], [0, 1]]),
                        value=['Survive', 'Fail'])

    return (M, vars_)


def main():

    M, vars_ = define_system()

    ## isCompatible(C, vars, checkVars, checkStates, vars)
    # # # Note: "vars", "checkVars", "checkStates" must be vectors. # # # 

    """
    # No composite state
    C = M(varInds.x(1)).C
    variables = M(varInds.x(1)).variables
    checkVars = varInds.h
    checkStates = [1]
    vInfo = vars
    compat_test1 = isCompatible(C, variables, checkVars, checkStates, vInfo) 

    # Presence of composite states
    compat_test2 = isCompatible(M(varInds.s).C, M(varInds.s).variables, varInds.x(2:3), [1 1], vars) 

    # No common variables between "vars" and "checkVars", so all rules are compatible
    compat_test3 = isCompatible(M(varInds.h).C, varInds.h, varInds.x(2:3), [1 1], vars) 



    ## getCpmSubset( M, rowIndex, getFlag )

    # The following two operations lead to the same result.
    M_sys_select = getCpmSubset( M(varInds.s), [1], 1 )
    M_sys_delete = getCpmSubset( M(varInds.s), [2 3 4], 0 )


    ## isComplatibleCpm( M, Mcompare, vars )
    # # # Difference from isCompatible: "isCompatibleCpm" compares Cpm objects, while "isCompatible" accepts parts of Cpm information (i.e. variables and event matrices). # # # 
    # # # By the way, there is a typo in the function name: Complatible --> Compatible  # # # 

    compatM_test1 = isComplatibleCpm( M(varInds.x(2)), M_sys_select, vars )
    compatM_test2 = isComplatibleCpm( M(varInds.x(3)), M_sys_select, vars )

    # No common variables between M.variables and Mcompare.variables --> All rules are compatible
    compatM_test3 = isComplatibleCpm( M(varInds.h), M_sys_select, vars )

    ## product(M1, M2, vars)
    # # # Note: "vars" should also be an output variable as it may be updated during the operation this may happen because of composite states. # # #

    # When there is no common variable
    [Mprod_test1, vars] = product( M(varInds.x(1)), M(varInds.x(2)), vars )

    # When there are common variables
    [Mprod_test2, vars] = product( Mprod_test1, M(varInds.s), vars )


    ##  sum(M,sumVars,sumFlag)

    # Parent nodes are not summed up. So nothing happens with the following operation. 
    Msum_test1 = sum( Mprod_test2, varInds.h, 1 )

    # The following two operations lead to the same result.
    Msum_test2 = sum( Mprod_test2, varInds.x(1:2), 1 )
    Msum_test3 = sum( Mprod_test2, varInds.s, 0 )


    ## condition(M, condVars, condStates, vars, <sampleInd>)
    # # # Note: "vars" should also be an output variable as it may be updated during the operation this may happen because of composite states. # # #

    # Conditioning on a child node
    [Mcond_test1, vars] = condition(Mprod_test2, varInds.x(1), 1, vars)

    # Conditioning on a parent node
    [Mcond_test2, vars] = condition(Mprod_test2, varInds.h, 1, vars)

    # Conditioning on multiple nodes
    [Mcond_test3, vars] = condition(Mprod_test2, [varInds.x(1) varInds.h], [1 1], vars)

    # When there are composite states
    [Mcond_test4, vars] = condition(M(varInds.s), [varInds.x(1) varInds.x(2)], [1 1], vars)


    ## getSamplingOrder( cpmArray )
    # # # Note that sampling results change with different random seeds. # # #
    rng(1)

    # The following operation returns errors because there must not be any parnet node with unknown states.
    Mmcs_test_error = mcsProduct( M([varInds.x(1), varInds.s]), 10, vars )

    Mmcs_test1 = mcsProduct( M([varInds.h, varInds.x(1), varInds.x(2)]), 10, vars )
    #{
    Nested functions:
    [sampleOrder, sampleVars, varAdditionOrder] = getSamplingOrder( M )
    leads to
    sampleOrder = [1, 2, 3]
    sampleVars = [1, 2, 3]
    varAdditionOrder = [1, 2, 3]
    [sample, sampleProb] = singleSample( M, sampleOrder, sampleVars, varAdditionOrder, vars, 1 )
    leads to
    sample = [1, 1, 1]
    sampleProb = 0.8464
    , or another possible exmple is
    sample = [2, 1, 1]
    sampleProb = 0.0765
    #}

    Mmcs_test2 = mcsProduct( M, 10, vars )
    #{
    Nested functions:
    [sampleOrder, sampleVars, varAdditionOrder] = getSamplingOrder( M )
    leads to
    sampleOrder = [1, 2, 3, 4, 5]
    sampleVars = [1, 2, 3, 4, 5]
    varAdditionOrder = [1, 2, 3, 4, 5]
    [sample, sampleProb] = singleSample( M, sampleOrder, sampleVars, varAdditionOrder, vars, 1 )
    leads to
    sample = [1, 1, 1, 1, 1]
    sampleProb = 0.8380
    , or another possible exmple is
    sample = [2, 1, 1, 1, 1]
    sampleProb = 0.0688
    #}
    """
    print('***')

if __name__=='__main__':
    main()

