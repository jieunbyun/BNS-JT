function [Msystem, varsSystem, VAR_INTERMEDIATE_NODE] = quantifyMbnOfKnsystem( K, nComponent, VAR_COMPONENT, varIndLocalGlobalDiff, STATE_SURVIVE, STATE_FAIL, COMPOSITE_STATE_COMP )

varLocalInd = 1;
varGlobalInd = varLocalInd + varIndLocalGlobalDiff;
VAR_INTERMEDIATE_NODE( varLocalInd ) = varGlobalInd;
diffBwStateAndNumberOfSurvivingComp = 3; % survival, failure, 0
Msystem( varLocalInd ) = Cpm( varGlobalInd, 1, [diffBwStateAndNumberOfSurvivingComp], [1]);
varsSystem( varLocalInd ) = Variable( eye(diffBwStateAndNumberOfSurvivingComp), {'Survive' 'Fail' '0'}' );


for compInd = 1:nComponent
    varLocalInd = varLocalInd + 1;    
    varGlobalInd = varGlobalInd + 1;
    VAR_INTERMEDIATE_NODE( compInd + 1 ) = varGlobalInd;

    iPreviousNodeLocalInd = varLocalInd - 1;
    iPreviousNodeUniqueState = unique( Msystem( iPreviousNodeLocalInd ).C(:,1) );

    iC = []; ip = [];
    for previousNodeState = iPreviousNodeUniqueState(:)'
        if ismember( previousNodeState, [STATE_SURVIVE STATE_FAIL] )
            iC = [iC; previousNodeState COMPOSITE_STATE_COMP previousNodeState];
            ip = [ip; 1];
        else
            nSurvivingComp = previousNodeState - diffBwStateAndNumberOfSurvivingComp;
            if ( nSurvivingComp + 1) == K
                iC = [iC; STATE_SURVIVE STATE_SURVIVE previousNodeState];
                ip = [ip; 1];
            else
                iC = [iC; previousNodeState+1 STATE_SURVIVE previousNodeState];
                ip = [ip; 1];
            end

            if (K - nSurvivingComp) == (nComponent - compInd + 1)
                iC = [iC; STATE_FAIL STATE_FAIL previousNodeState];
                ip = [ip; 1];
            else
                iC = [iC; previousNodeState STATE_FAIL previousNodeState];
                ip = [ip; 1];
            end
        end
    end

    iScope = [varGlobalInd VAR_COMPONENT(compInd) varGlobalInd-1];
    Msystem( varLocalInd ) = Cpm( iScope, 1, iC, ip );
    varsSystem( varLocalInd ) = Variable( eye( max( iC(:,1) ) ), {'(Number of surviving components) - 3'} );           

end