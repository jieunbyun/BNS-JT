function branches = runBnB( sysFun, nextCompFun, nextStateFun, info, compMaxStates )


nComp = length( compMaxStates );

init_up = compMaxStates(:);
init_down = ones(nComp, 1); % Assume that the lowest state is 1

branches = Branch( init_down, init_up, false );

inCompleteBrInds = find( arrayfun(@(x) ~x.isComplete, branches) );
while ~isempty( inCompleteBrInds )

    iBranchInd = inCompleteBrInds(1);
    iBranch = branches(iBranchInd);
    iDown = iBranch.down;
    iUp = iBranch.up;

    [iDownState, iDownVal, iDownRes] = sysFun( iDown, info );
    [iUpState, iUpVal, iUpRes] = sysFun( iUp, info );

    if iDownState == iUpState
        branches(iBranchInd).isComplete = true;
        branches(iBranchInd).down_state = iDownState;
        branches(iBranchInd).up_state = iUpState;
        branches(iBranchInd).down_val = iDownVal;
        branches(iBranchInd).up_val = iUpVal;

        inCompleteBrInds(1) = [];

    else
        iCandNextComps = find( iUp>iDown );
        iNextComp = nextCompFun( iCandNextComps, iDownRes, iUpRes, info );
        iNextState = nextStateFun( iNextComp, [iDown(iNextComp) iUp(iNextComp)], iDownRes, iUpRes, info );
        
        iBranch_down = iBranch;
        iBranch_down.up( iNextComp ) = iNextState;

        iBranch_up = iBranch;
        iBranch_up.down( iNextComp ) = iNextState+1;

        branches(iBranchInd) = [];
        branches = [branches; iBranch_down; iBranch_up];

        inCompleteBrInds = find( arrayfun(@(x) ~x.isComplete, branches) );
    end
end



    

