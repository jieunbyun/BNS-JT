function [branches_surv, branches_fail, states, times] = doBranch( arcPaths_cell, pathTimes_array, unfinished, infTime )

nPath = length(arcPaths_cell);

% Initialise using the first (shortest) route
path1 = arcPaths_cell{1};
time1 = pathTimes_array(1);
state1 = 1; % Path in use

[branches_surv, branches_fail, states, times] = funTrans.doBranch1( path1, state1, time1, unfinished );

% % Branching begins
while any(times==unfinished)
    iBranchInd = find( times==unfinished, 1 );
    iBranch_surv = branches_surv{iBranchInd};
    iBranch_fail = branches_fail{iBranchInd};

    iSurvPathInd = cellfun( @(x) all( ismember(x, iBranch_surv) ), arcPaths_cell, 'UniformOutput',true );
    iSurvPathInd = find(iSurvPathInd, 1);

    if ~isempty(iSurvPathInd)
        times(iBranchInd) = pathTimes_array(iSurvPathInd);
        states(iBranchInd) = iSurvPathInd;

    else
        iFailPathInds = cellfun( @(x) any( ismember(iBranch_fail, x) ), arcPaths_cell, 'UniformOutput',true );

        if all(iFailPathInds)
            times(iBranchInd) = infTime;
            states(iBranchInd) = nPath+1;

        else
            iNextPathInd = find(~iFailPathInds, 1); % Next possible shortest path
            iNextPath = arcPaths_cell{iNextPathInd};
            iTime = pathTimes_array(iNextPathInd);

            iArcsToBranch = setdiff( iNextPath, [iBranch_surv(:)', iBranch_fail(:)'] );
            [iBranches_surv, iBranches_fail, iStates, iTimes] = funTrans.doBranch1( iArcsToBranch, iNextPathInd, iTime, unfinished );

            iBranches_surv = cellfun( @(x) sort( [x(:)' iBranch_surv(:)'] ), iBranches_surv, 'UniformOutput', false );
            iBranches_fail = cellfun( @(x) sort( [x(:)' iBranch_fail(:)'] ), iBranches_fail, 'UniformOutput', false );


            branches_surv(iBranchInd) = [];
            branches_fail(iBranchInd) = [];
            states(iBranchInd) = [];
            times(iBranchInd) = [];

            branches_surv = [branches_surv; iBranches_surv];
            branches_fail = [branches_fail; iBranches_fail];
            states = [states; iStates];
            times = [times; iTimes];
        end
    end

end