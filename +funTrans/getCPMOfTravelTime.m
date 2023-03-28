function [C, p, stateVals] = getCPMOfTravelTime( arcPaths_cell, pathTimes_array, nArc, arc_surv, arc_fail, arc_either )

if any(diff(pathTimes_array) < 0)
    [pathTimes_array, paths_sortInd] = sort(pathTimes_array, 'ascend');
    arcPaths_cell = arcPaths_cell(paths_sortInd);
end
arcPaths_cell = cellfun(@sort, arcPaths_cell, 'UniformOutput', false);

infTime = 1e2; % When no paths are available
unfinished = 0;

% Do branching first
[branches_surv, branches_fail, states, times] = funTrans.doBranch( arcPaths_cell, pathTimes_array, unfinished, infTime );

% Build a CPM
nBranch = length(branches_surv);
C = zeros(nBranch, nArc+1 );
p = ones(nBranch, 1);
for iBranchInd = 1:nBranch
    iSurvArcs = branches_surv{iBranchInd};
    iFailArcs = branches_fail{iBranchInd};
    iState = states(iBranchInd);

    iC = [iState arc_either*ones(1, nArc)];
    iC(iSurvArcs+1) = arc_surv;
    iC(iFailArcs+1) = arc_fail;

    C(iBranchInd,:) = iC;
end

stateVals = unique(times); % Note: This function sorts values in ascending order.
