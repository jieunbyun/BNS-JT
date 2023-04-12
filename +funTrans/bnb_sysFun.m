function [state, val, result] = bnb_sysFun( compStates, info )
%{
Input:
"info": a structure that contains any information about a given problem
Output:
"state": a positve integer indicating a computed system state
"val": any value that shows what the "state" means; if unnecessary, can be left empty
"result": a structure with any field that is required for the functions "nextComp" and "nextState"
%}

arcPath1 = info.arcPath1;
arcPaths_time1 = info.arcPaths_time1;

% Ensure shorter paths to be considered first
[arcPaths_time1, pathSortInd] = sort(arcPaths_time1 ,'ascend'); 
arcPath1 = arcPath1(pathSortInd);

% Find the shortest path possible
survComps = find(compStates==2);
isPathConn = cellfun( @(x) all(ismember(x, survComps)), arcPath1 );
isPathConn1 = find(isPathConn, 1 ); 
isPathConn1_origin = pathSortInd(isPathConn1);

% Result
if ~isempty(isPathConn1)
    state = isPathConn1_origin;
    val = arcPaths_time1(isPathConn1);
    result.path = arcPath1{isPathConn1};
else
    state = length(arcPaths_time1) + 1; % there is no path available
    val = inf;
    result.path = [];
end