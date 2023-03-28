%{
Ji-Eun Byun (ji-eun.byun@glasgow.ac.uk)
Created: 28 Mar 2023

Test functions for branch and bound used in demoTransport.m
%}

%% Data
% Network
nodeCoords_km = [-2 3; -2 -3; 2 -2; 1 1; 0 0];
arcs = [1 2; 1 5; 2 5; 3 4; 3 5; 4 5];
major = 1; urban = 2; bridge = 3;
arcsType = [major; major; major; urban; bridge; bridge];
arcs_Vavg_kmh = [40; 40; 40; 30; 30; 20];
ODs = [5 1; 5 2; 5 3; 5 4];
arc_surv = 1; arc_fail=2; arc_either=3; % Arcs' states index

arcLens_km = funTrans.getArcsLength( arcs, nodeCoords_km );
arcTimes_h = arcLens_km ./ arcs_Vavg_kmh;
G = graph(arcs(:,1), arcs(:,2), arcTimes_h);
[arcPaths, arcPaths_time] = funTrans.getAllPathsAndTimes( ODs, G, arcTimes_h );

%% Test functions
% funTrans.doBranch1.m
unfinished = 0; % An arbitrary constant to mark that a branch needs to be further branched out
odInd = 1;
pathIndToUse = 1;
path1 = arcPaths{odInd, pathIndToUse};
time1 = arcPaths_time{odInd}(pathIndToUse);
[branches_surv1, branches_fail1, states1, times1] = funTrans.doBranch1( path1, pathIndToUse, time1, unfinished );

% funTrans.doBranch.m
infTime = 1e2; % An arbitrarily large number to denote that an OD pair is disconnected 
odInd = 1;
paths1 = arcPaths{odInd};
pathTimes1 = arcPaths_time{odInd};
[branches_surv, branches_fail, states, times] = funTrans.doBranch( paths1, pathTimes1, unfinished, infTime );