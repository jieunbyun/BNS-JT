%{
Ji-Eun Byun (ji-eun.byun@glasgow.ac.uk)
Created: 11 Apr 2023

Generalise Branch-and-Bound (BnB) operation to build CPMs
%}

%% Problem
run demoTransport.m

%% Branch-and-Bound for OD1
odInd = 1;

info.arcPath1 = arcPaths{odInd};
info.arcPaths_time1 = arcPaths_time{odInd};
info.nArc = nArc;

sysFun = @funTrans.bnb_sysFun;
nextStateFun = @funTrans.bnb_nextStateFun;
nextCompFun = @funTrans.bnb_nextCompFun;
maxState = length( info.arcPaths_time1 );

branches = funs.runBnB( sysFun, nextCompFun, nextStateFun, info, maxState*ones(1,nArc) );
bnb2mbn_compsStatesCell = repmat({[2 1]},nArc,1);
[C_od, vars] = getCmat( branches, bnb2mbn_compsStatesCell, [], var_arcs, vars );

% Check if the results are correct
M_bnb = M;
M_bnb( odVarInd ).C = C_od;
M_bnb( odVarInd ).p = ones(size(C_od,1),1);
[M_bnb_VE, vars] = VE( M_bnb, varElimOrder, vars );

odVarInd = var_OD(odInd);
disconnState = size(vars( odVarInd ).B,2); % max basic state
disconnProb = getProb( M_bnb_VE, odVarInd, iDisconnState, vars );
DelayProb = getProb( M_bnb_VE, odVarInd, 1, vars, 0 );

disp( ['Are the disconnected probability the same using BnB?: ' num2str( isequal( disconnProb, ODs_prob_disconn(odInd) ) )] )
disp( ['Are the delay probability the same using BnB?: ' num2str( isequal( DelayProb, ODs_prob_delay(odInd) ) )] )
