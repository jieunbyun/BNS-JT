%{
Ji-Eun Byun (ji-eun.byun@glasgow.ac.uk)
Created: 27 Mar 2023

A small, hypothetical bridge system
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


% GM
GM_obs = [30 20 10 2 0.9 0.6]; % For the moment, we assume that ground motions are observed. Later, hazard nodes will be added.

% Fragility curves -- From HAZUS-EQ model (roads are regarded as disconnected when being extensively or completely damaged)
IMs = [60 24 1.1];
IMs_std = [0.7 0.7 3.9];


%% BN set up
varInd = 0;
M = Cpm;
vars = Variable;

% Arcs (components): P(X_i | GM = GM_ob ), i = 1 .. N (= nArc)
nArc = size(arcs,1);
var_arcs = zeros(1,nArc);
for iArcInd = 1:nArc
    varInd = varInd+1;

    iRoadType = arcsType(iArcInd);
    iFrag_mean = IMs(iRoadType); iFrag_std = IMs_std(iRoadType);
    iGM_obs = GM_obs(iArcInd);
    iFailProb = funTrans.normcdf( 1/iFrag_std * log( iGM_obs/iFrag_mean ));

    M(varInd,1) = Cpm( varInd, 1, [arc_surv; arc_fail], [1-iFailProb; iFailProb] );
    vars(varInd,1) = Variable( [eye(2); 1 1], {'Surv'; 'Fail'} );
    var_arcs(iArcInd) = varInd;
end

% Travel times (systems): P( OD_j | X_1 ... X_N ), j = 1 .. M (= nOD)
nOD = size(ODs,1);
var_OD = zeros(1,nOD);
for iOdInd = 1:nOD
    varInd = varInd+1;

    iPaths = arcPaths{iOdInd};
    iTimes = arcPaths_time{iOdInd};

    [iC, iP, iVals] = funTrans.getCPMOfTravelTime( iPaths, iTimes, nArc, arc_surv, arc_fail, arc_either );
    M(varInd,1) = Cpm( [varInd var_arcs], 1, iC, iP );
    vars(varInd,1) = Variable( eye(length(iVals)), iVals );
    var_OD(iOdInd) = varInd;
end


%% Inference - by variable elimination (would not work for large-scale systems)
% Probability of delay and disconnection
varElimOrder = var_arcs;
M_VE = M; % Becomes P(OD_1, ..., OD_M) since X_1, ..., X_N are eliminated
for iVarInd_ = 1:length(varElimOrder)
    iVarInd = varElimOrder(iVarInd_);
    iIsVarInScope = isXinScope( iVarInd, M_VE );
    
    [iMMult, vars] = multCPMs(M_VE(iIsVarInScope), vars);
    iMMult = sum( iMMult, iVarInd );

    M_VE(iIsVarInScope) = [];
    M_VE = [iMMult; M_VE];
end


% Retrieve example results
ODs_prob_disconn = zeros(1,nOD); % P( OD_j = 3 ), j = 1, ..., M, where State 3 indicates disconnection
ODs_prob_delay = zeros(1,nOD); % P( (OD_j = 2) U (OD_j = 2) ), j = 1, ..., M, where State 2 indicates the use of the second shortest path (or equivalently, P( (OD_j = 1)^c ), where State 1 indicates the use of the shortest path)
for iODInd = 1:nOD
    iVarInd = var_OD(iODInd);

    % Prob. of disconnection
    iDisconnState = size(vars(iVarInd).B,1);
    iM_VE = condition( M_VE, iVarInd, iDisconnState, vars );

    iDisconnProb = sum( iM_VE.p );
    ODs_prob_disconn(iODInd) = iDisconnProb;

    % Prob. of delay
    iVarLocInC = find(M_VE.variables == iVarInd);
    iRowsIndToKeep = find( M_VE.C(:, iVarLocInC)> 1 );
    iM_VE = getCpmSubset( M_VE, iRowsIndToKeep );

    iDelayProb = sum( iM_VE.p );
    ODs_prob_delay(iODInd) = iDelayProb;
end

figure;
bar( [ODs_prob_disconn(:) ODs_prob_delay(:)] )

grid on
xlabel( 'OD pair' )
ylabel( 'Probability' )
legend( {'Disconnection' 'Delay'}, 'location', 'northwest' )


% City 1 and 2 experienced a disruption in getting resources, City 3 was okay and 4 is unknown. Probability of damage of roads?
% % A composite state needs be created for City 1 and City 2
for iODInd = 1:nOD
    vars( var_OD(iODInd) ).B = [vars( var_OD(iODInd) ).B; 0 1 1];
end

% % Add observation nodes P( O_j | OD_j ), j = 1, ..., M
Mobs = M;
var_ODobs = zeros(1,nOD);
for iOdInd = 1:nOD
    varInd = varInd+1;

    iPaths = arcPaths{iOdInd};
    iTimes = arcPaths_time{iOdInd};

    Mobs(varInd,1) = Cpm( [varInd var_OD(iOdInd)], 1, [1 1; 2 2; 2 3], [1; 1; 1] );
    vars(varInd,1) = Variable( eye(2), {'No disruption'; 'Disruption'} );

    var_ODobs(iOdInd) = varInd;
end

[Mcond, vars] = condition(Mobs,[var_ODobs(1) var_ODobs(2) var_ODobs(3)], [2 2 1], vars);
[Mcond_mult, vars] = multCPMs(Mcond, vars);
Mcond_mult_sum = sum( Mcond_mult, [var_OD var_ODobs] );

arcs_prob_damage = zeros(1,nArc); % P( X_i = 2 | OD_1 = 2, OD_2 = 2, OD_3 = 1 ), i = 1, ..., N
for iArcInd = 1:nArc
    iVarInd = var_arcs(iArcInd);

    iM = sum( Mcond_mult_sum, iVarInd, 0 );
    [iM_fail, vars] = condition( iM, iVarInd, arc_fail, vars );
    
    iFailProb = iM_fail.p / sum(iM.p);
    if ~isempty(iFailProb)
        arcs_prob_damage(iArcInd) = iFailProb;
    else
        arcs_prob_damage(iArcInd) = 0;
    end
end


figure;
bar( [1-arcs_prob_damage(:) arcs_prob_damage(:)] )

grid on
xlabel( 'Arc' )
ylabel( 'Probability' )
legend( {'Survive' 'Fail'}, 'location', 'northwest' )


%% Repeat inferences again using new functions -- the results must be the same.
% Probability of delay and disconnection
[M_VE2, vars] = VE( M, varElimOrder, vars ); % "M_VE2" same as "M_VE"

% Retrieve example results
ODs_prob_disconn2 = zeros(1,nOD); % "ODs_prob_disconn2" same as "ODs_prob_disconn"
ODs_prob_delay2 = zeros(1,nOD); % "ODs_prob_delay2" same as "ODs_prob_delay"
for iODInd = 1:nOD
    iVarInd = var_OD(iODInd);

    % Prob. of disconnection
    iDisconnState = find( arrayfun(@(x) x==100, vars(iVarInd).v) ); % the state of disconnection is assigned an arbitrarily large number 100
    iDisconnProb = getProb( M_VE2, iVarInd, iDisconnState, vars );
    ODs_prob_disconn2(iODInd) = iDisconnProb;

    % Prob. of delay
    iDelayProb = getProb( M_VE2, iVarInd, 1, vars, 0 ); % Any state greater than 1 means delay.
    ODs_prob_delay2(iODInd) = iDelayProb;
end

% Check if the results are the same
disp( ['Are the disconnected probabilities the same?: ' num2str( isequal( ODs_prob_disconn2, ODs_prob_disconn ) )] )
disp( ['Are the delay probabilities the same?: ' num2str( isequal( ODs_prob_delay2, ODs_prob_delay ) )] )