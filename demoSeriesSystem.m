import demo.*
numComponent = 2;

%% MBN quantification
varInd = 0;
M = Cpm;
vars = Variable;

varInd = varInd + 1;
VAR_HAZARD = varInd;
M(varInd) = Cpm( varInd, 1, [1 2]', [.1 .9]' );
vars(varInd) = Variable( eye(2), {'Severe' 'Mild'}');

varInd = varInd + 1;
VAR_FAULT_INTENSITY(1) = varInd;
M(varInd) = Cpm( [varInd VAR_HAZARD], 1, [1 1; 2 2], [1 1]' );
vars(varInd) = Variable( eye(2), [.05 .001]');

varInd = varInd + 1;
VAR_FAULT_INTENSITY(2) = varInd;
M(varInd) = Cpm( [varInd VAR_HAZARD], 1, [1 1; 2 2], [1 1]' );
vars(varInd) = Variable( eye(2), [.03 .005]');

for compInd = 1:numComponent
    varInd = varInd + 1;
    VAR_COMPONENT(compInd) = varInd;
    M(varInd) = Cpm( [varInd VAR_FAULT_INTENSITY(compInd)], 1, [1 1; 2 1; 1 2; 2 2], [] ); % probability vector is left empty; it is computed later during JTA
    vars(varInd) = Variable( [eye(2); 1 1], {'Survive' 'Fail'});
end

varInd = varInd + 1;
VAR_SYSTEM = varInd;
M(varInd) = Cpm( [varInd VAR_COMPONENT], 1, [2 2 3; 2 1 2; 1 1 1], [1; 1; 1] );
vars(varInd) = Variable( eye(2), {'Survive' 'Fail'});

%% Junction Tree
cliqueInd = 0;
cliqueInd = cliqueInd + 1;
CLIQUE_HAZARD = cliqueInd;

cliqueInd = cliqueInd + 1;
CLIQUE_FAULT_INTENSITY = cliqueInd;

cliqueInd = cliqueInd + 1;
CLIQUE_COMPONENT = cliqueInd;

cliqueInd = cliqueInd + 1;
CLIQUE_SYSTEM = cliqueInd;


cpmInd = {};
cpmInd{ CLIQUE_HAZARD } = VAR_HAZARD;
cpmInd{ CLIQUE_FAULT_INTENSITY } = VAR_FAULT_INTENSITY;
cpmInd{ CLIQUE_COMPONENT } = VAR_COMPONENT;
cpmInd{ CLIQUE_SYSTEM } = VAR_SYSTEM;


messSched = [];
cliqueFun = {};

messSched = [messSched; CLIQUE_HAZARD CLIQUE_FAULT_INTENSITY];
cliqueFun = [cliqueFun; {@(cpms, mess, vars) cliqueFunHazard( cpms, mess, vars )}];

messSched = [messSched; CLIQUE_FAULT_INTENSITY CLIQUE_COMPONENT];
cliqueFun = [cliqueFun; {@(cpms, mess, vars) cliqueFunFaultIntensity( cpms, mess, vars, VAR_FAULT_INTENSITY )}];

messSched = [messSched; CLIQUE_COMPONENT CLIQUE_SYSTEM];
cliqueFun = [cliqueFun; {@(cpms, mess, vars) cliqueFunComponent( cpms, mess, vars, VAR_FAULT_INTENSITY, VAR_COMPONENT )}];

messSched = [messSched; CLIQUE_SYSTEM 0];
cliqueFun = [cliqueFun; {@(cpms, mess, vars) cliqueFunSystem( cpms, mess, vars )}];


jtree = JunctionTree;
jtree.cliqueFunctionCell = cliqueFun;
jtree.cpmInd = cpmInd;
jtree.messageSchedule = messSched;
jtree.cpm = M;
jtree.variables = vars;

[cliques, messages, vars] = runJunctionTree(jtree);


cpmSystem = sum( cliques{CLIQUE_SYSTEM}, VAR_SYSTEM, 0 );
STATE_FAIL = 2;
systemFailProb = cpmSystem.p( STATE_FAIL );
disp( ['[Junction Tree] System failure probability: ' num2str( systemFailProb )] )

%% Conditioning
cliqueInd = 0;
for compInd = 1:numComponent
    cliqueInd = cliqueInd + 1;
    CLIQUE_COND_FAULT_INTENSITY( compInd ) = cliqueInd;
end

for compInd = 1:numComponent
    cliqueInd = cliqueInd + 1;
    CLIQUE_COND_COMPONENT( compInd ) = cliqueInd;
end

cliqueInd = cliqueInd + 1;
CLIQUE_COND_SYSTEM = cliqueInd;


cpmInd_cond = {};
for compInd = 1:numComponent
    cpmInd_cond{ CLIQUE_COND_FAULT_INTENSITY( compInd ) } = VAR_FAULT_INTENSITY(compInd);
end
for compInd = 1:numComponent
    cpmInd_cond{ CLIQUE_COND_COMPONENT( compInd ) } = VAR_COMPONENT(compInd);
end
cpmInd_cond{ CLIQUE_COND_SYSTEM } = VAR_SYSTEM;


messSched_cond = [];
cliqueFun_cond = {};

for compInd = 1:numComponent
    messSched_cond = [messSched_cond; CLIQUE_COND_FAULT_INTENSITY( compInd ) CLIQUE_COND_COMPONENT( compInd )];
    cliqueFun_cond = [cliqueFun_cond; {@(cpms, mess, vars) cliqueFunFaultIntensity_conditioning( cpms, mess, vars )}];
end

for compInd = 1:numComponent
    messSched_cond = [messSched_cond; CLIQUE_COND_COMPONENT( compInd ) CLIQUE_COND_SYSTEM];
    cliqueFun_cond = [cliqueFun_cond; {@(cpms, mess, vars) cliqueFunComponent_conditioning( cpms, mess, vars, VAR_FAULT_INTENSITY( compInd ), VAR_COMPONENT( compInd ) )}];
end

messSched_cond = [messSched_cond; CLIQUE_COND_SYSTEM 0];
cliqueFun_cond = [cliqueFun_cond; {@(cpms, mess, vars) cliqueFunSystem_conditioning( cpms, mess, vars )}];


jtree_cond = JunctionTree;
jtree_cond.cliqueFunctionCell = cliqueFun_cond;
jtree_cond.cpmInd = cpmInd_cond;
jtree_cond.messageSchedule = messSched_cond;
jtree_cond.cpm = M;
jtree_cond.variables = vars;

[cliques_cond, messages_cond, vars] = runJunctionTree_conditioning(jtree_cond, M( VAR_HAZARD ));


cpmSystem_cond = sum( cliques_cond{CLIQUE_COND_SYSTEM}, VAR_SYSTEM, 0 );
systemFailProb_cond = cpmSystem_cond.p( STATE_FAIL );
disp( ['[Junction Tree by Conditioning] System failure probability: ' num2str( systemFailProb_cond )] )

%% Parameter sensitivity
sensitivity = [];
for compInd = 1:numComponent
    iVarParam = VAR_FAULT_INTENSITY( compInd );
    
    for paramInd = 1:length( vars( iVarParam ).v )
        iSensitiveClique = CLIQUE_COMPONENT;
        iSensitiveCliqueFun = {@(cpms, mess, vars) cliqueFunComponent_sensitivity( cpms, mess, vars, VAR_FAULT_INTENSITY, VAR_COMPONENT, compInd, paramInd )};
        
        [iCliques_sens, iMessages_sens, vars] = runJunctionTree_sensitivity(jtree, iSensitiveClique, iSensitiveCliqueFun);
        iCpmSystem_sens = sum( iCliques_sens{ CLIQUE_SYSTEM }, VAR_SYSTEM, 0 );
        iSystemFailProb_sens = iCpmSystem_sens.p( STATE_FAIL );
        
        sensitivity = [sensitivity; iSystemFailProb_sens];
    end
end

disp( ['[Parameter sensitivity] Sensitivity of system fail. prob. w.r.t. fault intensity: ' num2str( sensitivity' )] )

%% Parameter sensitivity & Conditioning
sensitivity_cond = [];
for compInd = 1:numComponent
    iVarParam = VAR_FAULT_INTENSITY( compInd );
    
    for paramInd = 1:length( vars( iVarParam ).v )
        iSensitiveClique = CLIQUE_COND_COMPONENT( compInd );
        iSensitiveCliqueFun = {@(cpms, mess, vars) cliqueFunComponent_sensitivity_conditioning( cpms, mess, vars, VAR_FAULT_INTENSITY( compInd ), VAR_COMPONENT( compInd ), paramInd )};
        
        [iCliques_sens_cond, iMessages_sens_cond, vars] = runJunctionTree_sensitivity_conditioning(jtree_cond, iSensitiveClique, iSensitiveCliqueFun, M(VAR_HAZARD));
        iCpmSystem_sens_cond = sum( iCliques_sens_cond{ CLIQUE_COND_SYSTEM }, VAR_SYSTEM, 0 );
        iSystemFailProb_sens_cond = iCpmSystem_sens_cond.p( STATE_FAIL );
        
        sensitivity_cond = [sensitivity_cond; iSystemFailProb_sens_cond];
    end
end

disp( ['[Parameter sensitivity by conditioning] Sensitivity of system fail. prob. w.r.t. fault intensity: ' num2str( sensitivity_cond' )] )