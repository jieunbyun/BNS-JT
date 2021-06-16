clear
import kN.*

%% MBN quantification
varInd = 0;
M = Cpm;
vars = Variable;

% Environment
varInd = varInd + 1;
VAR_WEATHER = varInd;

Cweather = (1:3)'; pWeather = [1/2 1/3 1/6]';
M(varInd) = Cpm( varInd, 1, Cweather, pWeather );
vars(varInd) = Variable( eye( max(Cweather) ), {'Sunny' 'Cloudy' 'Rainy'}' );

varInd = varInd + 1;
VAR_VISIBILITY = varInd;

Cvisibility = [1 1; 2 1; 2 2; 1 3; 2 3]; pVisibility = [5/6 1/6 1 1/3 2/3]';
M(varInd) = Cpm( [varInd VAR_WEATHER], 1, Cvisibility, pVisibility );
vars(varInd) = Variable( eye( max( Cvisibility(:,1) ) ), {'Good' 'Bad'}' );

varInd = varInd + 1;
VAR_TEMPERATURE = varInd;

Ctempature = (1:2)'; pTempature = [1/4 3/4]';
M(varInd) = Cpm( varInd, 1, Ctempature, pTempature );
vars(varInd) = Variable( eye( max( Ctempature) ), {'Below 0' 'Above 0'}' );


% Component
LAMBDA = 0.1 * [1 2 1 2 2.5 0.5 2 1.5 3 3.5; 1.5 3 2 2.5 3 2 2.5 3 3 4; 2 4.5 3 4 4 3.5 3.5 4 5 4.5; ...
    2.5 4.5 3.5 4 4 3.5 3.5 3.5 5 6; 3 5.5 4 4.5 7.5 3.5 4 3 4.5 4.5];

Menvironment = multCPMs( M([VAR_WEATHER VAR_VISIBILITY VAR_TEMPERATURE]), vars );
nEnvironment = size( Menvironment.C, 1 );

STATE_SURVIVE = 1; STATE_FAIL = 2; COMPOSITE_STATE_COMP = 3;
Ccomp = [repmat( [STATE_SURVIVE STATE_FAIL]', nEnvironment, 1 ) repelem( Menvironment.C, 2, 1 ) ];
Bcomp = [eye(2); 1 1]; 

nComponentType = size(LAMBDA, 1);

nCompEachType = 20;
parameterSensitivity = zeros( size(LAMBDA) );
    
varInd = 3;
compVarInd = 0;
compTypeArray = zeros(nComponentType*nCompEachType, 1);
for typeInd = 1:nComponentType

    iLambda = LAMBDA(typeInd,:);
    iPcomp = computeProbVectorFromLambda( iLambda );

    for compInd = 1:nCompEachType
        varInd = varInd + 1;
        compVarInd = compVarInd + 1;
        VAR_COMPONENT( compVarInd ) = varInd;
        compTypeArray( compVarInd ) = typeInd;

        M( varInd ) = Cpm( [varInd Menvironment.variables], 1, Ccomp, iPcomp );
        vars( varInd ) = Variable( Bcomp, {'Survive' 'Fail'}' );
    end    
end

% System
K = 40;
nComponent = length( VAR_COMPONENT );
[Msystem, varsSystem, VAR_INTERMEDIATE_NODE] = quantifyMbnOfKnsystem( K, nComponent, VAR_COMPONENT, varInd, STATE_SURVIVE, STATE_FAIL, COMPOSITE_STATE_COMP );  

M( VAR_INTERMEDIATE_NODE ) = Msystem;
vars( VAR_INTERMEDIATE_NODE ) = varsSystem;

%% Junction Tree
cliqueInd = 0;

for ii = 1:nComponent
    cliqueInd = cliqueInd + 1;
    CLIQUE_INTERMEDIATE(ii) = cliqueInd;
end

cliqueInd = cliqueInd + 1;
CLIQUE_SYSTEM = cliqueInd;


cpmInd = {};
cpmInd{ CLIQUE_INTERMEDIATE(1) } = [VAR_COMPONENT(1) VAR_INTERMEDIATE_NODE(1) VAR_INTERMEDIATE_NODE(2)];
for ii = 2:nComponent
    cpmInd{ CLIQUE_INTERMEDIATE(ii) } = [VAR_COMPONENT(ii) VAR_INTERMEDIATE_NODE(ii+1)];
end

cpmInd{ CLIQUE_SYSTEM } = [];


messSched = [];
cliqueFun = {};
for ii = 1:nComponent
    if ii < nComponent
        messSched = [messSched; CLIQUE_INTERMEDIATE(ii) CLIQUE_INTERMEDIATE(ii+1)];
    else
        messSched = [messSched; CLIQUE_INTERMEDIATE(ii) CLIQUE_SYSTEM];
    end

    cliqueFun = [cliqueFun; {@(cpms, mess, vars) messFromSystemToSystem( cpms, mess, vars, VAR_INTERMEDIATE_NODE(ii+1), Menvironment.variables )}];
end

messSched = [messSched; CLIQUE_SYSTEM 0];
cliqueFun = [cliqueFun; {@(cpms, mess, vars) computeSystemClique( cpms, mess, vars, VAR_INTERMEDIATE_NODE(end), Menvironment.variables )}];


jtree = JunctionTree;
jtree.cliqueFunctionCell = cliqueFun;
jtree.cpmInd = cpmInd;
jtree.messageSchedule = messSched;
jtree.cpm = M;
jtree.variables = vars;

[MweatherVisibility, vars] = multCPMs( M( [VAR_WEATHER VAR_VISIBILITY] ), vars );
conditionedCpms = [MweatherVisibility M( VAR_TEMPERATURE )];


environmentStates = unique( M( VAR_COMPONENT(1) ).C(:, 2:end), 'rows' );
nEnvironments = size( environmentStates, 1 );

parameterSensitivity = zeros( nComponentType, nEnvironments );
for sensitivityInd_compType = 1:nComponentType
    for sensitivityInd_environment = 1:nEnvironments

        sensitivityInd_environmentState = environmentStates( sensitivityInd_environment, : );
        paramValue = LAMBDA( sensitivityInd_compType, sensitivityInd_environment );

        compIndOfThisType = find( compTypeArray == sensitivityInd_compType );
        sensitiveCliqueIndArray = CLIQUE_INTERMEDIATE( compIndOfThisType );
        sensitiveCliqueFun = cell( size( sensitiveCliqueIndArray ) );
        for cliqueInd = 1:length( sensitiveCliqueFun )
            iCompInd = compIndOfThisType( cliqueInd );
            sensitiveCliqueFun(cliqueInd) = {@(cpms, mess, vars) messFromSystemToSystem_sensitivity( cpms, mess, vars, ...
                                                                 VAR_INTERMEDIATE_NODE(iCompInd+1), Menvironment.variables, sensitivityInd_environmentState, paramValue )};
        end


        [cliques_conditioned, messages_conditioned, vars] = runJunctionTree_sensitivity_conditioning(jtree, sensitiveCliqueIndArray, sensitiveCliqueFun, conditionedCpms);
        parameterSensitivity( sensitivityInd_compType, sensitivityInd_environment ) = cliques_conditioned{ CLIQUE_SYSTEM }.p( STATE_FAIL );
    end
end


save ex1_commonEnvironment_sensitivityAnalysis

figure('Renderer', 'painters', 'Position', [488   342   400   420]);
imagesc( log(parameterSensitivity') )
h = colorbar;

xticks( 1:nComponentType )
yticks( 1:nEnvironment )

lineWidth = 1.3; fontSizeLegend = 14; fontSizeTick = 13; fontSizeLabel = 16;

ax = gca;
ax.XAxis.FontSize = fontSizeTick;
ax.YAxis.FontSize = fontSizeTick;
ax.XAxis.FontName = 'times new roman';
ax.YAxis.FontName = 'times new roman';

xlabel( 'Component type, \it{q}', 'Fontsize', fontSizeLabel, 'FontName', 'times new roman' )
ylabel( 'State of common cause, \it{j}','Fontsize',fontSizeLabel,'FontName','times new roman' )
ylabel(h, 'Log scale','Fontsize',fontSizeLabel,'FontName','times new roman','Rotation',270)
h.Label.Position(1) = 3.5;
saveas(gcf,'figure/CommonEnvironment_sensitivity.emf')
saveas(gcf,'figure/CommonEnvironment_sensitivity.pdf')