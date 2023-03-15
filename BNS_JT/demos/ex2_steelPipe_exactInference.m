clear
import steel.*

%% Parameters
TIME_INTERVAL_IN_YEARS = 20;

MEAN = 1; STD = 2;
PARAM_CORROSION_RATE([MEAN STD]) = [0.2 0.2];
PARAM_MEASUREMENT_ERROR( [MEAN STD] ) = [0 0.5];
PARAM_DEMAND( MEAN ) = 40;

covRatioDemandCapacity = .5;
demandStdEquation = [-norminv(1e-6) PARAM_DEMAND(MEAN)-PARAM_CORROSION_RATE(MEAN)*TIME_INTERVAL_IN_YEARS ...
                     -covRatioDemandCapacity*PARAM_DEMAND(MEAN)*PARAM_CORROSION_RATE(STD)*TIME_INTERVAL_IN_YEARS];
demandStd = roots( demandStdEquation );
demandStd = demandStd( demandStd>0 );
PARAM_DEMAND( STD ) = demandStd;

INITIAL_CAPACITY_IN_MPA = -norminv(1e-6)*demandStd + PARAM_DEMAND( MEAN );

%% Quantification
varInd = 0;
M = Cpm;
vars = Variable;

% Corrosion rate
varInd = varInd+1;
VAR_CORROSION_RATE = varInd;

M(varInd) = Cpm( varInd, 1 );
vars(varInd) = Variable( [], {'Corrosion rate (/year)'} );

% Capacity
corrosionRateFunction = @(corrosionRate,time) INITIAL_CAPACITY_IN_MPA - corrosionRate * time;

for time = 1:TIME_INTERVAL_IN_YEARS
    varInd = varInd+1;
    VAR_CAPACITY( time ) = varInd;
    
    M(varInd) = Cpm( [varInd VAR_CORROSION_RATE], 1 );
    vars(varInd) = Variable( [], {'Capacity (MPa)'} );
end

% Measurements
for time = 1:TIME_INTERVAL_IN_YEARS
    varInd = varInd+1;
    VAR_MEASUREMENT( time ) = varInd;
    
    M(varInd) = Cpm( [varInd VAR_CAPACITY(time)], 1 );
    vars(varInd) = Variable( [], {'Measured capacity (MPa)'} );
end

% Demand
for time = 1:TIME_INTERVAL_IN_YEARS
    varInd = varInd+1;
    VAR_DEMAND( time ) = varInd;
    
    M(varInd) = Cpm( varInd, 1 );
    vars(varInd) = Variable( [], {'Demand (MPa)'} );
end

% Limit-state function
limitStateFunction = @(capacity, demand) capacity - demand;

for time = 1:TIME_INTERVAL_IN_YEARS
    varInd = varInd+1;
    VAR_LIMIT_STATE( time ) = varInd;
    
    M(varInd) = Cpm( [varInd VAR_DEMAND(time) VAR_CAPACITY(time)], 1 );
    vars(varInd) = Variable( [], {'Limit-state function'} );
end

% Pipe failure
for time = 1:TIME_INTERVAL_IN_YEARS
    varInd = varInd+1;
    VAR_FAILURE( time ) = varInd;
    
    M(varInd) = Cpm( [varInd VAR_LIMIT_STATE(1:time)], 1 );
    vars(varInd) = Variable( eye(2), {'Failure' 'Survival'}' );
end

%% Inference
cliqueInd = 0;

cliqueInd = cliqueInd + 1;
CLIQUE_MEASUREMENT = cliqueInd;

cliqueInd = cliqueInd + 1;
CLIQUE_CAPACITY = cliqueInd;

cliqueInd = cliqueInd + 1;
CLIQUE_LIMIT_STATE = cliqueInd;

cliqueInd = cliqueInd + 1;
CLIQUE_FAILURE = cliqueInd;


distributionInd{ CLIQUE_MEASUREMENT } = VAR_MEASUREMENT;
distributionInd{ CLIQUE_CAPACITY } = [VAR_CORROSION_RATE VAR_CAPACITY];
distributionInd{ CLIQUE_LIMIT_STATE } = [VAR_DEMAND VAR_LIMIT_STATE];
distributionInd{ CLIQUE_FAILURE } = VAR_FAILURE;


messSched = [];
cliqueFun = {};
messSched = [messSched; CLIQUE_MEASUREMENT CLIQUE_CAPACITY];
cliqueFun = [cliqueFun; {@(dist, mess, vars) cliqueFun_measurement( dist, mess, vars, measurement )}];

messSched = [messSched; CLIQUE_CAPACITY CLIQUE_LIMIT_STATE];
cliqueFun = [cliqueFun; {@(dist, mess, vars) cliqueFun_capacity( dist, mess, vars, ...
                                                                PARAM_CORROSION_RATE, PARAM_MEASUREMENT_ERROR, INITIAL_CAPACITY_IN_MPA, TIME_INTERVAL_IN_YEARS)}];
                                                            
messSched = [messSched; CLIQUE_LIMIT_STATE CLIQUE_FAILURE];    
cliqueFun = [cliqueFun; {@(dist, mess, vars) cliqueFun_limitState( dist, mess, vars, ...
                                                                PARAM_DEMAND, TIME_INTERVAL_IN_YEARS)}];
                                                                                                                        
messSched = [messSched; CLIQUE_FAILURE 0];    
cliqueFun = [cliqueFun; {@(dist, mess, vars) cliqueFun_failure( dist, mess, vars, ...
                                                                TIME_INTERVAL_IN_YEARS, VAR_FAILURE)}];

VALUE = 1; TIME = 2;
measurement( VALUE, : ) = [47.04 46.84];
measurement( TIME, : ) = [10 15];

save steelPipe_JT

nMeasurement = size( measurement, 2 );
systemFailureProb = zeros( TIME_INTERVAL_IN_YEARS, nMeasurement+1 );
for measureInd = 0:nMeasurement
    
    iMeasurement = measurement( :, 1:measureInd );
    cliqueFun{ CLIQUE_MEASUREMENT } = @(dist, mess, vars) cliqueFun_measurement( dist, mess, vars, iMeasurement );

    jtree = JunctionTree;
    jtree.cliqueFunctionCell = cliqueFun;
    jtree.cpmInd = distributionInd;
    jtree.messageSchedule = messSched;
    jtree.cpm = M;
    jtree.variables = vars;

    [cliques, messages, vars] = runJunctionTree(jtree);
    iSystemFailureProb = zeros( TIME_INTERVAL_IN_YEARS, 1 );
    STATE_TRUE = 1;
    for time = 1:TIME_INTERVAL_IN_YEARS
        iDistribution = sum( cliques{CLIQUE_FAILURE}, VAR_FAILURE(time), 0 );
        iSystemFailureProb(time) = iDistribution.p( STATE_TRUE );
    end
    
    systemFailureProb( :, measureInd+1 ) = iSystemFailureProb;
    
end

save ex2_steelPipe_exactInference

%% Plotting
lineWidth = 1.5; fontSizeLegend = 14; fontSizeTick = 13; fontSizeLabel = 16;
lineShape = {'+--' 'o--' '*--'};

figure;
hold on
plotTimes = [measurement( TIME, : ) TIME_INTERVAL_IN_YEARS];
for plotInd = 1:size( systemFailureProb, 2 )
    iPlotTimes = 1:plotTimes(plotInd);
    plot( iPlotTimes, systemFailureProb( iPlotTimes,plotInd ), lineShape{plotInd}, 'LineWidth', lineWidth );
end    
set(gca, 'YScale', 'log')
grid on

legend( {'No measurement' 'Measurement at 10 yr' 'Measurement at 15 yr'}, 'Fontsize', fontSizeLegend, 'Location', 'SouthEast', ...
    'FontName','times new roman')

ax = gca;
ax.XAxis.FontSize = fontSizeTick;
ax.YAxis.FontSize = fontSizeTick;
ax.XAxis.FontName = 'times new roman';
ax.YAxis.FontName = 'times new roman';

xlabel( 'Years', 'Fontsize', fontSizeLabel, 'FontName', 'times new roman' )
ylabel( 'Failure probability','Fontsize',fontSizeLabel,'FontName','times new roman' )
saveas(gcf,'figure/steelPipe_failureProbability_exact.emf')
saveas(gcf,'figure/steelPipe_failureProbability_exact.pdf')