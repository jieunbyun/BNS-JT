clear
import steel.*
load steelPipe_JT

%% Inference
rng(1)
VALUE = 1; TIME = 2;
measurement = [];
measurement( VALUE, : ) = [47.04 46.84];
measurement( TIME, : ) = [10 15];
nMeasurement = size( measurement, 2 );

failureProbMean = zeros( TIME_INTERVAL_IN_YEARS, nMeasurement + 1 );
failureProbStd = zeros( TIME_INTERVAL_IN_YEARS, nMeasurement + 1 );
TRUE = 1;
    
for measureInd = 0:nMeasurement
    iMeasurement = measurement(:,1:measureInd);
    
    cliqueFun{ CLIQUE_MEASUREMENT } = @(dist, mess, vars) cliqueFun_measurement( dist, mess, vars, iMeasurement );

    corrosionRateDistributionType = 'Lognormal';
    cliqueFun{ CLIQUE_CAPACITY } = @(dist, mess, vars) cliqueFun_capacity_sampling( dist, mess, vars, ...
                                                                                    PARAM_CORROSION_RATE, PARAM_MEASUREMENT_ERROR, INITIAL_CAPACITY_IN_MPA, TIME_INTERVAL_IN_YEARS, ...
                                                                                    corrosionRateDistributionType, ...
                                                                                    VAR_CORROSION_RATE, VAR_CAPACITY);

    demandDistributionType = 'Lognormal';
    cliqueFun{ CLIQUE_LIMIT_STATE } = @(dist, mess, vars) cliqueFun_limiteState_sampling( dist, mess, vars, ...
                                                                                          demandDistributionType, PARAM_DEMAND, TIME_INTERVAL_IN_YEARS, ...
                                                                                          VAR_DEMAND, VAR_LIMIT_STATE);

    cliqueFun{ CLIQUE_FAILURE } = @(dist, mess, vars) cliqueFun_failure_sampling( dist, mess, vars, ...
                                                                                  VAR_FAILURE);


    jtree = JunctionTree;
    jtree.cliqueFunctionCell = cliqueFun;
    jtree.cpmInd = distributionInd;
    jtree.messageSchedule = messSched;
    jtree.cpm = M;
    jtree.variables = vars;
    jtree.nSample = 1e6;
    [cliques, messages, vars] = runJunctionTree(jtree);

    %% Results
    failureCpm = cliques{ CLIQUE_FAILURE };
    failureSampleWeight = failureCpm.p;

    for time = 1:TIME_INTERVAL_IN_YEARS
        
        if measureInd == 0
            [iFailureProbMean, iFailureProbStd] = computeSample( failureCpm, VAR_FAILURE( time ), TRUE, 'mcs' );
        else            
            [iFailureProbMean, iFailureProbStd] = computeSample( failureCpm, VAR_FAILURE( time ), TRUE, 'normalizedIS' );
        end

        failureProbMean( time, measureInd+1 ) = iFailureProbMean;
        failureProbStd( time, measureInd+1 ) = iFailureProbStd;
    end

end

%% Plotting
% Sampling
figure;
hold on
samplingResultColorSpec = [ .98 .89 .56; 1 0.71 .6; 0.5843 .8157 .9882];
timeForPlot = [measurement( TIME, : ) TIME_INTERVAL_IN_YEARS];

for measureInd = nMeasurement:-1:0
    iTimePlot = 1:timeForPlot( measureInd+1 );
    iUpperInterval95Confidence = failureProbMean( iTimePlot,measureInd+1 ) + 1.96*failureProbStd( iTimePlot,measureInd+1 );
    iLowerInterval95Confidence = failureProbMean( iTimePlot,measureInd+1 ) - 1.96*failureProbStd( iTimePlot,measureInd+1 );
    
    iUpperInterval95Confidence( iUpperInterval95Confidence<=0 ) = 1e-12;
    iLowerInterval95Confidence( iLowerInterval95Confidence <= 0 ) = 1e-12;
    
    f = fill( [iTimePlot fliplr(iTimePlot)], [iUpperInterval95Confidence' fliplr(iLowerInterval95Confidence')], samplingResultColorSpec( measureInd+1,: ), 'LineStyle', 'none' ); 
    if measureInd == 1 || measureInd == 0
        set(f, 'facealpha', .4)
    end
    
end
set(gca, 'YScale', 'log')
axis( [0 20 1e-6 1e-1] )
grid on

figureForLegned = fill( [0 1], [1e-7 1e-7], .8*[1 1 1], 'LineStyle', 'none' );

% Mean value
lineWidth = 1.5; fontSizeLegend = 14; fontSizeTick = 13; fontSizeLabel = 16;
lineShape = {'+--' 'o--' '*--'};
meanValueColorSpec = [0.9290 0.6940 0.1250; 0.8500 0.3250 0.0980; 0 0.4470 0.7410];
  
for plotInd = 1:size( failureProbMean, 2 )
    iTimePlot = 1:timeForPlot(plotInd);
    h{plotInd} = plot( iTimePlot, failureProbMean( iTimePlot,plotInd ), lineShape{plotInd}, 'LineWidth', lineWidth, 'Color', meanValueColorSpec( plotInd,: ) );
end  
set(gca, 'YScale', 'log')
grid on

legend( [h{1} h{2} h{3} figureForLegned], {'No measurement' 'Measurement at 10 yr' 'Measurement at 15 yr' '95% Confidence interval'}, 'Fontsize', fontSizeLegend, 'Location', 'SouthEast', ...
    'FontName','times new roman')

ax = gca;
ax.XAxis.FontSize = fontSizeTick;
ax.YAxis.FontSize = fontSizeTick;
ax.XAxis.FontName = 'times new roman';
ax.YAxis.FontName = 'times new roman';

xlabel( 'Years', 'Fontsize', fontSizeLabel, 'FontName', 'times new roman' )
ylabel( 'Failure probability','Fontsize',fontSizeLabel,'FontName','times new roman' )

saveas(gcf,'figure/steelPipe_failureProbability_lognormal.emf')
saveas(gcf,'figure/steelPipe_failureProbability_lognormal.pdf')