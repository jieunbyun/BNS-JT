function [distribution, sendingMessage, vars] = cliqueFun_limiteState_sampling( distribution, receivedMessage, vars, ...
                                                                                demandDistributionType, PARAM_DEMAND, TIME_INTERVAL_IN_YEARS, ...
                                                                                VAR_DEMAND, VAR_LIMIT_STATE)
    
import steel.*
capacityCPM = receivedMessage{1};
MEAN = 1; STD = 2;

switch demandDistributionType
    case 'Normal'
        PARAM_DEMAND = PARAM_DEMAND;
    case 'Lognormal'
        PARAM_DEMAND = computeLognormalParametersFromNormal( PARAM_DEMAND );
    otherwise
        error( 'Distribution type must be either "Normal" or "Lognormal"' )
end

global nSample
demandSample = random( demandDistributionType, PARAM_DEMAND( MEAN ), PARAM_DEMAND(STD), nSample, TIME_INTERVAL_IN_YEARS );
demandSamplingProb = pdf( demandDistributionType, demandSample, PARAM_DEMAND( MEAN ), PARAM_DEMAND(STD) );
demandSamplingProb = exp( sum( log(demandSamplingProb), 2 ) );

distribution = Cpm( VAR_DEMAND, length( VAR_DEMAND ) );
distribution.C = demandSample;
distribution.q = demandSamplingProb;
distribution.sampleIndex = (1:nSample)';


limitStateSample = capacityCPM.C - demandSample;
limitStateSamplingPosteior = exp( log( capacityCPM.p ) + log( demandSamplingProb ) );
limitStateSamplingProb = exp( log( capacityCPM.q ) + log( demandSamplingProb ) );
sendingMessage = Cpm( VAR_LIMIT_STATE, length(VAR_LIMIT_STATE) );
sendingMessage.C = limitStateSample;
sendingMessage.p = limitStateSamplingPosteior;
sendingMessage.q = limitStateSamplingProb;
sendingMessage.sampleIndex = (1:nSample)';