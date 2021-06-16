function [distribution, sendingMessage, vars] = cliqueFun_capacity_sampling( distribution, receivedMessage, vars, ...
                                                                             PARAM_CORROSION_RATE, PARAM_MEASUREMENT_ERROR, INITIAL_CAPACITY_IN_MPA, TIME_INTERVAL_IN_YEARS, ...
                                                                             corrosionRateDistributionType, ...
                                                                             VAR_CORROSION_RATE, VAR_CAPACITY)

import steel.*
measurement = receivedMessage{1};
MEAN = 1; STD = 2;
VALUE = 1; TIME = 2;

switch corrosionRateDistributionType
    case 'Normal'
        PARAM_CORROSION_RATE = PARAM_CORROSION_RATE;
    case 'Lognormal'
        PARAM_CORROSION_RATE = computeLognormalParametersFromNormal( PARAM_CORROSION_RATE );
    otherwise
        error( 'Distribution type must be either "Normal" or "Lognormal"' )
end

global nSample
corrosionRateSample = random( corrosionRateDistributionType, PARAM_CORROSION_RATE( MEAN ), PARAM_CORROSION_RATE( STD ), nSample, 1 );
corrosionRateSamplingProb = pdf( corrosionRateDistributionType, corrosionRateSample, PARAM_CORROSION_RATE( MEAN ), PARAM_CORROSION_RATE( STD ) );

distribution = Cpm( VAR_CORROSION_RATE, 1 );
distribution.C = corrosionRateSample;
distribution.q = corrosionRateSamplingProb;
distribution.sampleIndex = (1:nSample)';

capacitySample = INITIAL_CAPACITY_IN_MPA - corrosionRateSample * (1:TIME_INTERVAL_IN_YEARS);
capacitySampleProbPosterior = corrosionRateSamplingProb;
for measureInd = 1:size( measurement, 2)
    iMeasurement = measurement( VALUE, measureInd );
    iMeasureTime = measurement( TIME, measureInd );
    iCapacity = capacitySample( :, iMeasureTime );
    
    capacitySampleProbPosterior = exp( log(capacitySampleProbPosterior) + log( normpdf( iMeasurement, iCapacity + PARAM_MEASUREMENT_ERROR( MEAN ), PARAM_MEASUREMENT_ERROR(STD) ) ) );
end
    
sendingMessage = Cpm( VAR_CAPACITY, length( VAR_CAPACITY ) );
sendingMessage.C = capacitySample;
sendingMessage.p = capacitySampleProbPosterior;
sendingMessage.q = corrosionRateSamplingProb;
distribution.sampleIndex = (1:nSample)';