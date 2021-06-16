function [posteriorCorrosionRateMean, posteriorCorrosionRateStd] = computePosteriorOfCorrosionRate( measurement, PARAM_CORROSION_RATE, PARAM_MEASUREMENT_ERROR, INITIAL_CAPACITY_IN_MPA )

MEAN = 1; STD = 2;
VALUE = 1; TIME = 2;

if isempty( measurement )    
    posteriorCorrosionRateStd =  PARAM_CORROSION_RATE( MEAN );
    posteriorCorrosionRateMean =  PARAM_CORROSION_RATE( STD );
    
else
    posteriorCorrosionRateStd = exp( 2*log(PARAM_CORROSION_RATE(STD)) - 2*log(PARAM_MEASUREMENT_ERROR(STD)) + log( sum( exp( 2*log(measurement( TIME,: )) ) ) ) );
    posteriorCorrosionRateStd = exp( .5*log( 1+ posteriorCorrosionRateStd ) );
    posteriorCorrosionRateStd = exp( log(PARAM_CORROSION_RATE(STD)) - log(posteriorCorrosionRateStd) );
    
    posteriorCorrosionRateMean = exp( log(sum( measurement( TIME,: ) .* (INITIAL_CAPACITY_IN_MPA - measurement( VALUE,: )) ) ) - 2*log( PARAM_MEASUREMENT_ERROR(STD) ) );
    posteriorCorrosionRateMean = posteriorCorrosionRateMean + exp( log(PARAM_CORROSION_RATE(MEAN)) - 2*log(PARAM_CORROSION_RATE(STD)) );
    posteriorCorrosionRateMean = exp( log(posteriorCorrosionRateMean) + 2*log(posteriorCorrosionRateStd) );
end