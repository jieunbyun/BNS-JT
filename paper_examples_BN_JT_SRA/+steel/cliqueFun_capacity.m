function [distribution, sendingMessage, vars] = cliqueFun_capacity( distribution, receivedMessage, vars, ...
                                                                    PARAM_CORROSION_RATE, PARAM_MEASUREMENT_ERROR, INITIAL_CAPACITY_IN_MPA, TIME_INTERVAL_IN_YEARS)

                                                                
import steel.*

receivedMessage = receivedMessage{1};
[posteriorCorrosionRateMean, posteriorCorrosionRateStd] = computePosteriorOfCorrosionRate( receivedMessage, PARAM_CORROSION_RATE, PARAM_MEASUREMENT_ERROR, INITIAL_CAPACITY_IN_MPA );

posteriorCapacityMean = INITIAL_CAPACITY_IN_MPA - (1:TIME_INTERVAL_IN_YEARS) * posteriorCorrosionRateMean;
posteriorCapacityCovariance = (1:TIME_INTERVAL_IN_YEARS)' * (1:TIME_INTERVAL_IN_YEARS) * posteriorCorrosionRateStd^2;


sendingMessage  = [posteriorCapacityMean; posteriorCapacityCovariance];
distribution = [];