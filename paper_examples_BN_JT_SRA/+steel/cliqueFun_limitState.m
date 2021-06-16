function [distribution, sendingMessage, vars] = cliqueFun_limitState( distribution, receivedMessage, vars, ...
                                                                    PARAM_DEMAND, TIME_INTERVAL_IN_YEARS)

receivedMessage = receivedMessage{1};

MEAN = 1; COVARIANCE = 1 + (1:TIME_INTERVAL_IN_YEARS);
posteriorCapacityMean = receivedMessage( MEAN, : );
posteriorCapacityCovariance = receivedMessage( COVARIANCE, : );

STD = 2;
limitStateMean = posteriorCapacityMean - PARAM_DEMAND( MEAN );
limitStateCovariance = posteriorCapacityCovariance + diag( repmat( PARAM_DEMAND( STD )^2, 1, TIME_INTERVAL_IN_YEARS ) );

sendingMessage = [limitStateMean; limitStateCovariance];
distribution = [];