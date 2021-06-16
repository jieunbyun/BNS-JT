function [distribution, sendingMessage, vars] = cliqueFun_failure( distribution, receivedMessage, vars, ...
                                                                   TIME_INTERVAL_IN_YEARS, VAR_FAILURE)

STATE_TRUE = 1; STATE_FALSE = 2;
C_failure = STATE_TRUE * ones( TIME_INTERVAL_IN_YEARS );
C_failure = triu( C_failure );
C_failure( ~C_failure ) = STATE_FALSE;
C_failure = [C_failure; STATE_FALSE * ones( 1, TIME_INTERVAL_IN_YEARS )];

MEAN = 1; COVARIANCE = 1 + (1:TIME_INTERVAL_IN_YEARS);
receivedMessage = receivedMessage{1};
limitStateMean = receivedMessage( MEAN, : );
limitStateCovariance = receivedMessage( COVARIANCE, : );

p_failure = zeros( TIME_INTERVAL_IN_YEARS + 1, 1 );
for time = 1:TIME_INTERVAL_IN_YEARS
    
    iMean = limitStateMean( 1:time );
    iMean( 1:(time-1) ) = - iMean( 1:(time-1) );
    
    iCovariance = limitStateCovariance( 1:time, : );
    iCovariance = iCovariance( :, 1:time );
    iCovariance( :, time ) = -iCovariance( :, time );
    iCovariance( time, : ) = -iCovariance( time, : );
   
    p_failure( time ) = mvncdf( zeros(1, time), iMean, iCovariance  );
end

p_failure(end) = mvncdf( zeros(1, time), -limitStateMean, limitStateCovariance, statset('MaxFunEvals', 1e8)  );

distribution = Cpm( VAR_FAILURE, TIME_INTERVAL_IN_YEARS, C_failure, p_failure );
sendingMessage = [];