function lognormalParameters = computeLognormalParametersFromNormal( normalParameters )

MEAN = 1; STD = 2;
mean = normalParameters( MEAN );
std = normalParameters( STD );
var = exp( 2*log( std ) );

logMean = log( mean^2 / sqrt(mean^2+var) );
logVar = log( 1 + var / mean^2 );
logStd = sqrt( logVar );

lognormalParameters = [logMean logStd];