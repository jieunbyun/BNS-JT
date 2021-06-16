function [alpha, beta] = mleOfFaultIntensityParameters( faultObservationTimes, beta0 )

import kN.*

if ~isvector( faultObservationTimes )
    error( 'Fault observation times must be a vector' )
else
    faultObservationTimes = faultObservationTimes(:);
end

if ~isscalar( beta0 ) || beta0 <= 0
    error( 'Initial value of beta must be a positive scalar' )
end



endTime = faultObservationTimes( end ); 
totalNFaults = length( faultObservationTimes );

nFaults = (1:totalNFaults)';
fromZeroNFaults = nFaults - 1;
fromZeroTimes = [0; faultObservationTimes(1:end-1)];

equation = @(beta) ( endTime * totalNFaults * exp( -beta*endTime ) ) / ( 1 - exp( -beta*endTime ) ) - ...
                    sum( ( nFaults - fromZeroNFaults ) .* ( faultObservationTimes .* exp( -beta*faultObservationTimes ) - fromZeroTimes .* exp( -beta*fromZeroTimes ) ) ./ ...
                    ( exp( -beta * fromZeroTimes ) -exp( -beta*faultObservationTimes ) ) );
                
[beta,~,exitflag] = fzero( equation, beta0 );


if exitflag ~= 1
    warning( 'Matlab fun "fzero" failed to find beta value' )
    alpha = nan;
elseif beta < 0
    warning( 'Obtained beta value is negative' )
    alpha = nan;
else
    alpha = totalNFaults / (1-exp(-beta*endTime));
end

