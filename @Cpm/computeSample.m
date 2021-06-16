function [sampleMean, sampleStd] = computeSample( cpm, queryVariables, queryStates, samplingOption )

if length( cpm ) ~= 1; error( 'Given CPM must be a single Cpm' ); end
if length( queryVariables ) ~= length( queryStates ); error( 'Given variables and states of query must have the same length' ); end
if all( ~ismember( queryVariables, cpm.variables ) ); error( 'None of the variables is in the scope' ); end

if any( ~ismember( queryVariables, cpm.variables ) ); warning( 'Some of the query variables are not in the given scope' ); end

if nargin < 4; samplingOption = 'mcs'; end


[~, variableIndInC] = ismember( queryVariables, cpm.variables );
variableNotInScope = find( ~variableIndInC );
queryStates( variableNotInScope ) = [];
variableIndInC( variableNotInScope ) = [];

instanceBoolean = ismember(cpm.C( :, variableIndInC ), queryStates, 'rows' );

switch samplingOption
    case 'mcs'
        nSample = length( cpm.q );
        sampleMean = sum( instanceBoolean ) / nSample;
        sampleVar = sampleMean * (1-sampleMean) / nSample;
        sampleStd = sqrt( sampleVar );
        
    case 'unnormalizedIS'
        nSample = length( cpm.q );
        sampleWeight = exp( log( cpm.p ) - log( cpm.q ) );
        sampleMean = sum( sampleWeight( instanceBoolean ) ) / nSample;
        sampleVar = sum( ( sampleWeight( instanceBoolean ) - sampleMean ).^2 ) / nSample;
        sampleStd = sqrt( sampleVar );
        
    case 'normalizedIS'
        sampleWeight = exp( log( cpm.p ) - log( cpm.q ) );
        sampleMean = sum( sampleWeight( instanceBoolean ) ) / sum( sampleWeight );
        sampleVar = sum( exp( 2*log(sampleWeight) + log( ( instanceBoolean-sampleMean ).^2 ) ) );
        sampleVar = exp( log(sampleVar) - 2*log( sum( sampleWeight ) ) );
        sampleStd = sqrt( sampleVar );
        
    otherwise
        error( "Sampling option must be given either 'mcs', 'unnormalizedIS', or 'normalizedIS'" )
end
        