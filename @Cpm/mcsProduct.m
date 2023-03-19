function cpm = mcsProduct( cpmArray, nSample, vars )


[sampleOrder, sampleVars, varAdditionOrder] = getSamplingOrder( cpmArray );

nVars = length( sampleVars );
Cproduct = zeros( nSample, nVars );
qproduct = zeros( nSample, 1 );
sampleIndProduct = zeros( nSample, 1 );
for sampleInd = 1:nSample
    
    [sample, sampleProb] = singleSample( cpmArray, sampleOrder, sampleVars, varAdditionOrder, vars, sampleInd );
    Cproduct(sampleInd,:) = sample;
    qproduct(sampleInd) = sampleProb;
    sampleIndProduct(sampleInd) = sampleInd;
    
end


Cproduct  = Cproduct(:, end:-1:1); % Just for asthetic ordering
sampleVars = sampleVars(end:-1:1);
cpm = Cpm(sampleVars, nVars, Cproduct, [], qproduct, sampleIndProduct );

end

function [sampleOrder, sampleVars, varAdditionOrder] = getSamplingOrder( cpmArray )

nCpm = length( cpmArray );

cpmArrayLoop = cpmArray;
cpmIndArray = 1:nCpm;

sampleOrder = [];
sampleVars = [];

varAdditionOrder = [];

operation = 0;
while length( sampleOrder ) < nCpm
    
    operation = operation + 1;
    
    cpmProductInd = arrayfun( @(x) isempty( setdiff( x.variables( (x.numChild+1):end ), sampleVars ) ), cpmArrayLoop );
    cpmProductInd = find( cpmProductInd );
    if isempty( cpmProductInd )
        error( 'Given CPMs include undefined parent nodes' )
    else
        cpmProductInd = cpmProductInd(1);
    end
    
    sampleOrder = [sampleOrder cpmIndArray(cpmProductInd)];
    
    cpmProduct = cpmArrayLoop( cpmProductInd );
    varsToBeProduct = cpmProduct.variables( 1:cpmProduct.numChild );
    if ~isempty( intersect( sampleVars,varsToBeProduct ) )
        error( 'Given Cpms must not have common child nodes' )
    else
        sampleVars = [sampleVars varsToBeProduct];
    end
    
    varAdditionOrder = [varAdditionOrder operation*ones( size( varsToBeProduct ) )];
    
    cpmArrayLoop( cpmProductInd ) = [];
    cpmIndArray( cpmProductInd ) = [];
   
end

end

function [sample, sampleProb] = singleSample( cpmArray, sampleOrder, sampleVars, varAdditionOrder, vars, sampleInd )

nCpm = length( cpmArray );
nVars = length( sampleVars );
sample = zeros(1, nVars);
logSampleProb = log(1);

for operation = 1:nCpm
    cpmInd = sampleOrder( operation );
    cpmToSample = cpmArray( cpmInd );
    cpmToSample = condition( cpmToSample, sampleVars( sample>0 ), sample( sample>0 ), vars, sampleInd );

    if ( sampleInd == 1 ) && ( sum( cpmToSample.p ) ~= 1 )
        warning( 'Given probability vector does not sum to 1' )
    end
%     iSampleRowInd = randsample( length(cpmToSample.p), 1, true, cpmToSample.p );
    iP_cumsum = cumsum(cpmToSample.p); iRand = rand( 1 ); % To not use randsample (because of a toolkit issue)
    iSampleRowInd = find( iP_cumsum > iRand, 1 );
    iSample = cpmToSample.C( iSampleRowInd,1:cpmToSample.numChild );
    logSampleProb = logSampleProb + log( cpmToSample.p( iSampleRowInd ) );

    sample( varAdditionOrder == operation ) = iSample;             
end

sampleProb = exp( logSampleProb );
    
end