function pComponent = computeProbVectorFromLambda( lambdaArray )

reliability = exp( -lambdaArray(:) );
failureProb = 1 - reliability;


STATE_SURVIVE = 1; STATE_FAIL = 2;

pComponent = zeros( size( [reliability; failureProb] ) );
pComponent( STATE_SURVIVE:2:end ) = reliability;
pComponent( STATE_FAIL:2:end ) = failureProb;