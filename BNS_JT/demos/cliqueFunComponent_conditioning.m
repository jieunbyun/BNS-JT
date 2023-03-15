function [cliqueComponent1, messToSystem, vars] = cliqueFunComponent_conditioning( cliqueComponent1, messToComponent1, vars, VAR_FAULT_INTENSITY1, VAR_COMPONENT1 )

STATE_SURVIVE = 1; STATE_FAIL = 2;
COMPONENT = 1; FAULT_INTENSITY = 2;


numRules = size( cliqueComponent1.C, 1 );
p = zeros( numRules, 1 );
paramValues = vars( VAR_FAULT_INTENSITY1 ).v;

for ruleInd = 1:numRules
    iRule = cliqueComponent1.C( ruleInd, : );
    iParam = paramValues( iRule( FAULT_INTENSITY ) );

    if iRule( COMPONENT ) == STATE_SURVIVE
        p( ruleInd ) = exp( -iParam );
    else
        p( ruleInd ) = 1 - exp( -iParam );
    end
end

cliqueComponent1.p = p;


messToComponent1 = messToComponent1{1};
[cliqueComponent1, vars] = product( cliqueComponent1, messToComponent1, vars );
messToSystem = sum( cliqueComponent1, VAR_COMPONENT1, 0 );