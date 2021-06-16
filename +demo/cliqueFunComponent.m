function [cliqueComponent, messToSystem, vars] = cliqueFunComponent( cliqueComponent, messToComponent, vars, VAR_FAULT_INTENSITY, VAR_COMPONENT )

numComponent = length( cliqueComponent );
STATE_SURVIVE = 1; STATE_FAIL = 2;
COMPONENT = 1; FAULT_INTENSITY = 2;

for compInd = 1:numComponent
    iClique = cliqueComponent( compInd );
    iNumRules = size( iClique.C, 1 );
    ip = zeros( iNumRules, 1 );
    iParamValues = vars( VAR_FAULT_INTENSITY(compInd) ).v;
    
    for ruleInd = 1:iNumRules
        ijRule = iClique.C( ruleInd, : );
        ijParam = iParamValues( ijRule( FAULT_INTENSITY ) );
        
        if ijRule( COMPONENT ) == STATE_SURVIVE
            ip( ruleInd ) = exp( -ijParam );
        else
            ip( ruleInd ) = 1 - exp( -ijParam );
        end
    end
    
    cliqueComponent( compInd ).p = ip;
end

messToComponent = messToComponent{1};
[cliqueComponent, vars] = multCPMs( cliqueComponent, vars );
[cliqueComponent, vars] = product( cliqueComponent, messToComponent, vars );
messToSystem = sum( cliqueComponent, VAR_COMPONENT, 0 );