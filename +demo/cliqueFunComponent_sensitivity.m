function [cliqueComponent, messToSystem, vars] = cliqueFunComponent_sensitivity( cliqueComponent, messToComponent, vars, VAR_FAULT_INTENSITY, VAR_COMPONENT, sensitivity_compInd, sensitivity_paramInd )

numComponent = length( cliqueComponent );
STATE_SURVIVE = 1; STATE_FAIL = 2;
COMPONENT = 1; FAULT_INTENSITY = 2;

for compInd = 1:numComponent
    
    iClique = cliqueComponent( compInd );
    iNumRules = size( iClique.C, 1 );
    ip = zeros( iNumRules, 1 );
    iParamValues = vars( VAR_FAULT_INTENSITY(compInd) ).v;
    
    if compInd == sensitivity_compInd
        for ruleInd = 1:iNumRules
            ijRule = iClique.C( ruleInd, : );
            ijParam = iParamValues( ijRule( FAULT_INTENSITY ) );
            
            if ijRule( FAULT_INTENSITY ) == sensitivity_paramInd
                if ijRule( COMPONENT ) == STATE_SURVIVE
                    ip( ruleInd ) = -ijParam * exp( -ijParam );
                else
                    ip( ruleInd ) = ijParam * exp( -ijParam );
                end
            end
        end
        
    else
        for ruleInd = 1:iNumRules
            ijRule = iClique.C( ruleInd, : );
            ijParam = iParamValues( ijRule( FAULT_INTENSITY ) );

            if ijRule( COMPONENT ) == STATE_SURVIVE
                ip( ruleInd ) = exp( -ijParam );
            else
                ip( ruleInd ) = 1 - exp( -ijParam );
            end
        end
    end
    
    cliqueComponent( compInd ).p = ip;
end


messToComponent = messToComponent{1};
[cliqueComponent, vars] = multCPMs( cliqueComponent, vars );
[cliqueComponent, vars] = product( cliqueComponent, messToComponent, vars );
messToSystem = sum( cliqueComponent, VAR_COMPONENT, 0 );