function [cliqueEnvironment_conditioned, messageToComp1, vars] = messFromEnvironmentToComp1( cliqueEnvironment, messageToEnvironment, vars, instanceInd, VAR_ENVIRONMENT )

if length( cliqueEnvironment ) > 1
    [cliqueEnvironment, vars] = multCPMs( cliqueEnvironment, vars );
end

[cliqueEnvironment_conditioned, vars] = condition( cliqueEnvironment, cliqueEnvironment.variables, cliqueEnvironment.C(instanceInd,:), vars );
messageToComp1 = cliqueEnvironment_conditioned;