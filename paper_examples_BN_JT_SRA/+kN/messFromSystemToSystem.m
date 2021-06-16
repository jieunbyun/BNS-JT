function [cliqueSystem, messToNextSystem, vars] = messFromSystemToSystem( cliqueSystem, messToSystem, vars, VAR_NEXT_SYSTEM, VAR_CONDITIONED )

import kN.*
if ~isempty( messToSystem ) 
    messToSystem = messToSystem{1};
end

[cliqueSystem, vars] = multCPMs( [cliqueSystem(:); messToSystem(:)], vars );

messToNextSystem = sum(cliqueSystem, [VAR_NEXT_SYSTEM(:)' VAR_CONDITIONED(:)'], 0);