function [cliqueSystem, messToNextSystem, vars] = messFromSystemToSystem_sensitivity( cliqueSystem, messToSystem, vars, VAR_NEXT_SYSTEM, VAR_CONDITIONED, sensitivityInd_environmentState, paramValue )

import kN.*

CPM_COMPONENT = 1;
CpmComponent = cliqueSystem( CPM_COMPONENT );
pSensitive = zeros( size( CpmComponent.C,2 ), 1 );

STATE_SURVIVE = 1; STATE_FAIL = 2;

rowInd_environment_survive = ismember( CpmComponent.C, [STATE_SURVIVE sensitivityInd_environmentState], 'rows' );
pSensitive( rowInd_environment_survive ) = -paramValue * exp( - paramValue );

rowInd_environment_fail = ismember( CpmComponent.C, [STATE_FAIL sensitivityInd_environmentState], 'rows' );
pSensitive( rowInd_environment_fail ) = paramValue * exp( - paramValue );


cliqueSystem( CPM_COMPONENT ).p = pSensitive;
if ~isempty( messToSystem ); messToSystem = messToSystem{1}; end

[cliqueSystem, vars] = multCPMs( [cliqueSystem(:); messToSystem(:)], vars );
messToNextSystem = sum(cliqueSystem, [VAR_NEXT_SYSTEM(:)' VAR_CONDITIONED(:)'], 0);