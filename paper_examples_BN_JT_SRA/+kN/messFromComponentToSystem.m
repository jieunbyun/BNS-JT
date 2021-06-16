function [cliqueComponent, messToSystem, vars] = messFromComponentToSystem( cliqueComponent, messToComponent, vars, faultIntensityVarInd, reliabilityVarInd, operatingTime, operationStartTime, faultObservationTimes, beta0 )

import kN.*

[alpha, beta] = mleOfFaultIntensityParameters( faultObservationTimes, beta0 );

faultIntensity = alpha * beta *exp( -beta*operationStartTime );
vars( faultIntensityVarInd ).v = faultIntensity;

reliability = exp( -faultIntensity * operatingTime );
vars( reliabilityVarInd ).v = reliability;

cliqueComponent(3).p = [reliability; 1-reliability];
messToSystem = cliqueComponent(3);

cliqueComponent = cliqueComponent(3);
