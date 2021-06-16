function [cliqueComponent, messToSystem, vars] = messFromComponentToSystem_sensitivity( cliqueComponent, messToComponent, vars, faultIntensityVarInd, reliabilityVarInd, operatingTime, operationStartTime, faultObservationTimes, beta0 )

import kN.*

[alpha, beta] = mleOfFaultIntensityParameters( faultObservationTimes, beta0 );

faultIntensity = alpha * beta *exp( -beta*operationStartTime );
vars( faultIntensityVarInd ).v = faultIntensity;

reliability = exp( -faultIntensity * operatingTime );
vars( reliabilityVarInd ).v = reliability;

dReliabilityDFaultIntensity = -operatingTime * reliability;

cliqueComponent(3).p = [1; -1] * dReliabilityDFaultIntensity;
messToSystem = cliqueComponent(3);

cliqueComponent = cliqueComponent(3);