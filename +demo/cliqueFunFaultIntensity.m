function [cliqueFaultIntensity, messToComponent, vars] = cliqueFunFaultIntensity( cliqueFaultIntensity, messToFaultIntensity, vars, VAR_FAULT_INTENSITY )

messToFaultIntensity = messToFaultIntensity{1};
[cliqueFaultIntensity, vars] = multCPMs( cliqueFaultIntensity, vars );
[cliqueFaultIntensity, vars] = product( cliqueFaultIntensity, messToFaultIntensity, vars );
messToComponent = sum( cliqueFaultIntensity, VAR_FAULT_INTENSITY, 0 );