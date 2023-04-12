function [cpm1, vars] = VE( cpms, varElimOrder, vars )

cpms = cpms(:);
for iVarInd_ = 1:length(varElimOrder)
    iVarInd = varElimOrder(iVarInd_);
    iIsVarInScope = isXinScope( iVarInd, cpms );

    [iMMult, vars] = multCPMs(cpms(iIsVarInScope), vars);
    iMMult = sum( iMMult, iVarInd );

    cpms(iIsVarInScope) = [];
    cpms = [iMMult; cpms];
end

[cpm1, vars] = multCPMs(cpms, vars);