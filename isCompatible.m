function compatFlag = isCompatible(C,vars,checkVars,checkStates,vInfo)


[~,idxInCheckVars] = ismember(checkVars,vars);
checkVars(~idxInCheckVars) = [];
checkStates(~idxInCheckVars) = [];
idxInCheckVars(~idxInCheckVars) = [];

C1_common = [];
for vv = 1:length(checkVars)
   C1_common = [C1_common C(:,idxInCheckVars(vv))];
end
    
compatFlag = true( size(C,1),1 );
for vv = 1:length(checkVars)
    checkVar_v = checkVars(vv);
    checkState_v = checkStates(vv);
    
    B_v = vInfo(checkVar_v).B;
    C1_v = C1_common(:,vv);
    
    compatCheck_v = B_v(C1_v(compatFlag),:) .* B_v( checkState_v,: );
    compatFlag(compatFlag) = ( sum(compatCheck_v,2)>0 );
    
end

end
