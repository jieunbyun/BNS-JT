function compatFlag = isComplatibleCpm( M, Mcompare, vars )

if length(M) ~= 1, error( 'Given CPM must be a single array of Cpm' ); end
if size(Mcompare.C,1) ~= 1, error( 'Given CPM to compare must include only a single row' ); end

if nargin < 3
    isCompositeStateConsidered = 0;
else
    isCompositeStateConsidered = 1;
end

compareVars = Mcompare.variables;
compareStates = Mcompare.C;

C = M.C;

[~,idxInCompareVars] = ismember(compareVars, M.variables);
compareVars(~idxInCompareVars) = [];
compareStates(~idxInCompareVars) = [];
idxInCompareVars(~idxInCompareVars) = [];

C_common = [];
for vv = 1:length(compareVars)
   C_common = [C_common C(:,idxInCompareVars(vv))];
end

if ~isempty( Mcompare.sampleIndex ) && ~isempty( M.sampleIndex )
    compatFlag = ( M.sampleIndex == Mcompare.sampleIndex );
else    
    compatFlag = true( size(C,1),1 );
end


for vv = 1:length(compareVars)
    compareVar_v = compareVars(vv);
    compareState_v = compareStates(vv);
    
    C_v = C_common(:,vv);
    if isCompositeStateConsidered
        B_v = vars(compareVar_v).B;        
    else
        B_v = eye( max(C_v) );
    end
    
    compatCheck_v = B_v(C_v(compatFlag),:) .* B_v( compareState_v,: );
    compatFlag(compatFlag) = ( sum(compatCheck_v,2)>0 );    
end