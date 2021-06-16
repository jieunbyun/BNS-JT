function [CpmArray, vars] = createCompositeStates(varIndex, CpmArray, vars)

if ~isscalar(varIndex), error('var must be a single variable'); end
if any( arrayfun( @(x) ~isempty(x.q), CpmArray ) )
    warning( 'This operation does not consider q vector (sample weight. q vector is returned empty' )
end

B = vars(varIndex).B;
nStates = size(B,2);
for ii = 1:length(CpmArray)
    iCpm = CpmArray(ii);
    
    iVars = iCpm.variables;
    [~,iVarIndInC] = intersect(iVars, varIndex);
        
    iC_old = iCpm.C;
    i_nVars = length(iVars);
    iOtherVarsIndInC = setdiff(1:i_nVars, iVarIndInC);
    ip_old = iCpm.p;

    iCnew = []; ipnew = [];
    while ~isempty(iC_old)

        rowsToBeMerged = ismember([iC_old(:,iOtherVarsIndInC) ip_old], [iC_old(1,iOtherVarsIndInC) ip_old(1)], 'rows');
        
        statesToBeMerged = iC_old( find(rowsToBeMerged), iVarIndInC );
        basicStatesToBeMerged = [];
        for jj = 1:length( statesToBeMerged )
            basicStatesToBeMerged = [basicStatesToBeMerged find(B(statesToBeMerged(jj),:))];
        end
        basicStatesToBeMerged = unique( basicStatesToBeMerged );
        
        if length(basicStatesToBeMerged) > 1
            compositeState = zeros(1,nStates);
            compositeState( basicStatesToBeMerged ) = 1;
            
            stateInd = ismember(B, compositeState, 'rows');
            if any( stateInd )
                stateInd = find( stateInd );
            else
                B = [B; compositeState];
                stateInd = size(B,1);
            end
        else
            stateInd = basicStatesToBeMerged;
        end
        
        newRow = zeros(1,i_nVars);
        newRow(iOtherVarsIndInC) = iC_old(1,iOtherVarsIndInC);
        newRow(iVarIndInC) = stateInd;
        
        iCnew = [iCnew; newRow];
        ipnew = [ipnew; ip_old(1)];
        
        iC_old(rowsToBeMerged,:) = [];
        ip_old(rowsToBeMerged,:) = [];
        
    end
    
    CpmArray(ii).C = iCnew;
    CpmArray(ii).p = ipnew;
    CpmArray(ii).q = [];
end

vars(varIndex).B = B;