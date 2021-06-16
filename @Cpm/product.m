function [Mproduct,vInfo] = product(M1,M2,vInfo)

if ~isempty( intersect( M1.variables(1:M1.numChild),M2.variables(1:M2.numChild) ) )
    error('PMFs must not have common child nodes')
end

if ~isempty( M1.p )
    if isempty( M2.p )
        M2.p = ones( size(M2.C,1),1 );
    end
else
    if ~isempty( M2.p )
        M1.p = ones( size(M1.C,1),1 );
    end
end

if ~isempty( M1.q )
    if isempty( M2.q )
        M2.q = ones( size(M2.C,1),1 );
    end
else
    if ~isempty( M2.q )
        M1.q = ones( size(M1.C,1),1 );
    end
end


if size(M1.C,2) > size(M2.C,2)
    M1_ = M1;
    M1 = M2;
    M2 = M1_;
end

if ~isempty(M1.C)
    commonVars=intersect(M1.variables,M2.variables);

    flagCommonVarsInM1 = ismember(M1.variables,M2.variables);
    commonVars = M1.variables(flagCommonVarsInM1);
    
    C1 = M1.C; p1 = M1.p; q1 = M1.q;
    sampleInd1 = M1.sampleIndex;
    Cproduct = []; pproduct = []; qproduct = []; sampleIndProduct = [];
    for rr = 1:size(C1,1)
        c1_r = C1(rr,flagCommonVarsInM1);
        c1_notCommon_r = C1(rr,~flagCommonVarsInM1);
        
        if ~isempty( sampleInd1 )
            sampleInd1_r = sampleInd1(rr);
        else
            sampleInd1_r = [];
        end

        [M2_r,vInfo] = condition(M2, commonVars, c1_r, vInfo, sampleInd1_r);
        Cproduct = [Cproduct; M2_r.C repmat(c1_notCommon_r, size(M2_r.C,1), 1)];
        
        if ~isempty( sampleInd1_r )
            sampleIndProduct = [sampleIndProduct; repmat(sampleInd1_r, size(M2_r.C,1), 1)];
        elseif ~isempty( M2_r.sampleIndex )
            sampleIndProduct = [sampleIndProduct; M2_r.sampleIndex];
        end
        
        if ~isempty( p1 )
            pproductSign_r = sign( M2_r.p * p1(rr) );
            pproductVal_r = exp( log( abs(M2_r.p) )+log( abs(p1(rr)) ) );
            pproduct = [pproduct; pproductSign_r .* pproductVal_r];
        end
        
        if ~isempty( q1 )
            qproductSign_r = sign( M2_r.q * q1(rr) );
            qproductVal_r = exp( log( abs(M2_r.q) )+log( abs(q1(rr)) ) );
            qproduct = [qproduct; qproductSign_r .* qproductVal_r];
        end
         
    end

    Cproduct_vars = [M2.variables M1.variables(~flagCommonVarsInM1)];

    newVarsChild = [M1.variables(1:M1.numChild) M2.variables(1:M2.numChild)];
    newVarsChild = sort(newVarsChild);
    newVarsParent = [M1.variables(M1.numChild+1:end) M2.variables(M2.numChild+1:end)];
    newVarsParent = setdiff(newVarsParent,newVarsChild);
    newVars = [newVarsChild newVarsParent];

    [~,idxVars] = ismember(newVars,Cproduct_vars);
    Cproduct = Cproduct(:,idxVars);

    Mproduct = Cpm;
    Mproduct.variables = newVars;
    Mproduct.numChild = length(newVarsChild);
    Mproduct.C = Cproduct; Mproduct.p = pproduct; Mproduct.q = qproduct;
    Mproduct.sampleIndex = sampleIndProduct;
    
    Mproduct = sort(Mproduct);
else
    Mproduct = M2;
end

