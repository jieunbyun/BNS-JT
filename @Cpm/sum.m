function Msum = sum(M,sumVars,sumFlag)
% sumFlag: 1 (default) - sum out sumVars; 0 - leave only sumVars

if length(M) > 1
    error('M must be a single CPM')
end
if nargin < 3
    sumFlag = 1;
end

if sumFlag && ~isempty( intersect( M.variables(M.numChild+1:end),sumVars ) )
    warning('Parent nodes are NOT summed up')
end

if sumFlag
    [varsRemain,varsRemainIdx] = setdiff(M.variables(1:M.numChild),sumVars,'stable');
    varsRemainIdx = varsRemainIdx(:)';
else
    [~,varsRemainIdx] = ismember( sumVars,M.variables(1:M.numChild) );
    varsRemainIdx = varsRemainIdx(varsRemainIdx>0);
    varsRemainIdx = sort(varsRemainIdx(:)');
    varsRemain = M.variables( varsRemainIdx );
end
numChild = length( varsRemain );

if ~isempty( M.variables(M.numChild+1:end) )
    varsRemain = [varsRemain M.variables(M.numChild+1:end)];
    varsRemainIdx = [varsRemainIdx (M.numChild+1):length(M.variables)];
end

Mloop = Cpm( M.variables( varsRemainIdx ), length( varsRemainIdx ), M.C(:,varsRemainIdx), M.p, M.q, M.sampleIndex );
if isempty(Mloop.C)
    Msum = Cpm(varsRemain, numChild, zeros(1,0), sum(M.p), [], [] ); 
else
    Csum = []; psum = []; qsum = []; sampleIndSum = [];
    while ~isempty(Mloop.C)
            
        Mcompare = getCpmSubset( Mloop, 1 );
        compatFlag = isComplatibleCpm( Mloop, Mcompare );
     
        Csum = [Csum; Mloop.C(1,:)];
        
        if ~isempty(Mloop.p)
            psum = [psum; sum(Mloop.p(compatFlag))];
        end
        
        if ~isempty(Mloop.q)
            if ~all( Mloop.q( compatFlag ) == Mloop.q(1) )
                error( 'Compatible samples cannot have different weights' )
            else
                qsum = [qsum; Mloop.q(1)];
            end
        end
            
        if ~isempty( Mloop.sampleIndex )
            sampleIndSum = [sampleIndSum; Mloop.sampleIndex(1)];
        end
        
        Mloop = getCpmSubset( Mloop, find( compatFlag ), 0 );

    end
    
    Msum = Cpm( varsRemain, numChild, Csum, psum, qsum, sampleIndSum );
end


    

