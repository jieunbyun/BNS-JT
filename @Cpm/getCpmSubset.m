function Msubset = getCpmSubset( M, rowIndex, getFlag )

if length(M)~= 1; error( 'Given CPM must be a single CPM array' ); end

KEEP_ROW_INDEX = 1; DELETE_ROW_INDEX = 0;
if nargin < 3
    getFlag = KEEP_ROW_INDEX;
end

switch getFlag
    case KEEP_ROW_INDEX
        rowIndex = rowIndex;
    case DELETE_ROW_INDEX
        rowIndex = setdiff( (1:size(M.C,1))', rowIndex );
    otherwise
        error( 'Operation flag must be either 1 (keeping given row indices; default) or 0 (deleting given indices)' )
end


Csubset = M.C(rowIndex,:);

if ~isempty( M.p )
    pSubset = M.p( rowIndex );
else
    pSubset = [];
end

if ~isempty( M.q )
    qSubset = M.q( rowIndex );
else
    qSubset = [];
end

if ~isempty( M.sampleIndex )
    sampleIndSubset = M.sampleIndex( rowIndex );
else
    sampleIndSubset = [];
end

Msubset = Cpm( M.variables, M.numChild, Csubset, pSubset, qSubset, sampleIndSubset );