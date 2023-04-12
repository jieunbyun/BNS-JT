function prob = getProb( M, varInds, varStates, vars, getFlag )

if length(M)~= 1; error( 'Given CPM must be a single CPM array' ); end
if length(varInds) ~= length(varStates); error( '"varInds" and "varStates" must have the same length.' ); end

KEEP_ROW_INDEX = 1; DELETE_ROW_INDEX = 0;
if nargin < 5
    getFlag = KEEP_ROW_INDEX;
elseif ~ismember( getFlag, [KEEP_ROW_INDEX, DELETE_ROW_INDEX] )
    error( 'Operation flag must be either 1 (keeping given row indices; default) or 0 (deleting given indices)' )
end


Mcompare = Cpm( varInds, length(varInds), varStates );
compatFlag = isComplatibleCpm( M, Mcompare, vars );
Msubset = getCpmSubset( M, find(compatFlag), getFlag );
prob = sum( Msubset.p );