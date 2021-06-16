function Msorted = sort(M)

Msorted = M;

if ~isempty( M.sampleIndex )
    [~,rowSortIdx] = sort(M.sampleIndex);
else
    [~,rowSortIdx] = sortrows(M.C(:,end:-1:1));
end
    
    
Msorted.C = M.C(rowSortIdx,: );
if ~isempty( M.p ), Msorted.p = M.p(rowSortIdx); end
if ~isempty( M.q ), Msorted.q = M.q(rowSortIdx); end
if ~isempty( M.sampleIndex ), Msorted.sampleIndex = M.sampleIndex(rowSortIdx); end