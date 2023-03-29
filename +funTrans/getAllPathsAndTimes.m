function [arcPaths_cell, pathTimes_cell] = getAllPathsAndTimes( ODs, G, arcTimes )

if size(ODs,2) ~= 2
    if size(ODs,2) ~= 1
        ODs = ODs';
    else
        error( '"ODs" must have either two columns or two rows (each noting nodes of origin and destination).' )
    end
end


nOD = size(ODs, 1);
arcPaths_cell = cell(nOD,1);
pathTimes_cell = cell(nOD,1);
for iODInd = 1:nOD
    iO = ODs(iODInd, 1);
    iD = ODs(iODInd, 2);

    [~, iArcPaths] = allpaths(G,iO,iD);
    iTime = cellfun( @(x) sum( arcTimes(x) ), iArcPaths, 'UniformOutput', true );

    [iTime, iPath_sortInd] = sort(iTime,'ascend');
    iArcPaths = iArcPaths(iPath_sortInd);

    arcPaths_cell{iODInd} = iArcPaths;
    pathTimes_cell{iODInd} = iTime;
end