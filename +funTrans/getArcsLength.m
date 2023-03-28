function arcLens = getArcsLength( arcs, nodeCoords )

if size(arcs,2) == 2
    nArc = size(arcs,1);
elseif size(arcs,1) == 2
    arcs = arcs';
    nArc = size(arcs,1);
else
    error( '"arcs" must have either two columns or two rows (each noting start and end points).' )
end

if size(nodeCoords,2) ~= 2
    if size(nodeCoords,1) == 2
        nodeCoords = nodeCoords';
    else 
        error( '"nodeCoords" must have either two columns or two rows (each noting coordinates of x and y).' )
    end
end

arcLens = ( nodeCoords(arcs(:,1), : ) - nodeCoords(arcs(:,2), : ) ).^2;
arcLens = sum(arcLens,2);
arcLens = sqrt(arcLens);