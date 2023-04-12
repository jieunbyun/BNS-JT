function nextComp = bnb_nextCompFun( candComps, downRes, upRes, info )

arcPath1 = info.arcPath1;
arcPaths_time1 = info.arcPaths_time1;
nArc = info.nArc;

[~,sortInd] = sort(arcPaths_time1,'ascend');
arcPath1 = arcPath1(sortInd);

compsOrder = [];
for iPathInd = 1:length( arcPath1 )
    iComps = arcPath1{iPathInd};
    compsOrder = union(compsOrder, iComps, 'stable'); % Do not change the order so that components on shorter paths come forward.
end
compsOrder = [compsOrder setdiff(1:nArc, compsOrder)];

nextComp_ = find( ismember( candComps, compsOrder ) , 1 );
nextComp = candComps(nextComp_);
