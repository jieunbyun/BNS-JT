%{
MCS for comparison
%}
clear;
rng(1)
import kN.*

load( 'ex1_1_commonEnvironment_systemFailureProb.mat', 'M', 'vars', 'VAR_WEATHER', 'VAR_VISIBILITY', 'VAR_TEMPERATURE', ...
    'STATE_SURVIVE', 'nComponentEachTypeArray', 'LAMBDA', 'K', 'nComponentType', 'systemFailureProbArray' )

nNCompToAnalyze = 7;
VAR_COMP1 = VAR_TEMPERATURE + 1;

pComp = zeros( length( M( VAR_COMP1 ).p ), nComponentType );
for iNCompType = 1:nComponentType
    
    iLambda = LAMBDA( iNCompType, : );
    iPComp = computeProbVectorFromLambda( iLambda );
    pComp( :, iNCompType ) = iPComp;
end


targetCov = 5e-2;
nSampleRequired = ceil( (1-systemFailureProbArray)./systemFailureProbArray/targetCov^2 );
sampleMeanArray = zeros( nNCompToAnalyze, 1 ); sampleCovArray = zeros( nNCompToAnalyze, 1 ); sampleNumberArray = zeros( nNCompToAnalyze, 1 );
for iNCompIndex = 1:nNCompToAnalyze
    
    iNCompEachType = nComponentEachTypeArray( iNCompIndex );
    iNComp = iNCompEachType * nComponentType;
        
    iFailProbCov = 1; iNFail = 0; iNSample = 0;
    for jSampleInd = 1:nSampleRequired( iNCompIndex )
        
        iNSample = iNSample + 1;
        
        ijW = randsample( M( VAR_WEATHER ).C, 1, true, M( VAR_WEATHER ).p );

        ijVRowIndex = ( M( VAR_VISIBILITY ).C( :,2 ) == ijW );
        if sum( ijVRowIndex ) > 1
            ijVC = M( VAR_VISIBILITY ).C( ijVRowIndex, 1 );
            ijVp = M( VAR_VISIBILITY ).p( ijVRowIndex );
            ijV = randsample( ijVC, 1, true, ijVp );
        else
            ijV = M( VAR_VISIBILITY ).C( ijVRowIndex, 1 );
        end

        ijT = randsample( M( VAR_TEMPERATURE ).C, 1, true, M( VAR_TEMPERATURE ).p );


        ijXRowIndex = ismember( M( VAR_COMP1 ).C( :, 2:4 ), [ijW ijV ijT], 'rows' );
        ijXC = M( VAR_COMP1 ).C( ijXRowIndex, 1 );

        ijX = zeros( iNComp, 1 );
        for kCompTypeInd = 1:nComponentType
            ijkXp = pComp( ijXRowIndex, kCompTypeInd);
            ijkXArray = randsample( ijXC, iNCompEachType, true, ijkXp );
            
            ijX( iNCompEachType*(kCompTypeInd-1) + (1:iNCompEachType) ) = ijkXArray;
        end
        
        if sum( ijX == STATE_SURVIVE ) < K
            iNFail = iNFail + 1;
        end
        
        iFailProbMean = iNFail / iNSample;
        iFailProbStd = sqrt( (1-iFailProbMean) * iFailProbMean / iNSample );
        iFailProbCov = iFailProbStd / iFailProbMean;
    
        if ~rem( iNSample, 1e3 )
            disp( ['[' num2str(iNCompIndex) '-th Comp, No.] Sample no.: ' num2str( iNSample ) '; Sample mean: ' num2str( iFailProbMean ) '; c.o.v.: ' num2str( iFailProbCov ) ] )
        end
    end


    sampleMeanArray( iNCompIndex ) = iFailProbMean;
    sampleCovArray( iNCompIndex ) = iFailProbCov;
    sampleNumberArray( iNCompIndex ) = iNSample;
    
    disp( ['[' num2str(iNCompIndex) '-th Comp, No.] Sample no.: ' num2str( iNSample ) '; Sample mean: ' num2str( iFailProbMean ) '; c.o.v.: ' num2str( iFailProbCov ) ] )
end
    
save ex1_3_commonEnvironment_systemFailureProb_MCS

