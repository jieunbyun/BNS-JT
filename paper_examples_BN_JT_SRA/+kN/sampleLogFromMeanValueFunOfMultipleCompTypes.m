function faultObservationTimes = sampleLogFromMeanValueFunOfMultipleCompTypes(logMeanModelParameter_a, logMeanModelParameter_k, simulatedTime)

nComponentType = length( logMeanModelParameter_a );


faultObservationTimes = {};
for ii = 1:nComponentType
    ia = logMeanModelParameter_a(ii);
    ik = logMeanModelParameter_k(ii);
    
    faultObservationTimes{ii,1} = sampleLogFromMeanValueFun(ia, ik, simulatedTime);
end

end

function faultTimeSamples = sampleLogFromMeanValueFun(a, k, simulatedTime)

meanValueInverseFun = @( meanValue ) 1/a * ( exp(meanValue/k) - 1 );

loopflag = 1;
while loopflag    
    
    HPPsamplePoint = 0;
    faultTimeSamples = [];

    NHPPsamplePoint = 0;
    while NHPPsamplePoint < simulatedTime
       HPPsamplePoint = HPPsamplePoint + exprnd( 1 );

       NHPPsamplePoint = meanValueInverseFun( HPPsamplePoint );

       faultTimeSamples = [faultTimeSamples; NHPPsamplePoint];

       if isnan( NHPPsamplePoint ), break; end

    end
    
    if ~isnan( NHPPsamplePoint ) && NHPPsamplePoint > simulatedTime, break; end
    
end
end