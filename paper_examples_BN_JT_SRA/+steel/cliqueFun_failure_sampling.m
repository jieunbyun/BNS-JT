function [distribution, sendingMessage, vars] = cliqueFun_failure_sampling( distribution, receivedMessage, vars, ...
                                                                            VAR_FAILURE)

                                                                        
global nSample
TRUE = 1; FALSE = 2;

limitStateCpm = receivedMessage{1};

failureSample = zeros( size( limitStateCpm.C ) );
for time = 1:size( limitStateCpm.C, 2 )
    iFailureSample = any( limitStateCpm.C(:,1:time) <= 0, 2 )*1;
    iFailureSample( ~iFailureSample ) = FALSE;
    
    failureSample(:,time) = iFailureSample;
end

distribution = Cpm( VAR_FAILURE, length(VAR_FAILURE) );
distribution.C = failureSample;
distribution.p = limitStateCpm.p;
distribution.q = limitStateCpm.q;
distribution.sampleIndex = (1:nSample)';

sendingMessage = [];