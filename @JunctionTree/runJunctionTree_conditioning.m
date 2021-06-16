function [cliques_conditioned, messages_conditioned, vars] = runJunctionTree_conditioning(jtree, conditionedCpms)

cliqueFun = jtree.cliqueFunctionCell;
cpmInd = jtree.cpmInd;
messSched = jtree.messageSchedule;
vars = jtree.variables;
M = jtree.cpm;


global nSample
nSample = jtree.nSample;

messSender = 1; messReceiver = 2;

condVarArray = arrayfun( @(x) x.variables, conditionedCpms(:)', 'UniformOutput', false );
condVarArray = cell2mat( condVarArray );
nConditionedInstanceArray = arrayfun( @(x) size(x.C, 1), conditionedCpms );
nTotalConditionedInstance = prod( nConditionedInstanceArray );
nCondCpms = length( conditionedCpms );


nMessage = size( messSched, 1 );
cliques_conditioned = {};
messages_conditioned = {};
for condRound = 1:nTotalConditionedInstance
    
    iCondState = getCondStateFromInstanceIndex( condRound, nConditionedInstanceArray );
    iCondCpm = Cpm;
    for ii = 1:nCondCpms
        iiCondCpm = conditionedCpms(ii);
        [iiCondCpm, vars] = condition( iiCondCpm, iiCondCpm.variables, iiCondCpm.C( iCondState(ii),: ), vars );
        [iCondCpm, vars ] = product( iCondCpm, iiCondCpm, vars );
    end
    
    cliques = {};
    for ii = 1:length( cpmInd )
        cliques{ii} = M( cpmInd{ii} );
        [cliques{ii}, vars] = condition( cliques{ii}, iCondCpm.variables, iCondCpm.C, vars, iCondCpm.sampleIndex );
    end
    
    messages = {};
    for messageInd = 1:nMessage
        iMessSender = messSched(messageInd, messSender);
        incomingMessInd = find( messSched(:, messReceiver) == iMessSender );
        if any( incomingMessInd > messageInd )
            error( 'A clique can send message only after all incoming messages are received' )
        end        
        
        iMess = {};
        for jj = 1:length( incomingMessInd )
            iMess = [iMess; messages( incomingMessInd(jj) )];
        end
        
        iMessReceiver = messSched(messageInd, messReceiver);
        if iMessReceiver
            [cliques{iMessSender}, messages{messageInd}, vars] = cliqueFun{messageInd}( cliques{iMessSender}, iMess, vars );
        else % no sending message
            [cliques{iMessSender}, ~, vars] = cliqueFun{messageInd}( cliques{iMessSender}, iMess, vars );
        end
    end
    
    for ii = 1:length( cliques )
        for jj = 1:length( cliques{ii} )
            [cliques{ii}(jj), vars] = product( cliques{ii}(jj), iCondCpm, vars );
            cliques{ii}(jj) = sum( cliques{ii}(jj), condVarArray );
        end
        
        if condRound == 1
            cliques_conditioned{ii} = cliques{ii};
        else
            cliques_conditioned{ii} = cliques_conditioned{ii} + cliques{ii};
        end
    end
    for ii = 1:length( messages )
        for jj = 1:length( messages{ii} )
            [messages{ii}(jj), vars] = product( messages{ii}(jj), iCondCpm, vars );
            messages{ii}(jj) = sum( messages{ii}(jj), condVarArray );
        end
        
        if condRound == 1
            messages_conditioned{ii} = messages{ii};
        else
            messages_conditioned{ii} = messages_conditioned{ii} + messages{ii};
        end
    end
        
    
end

end


function condState = getCondStateFromInstanceIndex( instanceIndex, nConditionedInstanceArray )

    nCondCliques = length( nConditionedInstanceArray );
    
    condState = instanceIndex * ones( 1,nCondCliques );
    for ii = 1:nCondCliques
        iNstate = nConditionedInstanceArray(ii);
        condState(ii) = rem( condState(ii) - 1, iNstate ) + 1;
        condState( (ii+1):end ) = floor( ( condState( (ii+1):end ) - 1 ) / iNstate ) + 1;
    end
end