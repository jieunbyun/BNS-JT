function [cliques_sensitivity_conditioned, messages_sensitivity_conditioned, vars] = runJunctionTree_sensitivity_conditioning(jtree, sensitiveCliqueIndArray, sensitiveCliqueFunCell, conditionedCpms)

if length( sensitiveCliqueIndArray ) ~= length( sensitiveCliqueFunCell )
    error( 'Array of sensitive clique indcies must have the same length with given functions cell' )
end

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
nSensitiveClique = length( sensitiveCliqueIndArray );

cliques_sensitivity_conditioned = {};
messages_sensitivity_conditioned = {};
for sensRound = 1:nSensitiveClique
    
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
        for messInd = 1:nMessage
            iMessSender = messSched(messInd,messSender);
            incomingMessInd = find( messSched(:,messReceiver) == iMessSender );
            if any( incomingMessInd > messInd )
                error( 'A clique can send message only after all incoming messages are received' )
            end

            iMess = {};
            for jj = 1:length( incomingMessInd )
                iMess = [iMess; messages( incomingMessInd(jj) )];
            end

            iMessReceiver = messSched(messInd,messReceiver);
            if messInd ~= sensitiveCliqueIndArray( sensRound )
                if iMessReceiver
                    [ikCliques, messages{messInd}, vars] = cliqueFun{messInd}( cliques{iMessSender}, iMess, vars );
                    ikMessages = messages{messInd};
                else % no sending message
                    [ikCliques, ~, vars] = cliqueFun{messInd}( cliques{iMessSender}, iMess, vars );
                    ikMessages = [];
                end
            else
                if iMessReceiver
                    [ikCliques, messages{messInd}, vars] = sensitiveCliqueFunCell{sensRound}( cliques{iMessSender}, iMess, vars );
                    ikMessages = messages{messInd};
                else % no sending message
                    [ikCliques, ~, vars] = sensitiveCliqueFunCell{sensRound}( cliques{iMessSender}, iMess, vars );
                    ikMessages = [];
                end
            end
            
            
            for jj = 1:length( ikCliques )
                [ikCliques(jj), vars] = product( ikCliques(jj), iCondCpm, vars );
                ikCliques(jj) = sum(ikCliques(jj), condVarArray );
            end
            
            if ( sensRound == 1 ) && ( condRound == 1 )
                cliques_sensitivity_conditioned{iMessSender} = [];
                messages_sensitivity_conditioned{messInd} = [];
            end
            
            if isempty( cliques_sensitivity_conditioned{iMessSender} )
                cliques_sensitivity_conditioned{iMessSender} = ikCliques;
            else
                cliques_sensitivity_conditioned{iMessSender} = cliques_sensitivity_conditioned{iMessSender} + ikCliques;
            end
            
            if ~isempty( ikMessages )
                for jj = 1:length( ikMessages )
                    [ikMessages(jj), vars] = product( ikMessages(jj), iCondCpm, vars );
                    ikMessages(jj) = sum(ikMessages(jj), condVarArray );
                end
                
                if isempty( messages_sensitivity_conditioned{messInd} )
                    messages_sensitivity_conditioned{messInd} = ikMessages;
                else
                    messages_sensitivity_conditioned{messInd} = messages_sensitivity_conditioned{messInd} + ikMessages;
                end
            end
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