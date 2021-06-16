function [cliques_sensitivity, messages_sensitivity, vars] = runJunctionTree_sensitivity(jtree, sensitiveCliqueIndArray, sensitiveCliqueFunCell)

if length( sensitiveCliqueIndArray ) ~= length( sensitiveCliqueFunCell )
    error( 'Array of sensitive clique indcies must have the same length with given functions cell' )
end

cliqueFun = jtree.cliqueFunctionCell;
cpmInd = jtree.cpmInd;
messSched = jtree.messageSchedule;
vars = jtree.variables;
M = jtree.cpm;


cliques = {};
for ii = 1:length( cpmInd )
    cliques{ii} = M( cpmInd{ii} );
end

global nSample
nSample = jtree.nSample;

messSender = 1; messReceiver = 2;

nMessage = size( messSched, 1 );
nSensitiveClique = length( sensitiveCliqueIndArray );

cliques_sensitivity = {};
messages_sensitivity = {};
for kk = 1:nSensitiveClique
    messages = {};
    for ii = 1:nMessage
        iMessSender = messSched(ii,messSender);
        incomingMessInd = find( messSched(:,messReceiver) == iMessSender );
        if any( incomingMessInd > ii )
            error( 'A clique can send message only after all incoming messages are received' )
        end

        iMess = {};
        for jj = 1:length( incomingMessInd )
            iMess = [iMess; messages( incomingMessInd(jj) )];
        end

        iMessReceiver = messSched(ii,messReceiver);
        if ii ~= sensitiveCliqueIndArray( kk )
            if iMessReceiver
                [ikCliques, messages{ii}, vars] = cliqueFun{ii}( cliques{iMessSender}, iMess, vars );
                ikMessages = messages{ii};
            else % no sending message
                [ikCliques, ~, vars] = cliqueFun{ii}( cliques{iMessSender}, iMess, vars );
                ikMessages = [];
            end
        else
            if iMessReceiver
                [ikCliques, messages{ii}, vars] = sensitiveCliqueFunCell{kk}( cliques{iMessSender}, iMess, vars );
                ikMessages = messages{ii};
            else % no sending message
                [ikCliques, ~, vars] = sensitiveCliqueFunCell{kk}( cliques{iMessSender}, iMess, vars );
                ikMessages = [];
            end
        end
        
        if kk == 1
            cliques_sensitivity{iMessSender} = ikCliques;
            if ~isempty( ikMessages ), messages_sensitivity{ii} = ikMessages; end
        else
            cliques_sensitivity{iMessSender} = cliques_sensitivity{iMessSender} + ikCliques;
            if ~isempty( ikMessages ), messages_sensitivity{ii} = messages_sensitivity{ii} + ikMessages; end
        end
        
    end
end