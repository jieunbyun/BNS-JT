function [cliques, messages, vars] = runJunctionTree(jtree)

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
messages = {};
for ii = 1:nMessage
    iMessSender = messSched(ii,messSender);
    incomingMessInd = find( messSched(:,messReceiver) == iMessSender );
    if any( incomingMessInd > ii )
        error( 'A clique can send message only after all incoming messages are received' )
    end
    
    iMess = {};
    for jj = 1:length( incomingMessInd )
        iMess = [iMess; {messages{ incomingMessInd(jj) }}];
    end
    
    iMessReceiver = messSched(ii,messReceiver);
    if iMessReceiver
        [cliques{iMessSender}, messages{ii}, vars] = cliqueFun{ii}( cliques{iMessSender}, iMess, vars );
    else % no sending message
        [cliques{iMessSender}, ~, vars] = cliqueFun{ii}( cliques{iMessSender}, iMess, vars );
    end
end