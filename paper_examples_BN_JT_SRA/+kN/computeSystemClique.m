function [systemClique, nullMessage, vars] = computeSystemClique( systemClique, messToSystemClique, vars, VAR_INTERMEDIATE_LAST_NODE, VAR_CONDITIONED )

if ~isempty( messToSystemClique )
    messToSystemClique = messToSystemClique{1};
end

systemClique = sum(messToSystemClique, [VAR_INTERMEDIATE_LAST_NODE(:)' VAR_CONDITIONED(:)'], 0);
nullMessage = [];