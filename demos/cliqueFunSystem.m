function [cliqueSystem, messNull, vars] = cliqueFunSystem( cliqueSystem, messToSystem, vars )

messToSystem = messToSystem{1};
[cliqueSystem, vars] = product(cliqueSystem, messToSystem, vars);

messNull = [];