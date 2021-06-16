function [cliqueSystem, messNull, vars] = cliqueFunSystem_conditioning( cliqueSystem, messToSystem, vars )

messToSystemCell = messToSystem;
messToSystem = [];
for messInd = 1:length( messToSystemCell )
    messToSystem = [messToSystem; messToSystemCell{messInd}];
end

[messToSystem, vars] = multCPMs( messToSystem, vars );
[cliqueSystem, vars] = product(cliqueSystem, messToSystem, vars);

messNull = [];