function [Mmult,vInfo] = multCPMs(M,vInfo)

Mmult = nullCPM;
for mm = 1:length(M)
    [Mmult,vInfo] = product(Mmult,M(mm),vInfo);
end