function M = updateComponentCpm( M, compInd, reliability )

M( compInd ).p = [reliability; 1-reliability];