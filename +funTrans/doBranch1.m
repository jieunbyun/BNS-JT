function [branches_surv, branches_fail, branches_state, branches_val] = doBranch1( compInds, state, allSurv_state, unfinished_state )

nComp = length(compInds);

branches_surv = cell(nComp+1,1);
branches_fail = cell(nComp+1,1);
branches_state = zeros(nComp+1,1);
branches_val = zeros(nComp+1,1);

branches_surv{1} = compInds;
branches_state(1) = state;
branches_val(1) = allSurv_state;

for iCompInd_ = nComp:-1:1 % '_' denotes that it is not original index but the one in the path
    iComps_surv = compInds(1:(iCompInd_-1));
    iComps_fail = compInds(iCompInd_);

    branches_surv(nComp-iCompInd_+1+1) = {iComps_surv};
    branches_fail(nComp-iCompInd_+1+1) = {iComps_fail};
    branches_state(nComp-iCompInd_+1+1) = unfinished_state;
    branches_val(nComp-iCompInd_+1+1) = unfinished_state;
end