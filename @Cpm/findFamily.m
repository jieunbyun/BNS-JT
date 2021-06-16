function cpmInds = findFamily( cpmInd, cpmArray )

if ~isscalar( cpmInd ), error('CPM index must be a positive scalar'); end
if ~isa( cpmArray, 'Cpm' ), error( 'CPM array must be "cpm" class' ); end

CpmIndsToBeExamined = cpmInd;
isCpmYetExamined = setdiff(1:length(cpmArray), CpmIndsToBeExamined);
cpmInds = [];
while ~isempty( CpmIndsToBeExamined )
    
    iCpmInd = CpmIndsToBeExamined(1);
    CpmIndsToBeExamined(1) = [];
    cpmInds = [cpmInds; iCpmInd];
    
    iCpmSharingVariablesInd_local = find( isXinScope(cpmArray(iCpmInd).variables, cpmArray(isCpmYetExamined)) );
    iCpmSharingVariablesInd = setdiff( isCpmYetExamined( iCpmSharingVariablesInd_local ), cpmInds );
    
    CpmIndsToBeExamined = [CpmIndsToBeExamined iCpmSharingVariablesInd(:)'];
    isCpmYetExamined( iCpmSharingVariablesInd_local ) = [];    
    
end

cpmInds = sort(cpmInds);