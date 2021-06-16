function nElementsInCpms = countElementsInCpms( CpmArray )

nElementsInCpms = 0;
for ii = 1:length( CpmArray )
    iCpm = CpmArray(ii);
    
    iNElementInC = size( iCpm.C, 1 ) * size( iCpm.C, 2 );
    iNElementInP = length( iCpm.p );
    iNElementInQ = length( iCpm.q );
    
    nElementsInCpms = nElementsInCpms + ( iNElementInC + iNElementInP + iNElementInQ );
end