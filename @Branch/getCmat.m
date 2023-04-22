function [C, vars] = getCmat( branches, compVarInds, vars, flag_compStateOrder )

complBrs = branches(arrayfun( @(x) x.isComplete, branches ));
nRow = length( complBrs );
nComp = length(complBrs(1).down);

if nargin < 4
    flag_compStateOrder = 1; % 1 (default) if bnb and mbn have the same component states, 0 if bnb has a reverse ordering of components being better and worse
end


C = zeros(nRow, 1+nComp);
for iRowInd = 1:nRow

    iC = zeros(1,1+nComp);
    iBr = complBrs(iRowInd);
    
    % System state
    iC(1) = iBr.up_state;

    % Component states
    for jCompInd = 1:nComp
        jDown = iBr.down( jCompInd );
        jUp = iBr.up( jCompInd );

        jCompVarInd = compVarInds(jCompInd);
        jB = vars(jCompVarInd).B;
        jNState = size(jB,2);

        if flag_compStateOrder
            jDown_state = jDown;
            jUp_state = jUp;
        else
            jDown_state = (jNState+1) - jUp;
            jUp_state = (jNState+1) - jDown;
        end

        if jUp_state ~= jDown_state
            ijB = zeros(1,size(jB,2));
            ijB(jDown_state:jUp_state) = 1;

            [~,ijLoc] = ismember( ijB, jB, 'rows' );

            if ~isempty(ijLoc)
                iC(1+jCompInd) = ijLoc;
            else
                jB = [jB; ijB];
                vars( compVarInds(jCompInd) ).B = jB;
                iC(1+jCompInd) = size(jB,1);
            end
        else
            iC(1+jCompInd) = jUp_state;
        end

    end

    C(iRowInd,:) = iC;

end

