function [C, vars] = getCmat( branches, bnb2mbn_compsStatesCell, bnb2mbn_sysStates, compVarInds, vars )

complBrs = branches(arrayfun( @(x) x.isComplete, branches ));
nRow = length( complBrs );
nComp = length(complBrs(1).down);

C = zeros(nRow, 1+nComp);
for iRowInd = 1:nRow

    iC = zeros(1,1+nComp);
    iBr = complBrs(iRowInd);
    
    % System state
    if ~isempty(bnb2mbn_sysStates)
        iSysState = bnb2mbn_sysStates(iBr.up_state);
    else
        iSysState = iBr.up_state;
    end
    iC(1) = iSysState;

    % Component states
    for jCompInd = 1:nComp
        jDown = iBr.down( jCompInd );
        jUp = iBr.up( jCompInd );

        if ~isempty( bnb2mbn_compsStatesCell )
            jDown_state = bnb2mbn_compsStatesCell{jCompInd}(jDown);
            jUp_state = bnb2mbn_compsStatesCell{jCompInd}(jUp);
        else
            jDown_state = jDown;
            jUp_state = jUp;
        end

        if jUp_state ~= jDown_state
            jB = vars( compVarInds(jCompInd) ).B;
            ijB = zeros(1,size(jB,2));
            if jUp_state > jDown_state
                ijB(jDown_state:jUp_state) = 1;
            else
                ijB(jUp_state:jDown_state) = 1;
            end

            [~,ijLoc] = ismember( ijB, jB, 'rows' );

            if ~isempty(ijLoc)
                iC(1+jCompInd) = ijLoc;
            else
                jB = [jB; ijB];
                vars( compVarInds(jCompind) ).B = jB;
                iC(1+jCompInd) = size(jB,1);
            end
        else
            iC(1+jCompInd) = jUp_state;
        end

    end

    C(iRowInd,:) = iC;

end

