function CPMnew = plus( CPM1,CPM2 )
%{
Output:
Variable orders of CPM1 is preserved
%}

if length(CPM1) ~= length(CPM2 )
    error( 'Given CPMs have different lengths' )
end

if ~iscell(CPM1) && ( length(CPM1) == 1 )
    CPM1 = {CPM1};
end
if ~iscell(CPM2) && ( length(CPM2) == 1 )
    CPM2 = {CPM2};
end


for cpmInd = 1:length(CPM1)
    cpm1 = CPM1{cpmInd};
    cpm2 = CPM2{cpmInd};
    

    if ~isequal( cpm1.variables(1:cpm1.numChild),cpm2.variables(1:cpm2.numChild) ) || ...
            ~isequal( cpm1.variables((cpm1.numChild+1):end),cpm2.variables((cpm2.numChild+1):end) )
        error('Given CPMs must be defined over the same scope')
    end

    if ~isempty( cpm1.q ) || ~isempty( cpm1.q )
        warning('Plus operation is not registered to q vector (sample weight). q vector is returned empty')
    end
    

    Cnew = cpm1.C; pnew = cpm1.p;
    [~,vars2_idx] = ismember( cpm2.variables,cpm1.variables );
    [~,vars2_idx] = sort(vars2_idx);
    C2 = cpm2.C(:,vars2_idx); p2 = cpm2.p;
    for cc = 1:length(p2)
       idx = find( ismember(Cnew,C2(cc,:),'rows') );
       if isempty(idx)
           Cnew = [Cnew; C2(cc,:)]; pnew = [pnew; p2(cc)];
       else
           idx = idx(1);
           pnew(idx) = pnew(idx)+p2(cc);
       end
    end

    cpmNew = cpm1;
    cpmNew.C = Cnew; cpmNew.p = pnew;
    cpmNew.q = [];
    
    CPMnew{cpmInd} = cpmNew;
end


if length(CPMnew) == 1
    CPMnew = CPMnew{1};
end