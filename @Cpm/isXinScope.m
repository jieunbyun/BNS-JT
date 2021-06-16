function flagXinScope = isXinScope( X,M )
%{
Determine if M includes X in the scope
%}

flagXinScope = false( size(M) );
for xx = 1:length(X)
    flagXinScope = flagXinScope | arrayfun( @(x) ismember(X(xx),x.variables),M );
end