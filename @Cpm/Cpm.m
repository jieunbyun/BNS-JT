classdef Cpm
    
    properties
        variables
        numChild
        C % event matrix
        p % probability vector
        q % sampling weight vector
        sampleIndex % sample index (numbering) vector
    end
    
    methods
        function M = Cpm(variables,numChild,C,p,q,sampleIndex)
            if nargin>0
                M.variables=variables(:)';
                if nargin>1
                    M.numChild = numChild;
                    if nargin>2
                        M.C = C;
                        if nargin>3
                            M.p = p(:);
                            if nargin > 4
                                M.q = q(:);
                                if nargin > 5
                                    M.sampleIndex = sampleIndex(:);
                                end
                            end
                        end
                    end
                end
            end
            [M,errFlag,errMess] = errCheckCpm(M);
            if errFlag
                error(errMess);
            end
        end
            
        function [M,errFlag,errMess] = errCheckCpm(M)
            errFlag = 0;
            errMess = '';
            if ~isempty(M.variables) && ~isnumeric(M.variables)
                errFlag = 1;
                errMess = 'variables must be a numeric vector';
            elseif ~isempty(M.numChild) && (~isnumeric(M.numChild)||~isscalar(M.numChild))
                errFlag = 1;
                errMess = 'numChild must be a numeric scalar';
            elseif (~isempty(M.numChild)&&~isempty(M.variables)) && (M.numChild>length(M.variables))
                errFlag = 1;
                errMess ='numChild must be greater than the number of variables';
            elseif ~isempty(M.C) && ~isnumeric(M.C)
                errFlag = 1;
                errMess = 'Event matrix C must be a numeric matrix';
            elseif (~isempty(M.C)&&~isempty(M.variables)) && (size(M.C,2)~=length(M.variables))
                errFlag = 1;
                errMess ='C must have the same number of columns with that of variables';
            elseif ~isempty(M.p) && ~isnumeric(M.p)
                errFlag = 1;
                errMess ='Probability vector p must be a numeric vector';
            elseif (~isempty(M.p)&&~isempty(M.C)) && (length(M.p)~=size(M.C,1))
                errFlag = 1;
                errMess = 'p must have the same length with the number of rows in C';
            elseif ~isempty(M.q) && ~isnumeric(M.q)
                errFlag = 1;
                errMess ='Sampling probability vector q must be a numeric vector';
            elseif (~isempty(M.q)&&~isempty(M.C)) && (length(M.q)~=size(M.C,1))
                errFlag = 1;
                errMess = 'q must have the same length with the number of rows in C';
            elseif (~isempty(M.sampleIndex)&&~isempty(M.C)) && (length(M.sampleIndex)~=size(M.C,1))
                errFlag = 1;
                errMess = 'Sample index array must have the same length with the number of rows in C';
            end                
        end

    end
end