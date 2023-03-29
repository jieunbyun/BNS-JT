
classdef Variable
    
    properties
        B % basis set
        v % description of states
    end
    
    methods
        function var = Variable(B,v)            
            if nargin>0          
                if ismatrix(B)
                    var.B=B;
                elseif iscell(B)
                    var.B=B(:);
                else
                    error('B must be either a matrix (given one variable) or a cell (multiple variables)')
                end
                    
                if nargin>1
                    var.v=v;
                end
            end            
            [var,errFlag,errMess] = errCheckVariable(var);
            if errFlag
                error(errMess);
            end
        end
        
        function [var,errFlag,errMess] = errCheckVariable(var)
            errFlag = 0;
            errMess = '';
            if ~isempty(var.B)
                numBasicState = size(var.B,2);
                if ~isequal( var.B(1:numBasicState,:),eye(numBasicState) )
                    errFlag = 1;
                    errMess = 'The upper part corresponding to basic states must form an identity matrix';
                end 
%                 if ~isempty(var.v)
%                     if size(var.v,1)~=size(var.B,2)
%                         errFlag = 1;
%                         errMess = 'The number of rows of v (described states) must be the same with the number of columns in B';
%                     end 
%                 end
            end
        end
 
    end
end