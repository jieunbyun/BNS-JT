classdef Branch
    
    properties
        down
        up
        isComplete=false % 0 unknown, 1 confirmed
        down_state % associated system state on the lower bound (if 0, unknown)
        up_state % associated system state on the upper bound (if 0, unknown)
        down_val % (optional) a representative value of an associated state 
        up_val % (optional) a representative value of an associated state 
    end
    
    methods
        function branch = Branch(down,up,isComplete,down_state, up_state, down_val, up_val)
            if nargin>0
                branch.down=down(:)';
                if nargin>1
                    if length( up ) ~= length( down )
                        error( 'Vectors "down" and "up" must have the same length.' )
                    else
                        branch.up = up(:)';
                    end
                    if nargin>2
                        if ~ismember( isComplete, [true, false] )
                            error( '"isComplete" must be either true (or 1) or false (or 0).' )
                        else
                            branch.isComplete = isComplete;
                        end

                        if nargin>3
                            if down_state < 1 
                                error( '"down_state" must be a positive integer (if to be input).' )
                            else
                                branch.down_state = down_state(:);
                            end

                            if nargin > 4
                                if up_state < 1 
                                    error( '"up_state" must be a positive integer (if to be input).' )
                                else
                                    branch.up_state = up_state(:);
                                end 

                                if nargin > 5
                                    branch.down_val = down_val;

                                    if nargin > 6
                                        branch.up_val = up_val;
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end