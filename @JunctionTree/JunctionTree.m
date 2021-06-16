classdef JunctionTree
   
    properties
        cliqueFunctionCell    
        cpmInd
        messageSchedule
        nSample        
        cpm
        variables
    end
    
    methods
        
        function jtree = JunctionTree(cliqueFunctionCell, cpmInd, messageSchedule, nSample, cpm, variables)
            if nargin > 0
                if any( cellfun( @(x) ~isa(x, 'function_handle'), cliqueFunctionCell ) )
                    error( 'Clique function Cell must be a cell of functions for each clique' )
                else
                    jtree.cliqueFunctionCell = cliqueFunctionCell;
                end                
                
                if nargin > 1              
                    if ~isa(cpmInd, 'cell')
                        error('CPM indices must be a cell array')
                    else
                        jtree.cpmInd = cpmInd;
                    end
                    
                    if nargin > 2
                        if size(messageSchedule, 2) ~= 2
                            error('Message schecule must have two columns: sending and receiving cliques')
                        else
                            jtree.messageSchedule = messageSchedule;
                        end
                        
                        if nargin > 3                    
                            if ~isscalar( nSample ) || ~isnumeric( nSample )
                                error( 'Number of samples must be a numerical scalar' )
                            else
                                jtree.nSample = nSample;
                            end

                            if nargin > 4
                                if ~isa(cpm, 'Cpm')
                                    error( 'Cpm must be a "Cpm" array' )
                                else
                                    jtree.cpm = cpm;
                                end

                                if nargin > 5
                                    if ~isa( variables, 'Variable' )
                                        error( 'Variables information must be class "Variable"' )
                                    else
                                        jtree.variables = variables;
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