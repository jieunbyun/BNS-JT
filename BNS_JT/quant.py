import copy
import numpy as np
from BNS_JT import variable, cpm

# quantify cpms for standard system types

def sys_max_val( name, vars_p ):

    # Reference: variant of Algorithm MBN-quant-MSSP in Byun, J. E., & Song, J. (2021). Generalized matrix-based Bayesian network for multi-state systems. Reliability Engineering & System Safety, 211, 107468.
    # C_max is quantified for a system whose value is determined as the maximum values of variables in vars_p
    """
    INPUT: 
    vars_p: a list of variables (the parent nodes of the node of interest)
    OUTPUT:
    cpm_new: a new CPM representing the system node
    var_new: a new variable representing the system node
    """

    def get_mv( var ): # get minimum values
        return min(var.values)
    
    vars_p_s = copy.deepcopy(vars_p)
    vars_p_s.sort(key=get_mv) # sort variables to minimise # of C's rows

    C_new = np.empty(shape=(0,1+len(vars_p)), dtype='int32') # var: [X] + vars_p
    vals_new = []
    for i, p in enumerate(vars_p_s):
        vs_p = copy.deepcopy( vars_p_s[i].values )
        vs_p.sort()
        
        for v in vs_p:
            c_i = np.zeros(shape=(1,1+len(vars_p)), dtype='int32')

            j = vars_p.index(p)
            c_i[0][j+1] = p.values.index(v)

            add = True
            for i2, p2 in enumerate(vars_p_s):
                if i != i2:
                    if i2 < i:
                        vs_i2 = {y for y, z in enumerate(p2.values) if z < v}

                    if i2 > i:
                        vs_i2 = {y for y, z in enumerate(p2.values) if z <= v}
                        

                    if len(vs_i2) < 1:
                        add = False
                        break
                    else:
                        st_i2 = p2.B.index(vs_i2)

                        j2 = vars_p.index(p2)
                        c_i[0][j2+1] = st_i2

            if add:
                if v not in vals_new:
                    vals_new.append(v)
                c_i[0][0] = vals_new.index(v)
                C_new = np.vstack([C_new, c_i])

    vals_new.sort()

    var_new = variable.Variable( name=name, values=vals_new )
    cpm_new = cpm.Cpm( variables=[var_new]+vars_p, no_child = 1, C=C_new, p=np.ones(shape=(len(C_new),1), dtype='float64') )

    return cpm_new, var_new


def sys_min_val( name, vars_p ):

    # Reference: variant of Algorithm MBN-quant-MSSP in Byun, J. E., & Song, J. (2021). Generalized matrix-based Bayesian network for multi-state systems. Reliability Engineering & System Safety, 211, 107468.
    # C_max is quantified for a system whose value is determined as the minimum values of variables in vars_p
    """
    INPUT: 
    vars_p: a list of variables (the parent nodes of the node of interest)
    OUTPUT:
    cpm_new: a new CPM representing the system node
    var_new: a new variable representing the system node
    """

    def get_mv( var ): # get maximum values
        return max(var.values)
    
    vars_p_s = copy.deepcopy(vars_p)
    vars_p_s.sort(key=get_mv) # sort variables to minimise # of C's rows
    
    C_new = np.empty(shape=(0,1+len(vars_p)), dtype='int32') # var: [X] + vars_p
    vals_new = []
    for i, p in enumerate(vars_p_s):
        vs_p = copy.deepcopy( vars_p_s[i].values )
        vs_p.sort()
        
        for v in vs_p:
            c_i = np.zeros(shape=(1,1+len(vars_p)), dtype='int32')

            j = vars_p.index(p)
            c_i[0][j+1] = p.values.index(v)

            add = True
            for i2, p2 in enumerate(vars_p_s):
                if i != i2:
                    if i2 < i:
                        vs_i2 = {y for y, z in enumerate(p2.values) if z > v}

                    if i2 > i:
                        vs_i2 = {y for y, z in enumerate(p2.values) if z >= v}
                        

                    if len(vs_i2) < 1:
                        add = False
                        break
                    else:
                        st_i2 = p2.B.index(vs_i2)

                        j2 = vars_p.index(p2)
                        c_i[0][j2+1] = st_i2

            if add:
                if v not in vals_new:
                    vals_new.append(v)
                c_i[0][0] = vals_new.index(v)
                C_new = np.vstack([C_new, c_i])

    vals_new.sort()

    var_new = variable.Variable( name=name, values=vals_new )
    cpm_new = cpm.Cpm( variables=[var_new]+vars_p, no_child = 1, C=C_new, p=np.ones(shape=(len(C_new),1), dtype='float64') )

    return cpm_new, var_new