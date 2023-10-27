from scipy.optimize import LinearConstraint
from scipy.optimize import milp
import numpy as np


def run_pipes_fun(comps_st, edges, node_coords, es_idx, edges2comps, depots, varis, sub_bw_nodes, sub_bw_edges):

    # cost function
    ## decision variables (x, u, v), where x are flows on arcs, u are flows from sources, and v are flows to terminals.
    no_x = len(edges)

    orig_end = depots[0]
    dest_end = depots[-1]

    orig_end_inds = {}
    ind = 0
    for n in orig_end:
        ind += 1
        orig_end_inds[n] = ind

    dest_end_inds = {}
    ind = 0
    for n in dest_end:
        ind += 1
        dest_end_inds[n] = ind

    no_u = len( orig_end )

    no_d_vars = no_x + no_u

    # Cost function
    c = np.zeros( (no_d_vars,) )
    c[no_x:(no_x+no_u)] = -1

    # constraints

    ## Constraint matrices
    A = np.empty(shape = (0,no_d_vars))
    b_up = np.empty(shape=(0,))
    b_down = np.empty(shape=(0,))

    ### each origin node
    for o_i in orig_end:
        a_i = np.zeros((1,no_d_vars))
        a_i[0][no_x + orig_end_inds[o_i] - 1] = -1

        for e, p in edges.items():

            e_ind = es_idx[e]

            if p[0] == o_i:
                a_i[0][ e_ind-1 ] = 1

        A = np.append(A, a_i, axis = 0)
        b_up = np.append(b_up, np.array(0))
        b_down = np.append(b_down, np.array(0))

    ### for each subsystem
    #### Intermediate nodes whose in- and out-flows be summed to zero.
    no_sub = len( sub_bw_nodes ) + 1
    for m in range(1,no_sub):
        depots_prev = depots[m-1]
        depots_now = depots[m]
        for n in sub_bw_nodes[m-1]:

            if n not in depots_prev and n not in depots_now:

                a_i = np.zeros((1,no_d_vars))

                for e in sub_bw_edges[m-1]:
                    pair = edges[e]
                    e_ind = es_idx[e]

                    if pair[0] == n:
                        a_i[0][ e_ind-1 ] = 1
                    elif pair[1] == n:
                        a_i[0][ e_ind-1 ] = -1

                A = np.append(A, a_i, axis = 0)
                b_up = np.append(b_up, np.array(0))
                b_down = np.append(b_down, np.array(0))


        a_i = np.zeros((1,no_d_vars))
        a_i[0][no_x:(no_x+no_u)] = -1
        for n in depots[m]:
            for e in sub_bw_edges[m-1]:
                pair = edges[e]
                e_ind = es_idx[e]

                if pair[1] == n:
                    a_i[0][e_ind-1] = 1

        A = np.append( A,a_i,axis=0 )
        b_up = np.append(b_up, np.array(0))
        b_down = np.append(b_down, np.array(0))

    ### All incoming flows from each depot must be taken out.
    for m in range(2,no_sub):
        for n in depots[m-1]:
            a_i = np.zeros((1,no_d_vars))
            for e,pair in edges.items():
                e_ind = es_idx[e]

                if pair[0] == n:
                    a_i[0][e_ind-1] = 1
                elif pair[1] == n:
                    a_i[0][e_ind-1] = -1

            A = np.append(A, a_i, axis = 0)
            b_up = np.append(b_up, np.array(0))
            b_down = np.append(b_down, np.array(0))

    ### capacity: Depends on component vector state
    for e, pair in edges.items():
        x = edges2comps[e]
        capa_e = varis[x].values[ comps_st[x]-1 ]

        if comps_st[pair[0]] == 1 or comps_st[pair[1]] == 1:
            capa_e = 0

        e_ind = es_idx[e]

        a_i = np.zeros((1,no_d_vars))
        a_i[0][ e_ind ] = 1
        A = np.append(A, a_i, axis = 0)
        b_up = np.append(b_up, np.array(capa_e))
        b_down = np.append(b_down, np.array(0))


    constraints = LinearConstraint(A, b_down, b_up)
    integrality = np.ones_like(c) # all decision variables are integral

    res = milp(c=c, constraints=constraints, integrality=integrality)

    return res


def sys_fun_pipes(comps_st, thres, edges, node_coords, es_idx, edges2comps, depots, varis, sub_bw_nodes, sub_bw_edges):

    res = run_pipes_fun(comps_st, edges, node_coords, es_idx, edges2comps, depots, varis, sub_bw_nodes, sub_bw_edges)

    if res.success == True:
        sys_val = -res.fun
    else:
        sys_val = 0

    if sys_val < thres:
        sys_st = 'fail'
        min_comps_st = None

    else:
        sys_st = 'surv'

        min_comps_st = {}
        for e, pair in edges.items():

            e_idx = es_idx[e]

            if res.x[e_idx-1] > 0:
                x_name = edges2comps[e]

                if x_name not in min_comps_st:
                    x_st = comps_st[x_name]
                    min_comps_st[x_name] = x_st

                if pair[0] not in min_comps_st:
                    n_st = comps_st[pair[0]]
                    min_comps_st[pair[0]] = n_st

                if pair[1] not in min_comps_st:
                    n_st = comps_st[pair[1]]
                    min_comps_st[pair[1]] = n_st

    return sys_val, sys_st, min_comps_st


