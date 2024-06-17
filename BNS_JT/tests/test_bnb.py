"""
Ji-Eun Byun (ji-eun.byun@glasgow.ac.uk)
Created: 11 Apr 2023

Generalise Branch-and-Bound (BnB) operation to build CPMs
"""
import numpy as np
import copy
import pdb
import warnings
import pytest

from BNS_JT import bnb_fns, branch, cpm, operation


def test_bnb(setup_bridge, expected_probs):

    d_cpms_arc, d_vars_arc, arcs, var_ODs = setup_bridge

    cpms_arc = copy.deepcopy(d_cpms_arc)
    vars_arc = copy.deepcopy(d_vars_arc)

    ## Problem
    info = {'path': [['e2'], ['e3', 'e1']],
            'time': np.array([0.0901, 0.2401]),
            'arcs': list(arcs.keys()),
            'max_state': 2
            }

    max_state = 2
    comp_max_states = (max_state*np.ones(len(arcs))).tolist()

    # run_bnb not working
    branches = branch.run_bnb(sys_fn=bnb_fns.bnb_sys,
                       next_comp_fn=bnb_fns.bnb_next_comp,
                       next_state_fn=bnb_fns.bnb_next_state,
                       info=info,
                       comp_max_states=comp_max_states)

    C_od= branch.get_cmat(branches, [vars_arc[i] for i in arcs.keys()])

    # Check if the results are correct
    # FIXME: index issue
    od_var_id = 'od1' #7 - 1
    var_elim_order = [vars_arc[i] for i in arcs.keys()]
    #pdb.set_trace()
    #M_bnb = {i: cpms_arc[i] for i in list(arcs.keys()) + list(var_ODs.keys())}
    M_bnb = {i: cpms_arc[i] for i in list(arcs.keys()) + ['od1']}
    #M_bnb = [cpms_arc[i] for i in list(arcs.keys()) + ['od1']]
    #M_bnb[od_var_id].C = C_od
    #M_bnb[od_var_id].p = np.ones(shape=(C_od.shape[0], 1))
    M_bnb_VE= operation.variable_elim(M_bnb, var_elim_order)

    #print(M_bnb_VE)
    # FIXME: index issue
    disconn_state = 0 # max basic state
    disconn_prob = M_bnb_VE.get_prob([vars_arc['od1']], [disconn_state])
    delay_prob = M_bnb_VE.get_prob([vars_arc['od1']], [0]) + M_bnb_VE.get_prob([vars_arc['od1']], [1])

    # Check if the results are the same
    # FIXME: index issue
    np.testing.assert_array_almost_equal(expected_probs['disconn'][0], disconn_prob, decimal=4)
    np.testing.assert_array_almost_equal(expected_probs['delay'][0], delay_prob, decimal=4)

    # using variable name instead
    disconn_state = 0 # max basic state
    disconn_prob = M_bnb_VE.get_prob(['od1'], np.array([disconn_state]))
    delay_prob = M_bnb_VE.get_prob(['od1'], [0]) + M_bnb_VE.get_prob(['od1'], [1])
    #delay_prob = cpm.get_prob(M_bnb_VE, ['od1'], np.array([1-1]), 0)

    np.testing.assert_array_almost_equal(expected_probs['disconn'][0], disconn_prob, decimal=4)
    np.testing.assert_array_almost_equal(expected_probs['delay'][0], delay_prob, decimal=4)



