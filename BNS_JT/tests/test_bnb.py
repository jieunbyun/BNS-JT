"""
Ji-Eun Byun (ji-eun.byun@glasgow.ac.uk)
Created: 11 Apr 2023

Generalise Branch-and-Bound (BnB) operation to build CPMs
"""
import numpy as np
import pdb
import warnings
import pytest

from BNS_JT.bnb_fns import bnb_sys, bnb_next_comp, bnb_next_state
from BNS_JT.branch import get_cmat, run_bnb
from BNS_JT.cpm import variable_elim, Cpm, get_prob

expected_disconn = np.array([0.0096, 0.0011, 0.2102, 0.2102])
expected_delay = np.array([0.0583, 0.0052, 0.4795, 0.4382])

arcs = {'e1': ['n1', 'n2'],
	'e2': ['n1', 'n5'],
	'e3': ['n2', 'n5'],
	'e4': ['n3', 'n4'],
	'e5': ['n3', 'n5'],
	'e6': ['n4', 'n5']}

var_ODs = {'od1': ('n5', 'n1'),
           'od2': ('n5', 'n2'),
           'od3': ('n5', 'n3'),
           'od4': ('n5', 'n4')}

def test_bnb(setup_bridge):

    cpms_arc, vars_arc = setup_bridge

    # cpms_arc
    #vars_arc = {int(k): v for k, v in vars_arc.items()}

    #cpms_arc = {}
    #for k, v in cpms_arcs.items():
    #    cpms_arc[k] = v
        #cpms_arc[k].variables = [int(i) for i in v.variables]

    ## Problem
    info = {'path': [['e2'], ['e3', 'e1']],
            'time': np.array([0.0901, 0.2401]),
            'arcs': list(arcs.keys()),
            'max_state': 2
            }

    max_state = 2
    comp_max_states = (max_state*np.ones(len(arcs))).tolist()

    branches = run_bnb(sys_fn=bnb_sys,
                       next_comp_fn=bnb_next_comp,
                       next_state_fn=bnb_next_state,
                       info=info,
                       comp_max_states=comp_max_states)

    C_od= get_cmat(branches, [vars_arc[i] for i in arcs.keys()], False)

    # Check if the results are correct
    # FIXME: index issue
    od_var_id = 'od1' #7 - 1
    var_elim_order = [vars_arc[i] for i in arcs.keys()]

    #M_bnb = {i: cpms_arc[i] for i in list(arcs.keys()) + list(var_ODs.keys())}
    M_bnb = {i: cpms_arc[i] for i in list(arcs.keys()) + ['od1']}
    # FIXME: why M_bnb requires cpms of od2, od3, and od4??? 
    #M_bnb = [cpms_arc[i] for i in list(arcs.keys()) + ['od1']]
    M_bnb[od_var_id].C = C_od
    M_bnb[od_var_id].p = np.ones(shape=(C_od.shape[0], 1))
    M_bnb_VE= variable_elim(M_bnb, var_elim_order)

    # FIXME: index issue
    disconn_state = 3-1 # max basic state
    disconn_prob = get_prob(M_bnb_VE, [vars_arc['od1']], np.array([disconn_state]))
    delay_prob = get_prob(M_bnb_VE, [vars_arc['od1']], np.array([1-1]), 0)

    # Check if the results are the same
    # FIXME: index issue
    np.testing.assert_array_almost_equal(expected_disconn[0], disconn_prob, decimal=4)
    np.testing.assert_array_almost_equal(expected_delay[0], delay_prob, decimal=4)

    # using variable name instead
    disconn_state = 3-1 # max basic state
    disconn_prob = get_prob(M_bnb_VE, ['od1'], np.array([disconn_state]))
    delay_prob = get_prob(M_bnb_VE, ['od1'], np.array([1-1]), 0)

    np.testing.assert_array_almost_equal(expected_disconn[0], disconn_prob, decimal=4)
    np.testing.assert_array_almost_equal(expected_delay[0], delay_prob, decimal=4)




