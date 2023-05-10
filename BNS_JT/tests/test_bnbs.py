"""
Ji-Eun Byun (ji-eun.byun@glasgow.ac.uk)
Created: 11 Apr 2023

Generalise Branch-and-Bound (BnB) operation to build CPMs
"""
import numpy as np
import networkx as nx
import pdb
from scipy.stats import lognorm
import matplotlib
import pytest

matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

from Trans import bnb_fns
from BNS_JT.branch import get_cmat, run_bnb
from BNS_JT.cpm import variable_elim, Cpm, get_prob
from BNS_JT.variable import Variable
from Trans.trans import get_arcs_length, get_all_paths_and_times


def test_bnbs(main_bridge):

    ODs_prob_delay, ODs_prob_disconn, _, _, _, cpms_arcs, vars_arc=main_bridge

    # FIXME: not sure what happens, but cpms['12'].variable are integers not strings
    # cpms_arc
    vars_arc = {str(k): v for k, v in vars_arc.items()}

    cpms_arc = {}
    for k, v in cpms_arcs.items():
        cpms_arc[str(k)] = v
        #cpms_arc[str(k)].variables = [str(i) for i in v.variables]

    ## Problem
    #odInd = 1
    info = {'path': [['2'], ['3', '1']],
            'time': np.array([0.0901, 0.2401]),
            'arcs': np.array(['1', '2', '3', '4', '5', '6']),
            'max_state': 2
            }

    max_state = 2
    comp_max_states = (max_state*np.ones(len(info['arcs']))).tolist()

    branches = run_bnb(sys_fn=bnb_fns.bnb_sys,
                       next_comp_fn=bnb_fns.bnb_next_comp,
                       next_state_fn=bnb_fns.bnb_next_state,
                       info=info,
                       comp_max_states=comp_max_states)

    C_od = get_cmat(branches=branches,
                      comp_var=[vars_arc[i] for i in  info['arcs']], flag=False)

    # Check if the results are correct
    # FIXME: index issue
    od_var_id = 7 - 1
    var_elim_order = [vars_arc[i] for i in ['1', '2', '3', '4', '5', '6']]

    M_bnb = list(cpms_arc.values())[:10]
    M_bnb[od_var_id].C = C_od
    M_bnb[od_var_id].p = np.ones(shape=(C_od.shape[0], 1))
    M_bnb_VE = variable_elim(M_bnb, var_elim_order)

    # FIXME: index issue
    disconn_state = 3-1 # max basic state
    disconn_prob = get_prob(M_bnb_VE, [vars_arc['7']], np.array([disconn_state]))
    delay_prob = get_prob(M_bnb_VE, [vars_arc['7']], np.array([1-1]), 0 )

    # Check if the results are the same
    # FIXME: index issue
    np.testing.assert_array_almost_equal(ODs_prob_delay[0], delay_prob)
    np.testing.assert_array_almost_equal(ODs_prob_disconn[0], disconn_prob)

