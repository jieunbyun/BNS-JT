# replicate demoMCSproduct.ipynb

import numpy as np
import pytest
import copy
import pdb
import networkx as nx
from scipy.stats import lognorm

np.set_printoptions(precision=3)

from BNS_JT import cpm, variable, trans, operation


@pytest.fixture(scope='package')
def setup_sys(data_bridge):

    cpms = {}
    varis = {}

    low = 0
    high = 1
    #p_low = 0.95
    #p_high = 0.05

    # hazard
    varis['haz'] = variable.Variable(name='haz', values=['low', 'high'])

    C = np.array([[0, 1]]).T
    p = np.array([0.95, 0.05])
    cpms['haz'] = cpm.Cpm( variables = [varis['haz']], no_child = 1, C = C, p = p )

    # 
    node_coords = data_bridge['node_coords']

    var_ODs = data_bridge['var_ODs']

    arcs = data_bridge['arcs']

    frag = data_bridge['frag']

    arcs_type = data_bridge['arcs_type']

    arcs_avg_kmh = data_bridge['arcs_avg_kmh']

    # For the moment, we assume that ground motions are observed. Later, hazard nodes will be added.
    GM_obs = data_bridge['GM_obs']

    arc_lens_km = trans.get_arcs_length(arcs, node_coords)

    #arcTimes_h = arcLens_km ./ arcs_Vavg_kmh
    arc_times_h = {k: v/arcs_avg_kmh[k] for k, v in arc_lens_km.items()}

    # create a graph
    G = nx.Graph()
    for k, x in arcs.items():
        G.add_edge(x[0], x[1], time=arc_times_h[k], label=k)

    for k, v in node_coords.items():
        G.add_node(k, pos=v)

    path_time = trans.get_all_paths_and_times(var_ODs.values(), G, key='time')

    # Arcs (components): P(X_i | GM = GM_ob ), i = 1 .. N (= nArc)
    # Arcs' states index compatible with variable B index, and C
    arc_surv = 0
    arc_fail = 1
    arc_either = 2

    C = np.array([[arc_surv, low], [arc_fail, low],
              [arc_surv, high], [arc_fail, high]])

    for k in arcs.keys():
        varis[k] = variable.Variable(name=k, values=['Surv', 'Fail'])

        _type = arcs_type[k]
        prob = [lognorm.cdf(GM_obs[k], frag[_type]['std'], scale=frag[_type]['med']),
                lognorm.cdf(GM_obs[k]*1.5, frag[_type]['std'], scale=frag[_type]['med'])]

        p = np.array([1-prob[0], prob[0], 1-prob[1], prob[1]])
        cpms[k] = cpm.Cpm(variables = [varis[k], varis['haz']],
                              no_child = 1,
                              C = C,
                              p = p)

    # Travel times (systems): P(OD_j | X1, ... Xn) j = 1 ... nOD
    varis['od1'] = variable.Variable(name='od1',
        values=[0.0901, 0.2401, np.inf])

    _variables = [varis[k] for k in ['od1', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6']]

    c7 = np.array([
        [1,3,1,3,3,3,3],
        [2,1,2,1,3,3,3],
        [3,1,2,2,3,3,3],
        [3,2,2,3,3,3,3]]) - 1

    cpms['od1'] = cpm.Cpm(variables= _variables,
                           no_child = 1,
                           C = c7,
                           p = [1, 1, 1, 1],
                           )

    return arcs, cpms, varis


def test_exact(setup_sys):

    arcs, cpms, varis = setup_sys

    #cpm_od1 = cpms.variable_elim( cpms, ['haz', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6'] )

    cpms_cp = list(cpms.values())
    for i in ['haz'] + list(arcs.keys()):
        is_inscope = operation.isinscope([varis[i]], cpms_cp)
        cpm_sel = [y for x, y in zip(is_inscope, cpms_cp) if x]
        cpm_mult = cpm.product(cpm_sel)
        cpm_mult = cpm_mult.sum([varis[i]])

        cpms_cp = [y for (x,y) in zip(is_inscope, cpms_cp) if x == False]
        cpms_cp.insert(0, cpm_mult)

    prob_disruption = cpms_cp[0].p[1][0] + cpms_cp[0].p[2][0]
    prob_disconnection = cpms_cp[0].p[2][0]

    assert prob_disruption == pytest.approx(0.0634, abs=1.0e-4)
    assert prob_disconnection == pytest.approx(0.012, abs=1.0e-4)


def test_variable_elim(setup_sys):

    arcs, cpms, varis = setup_sys

    #M = [cpms[k] for k in ['haz'] + list(arcs.keys()) + ['od1']]
    M = [cpms[k] for k in ['od1'] + ['haz'] + list(arcs.keys())]
    elim = [varis[k] for k in ['haz', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6']]
    cpm_od1 = operation.variable_elim(M, elim)

    # prob disruption
    assert cpm_od1.p[1] + cpm_od1.p[2] == pytest.approx(0.0634, abs=1.0e-4)
    # prob disconnection
    assert cpm_od1.p[2] == pytest.approx(0.012, abs=1.0e-4)


#@pytest.mark.skip('FIXME')
def test_mcs_product(setup_sys):

    arcs, cpms, varis = setup_sys

    nsample = 10
    var_mcs = ['haz']
    cpm_h_mcs = operation.mcs_product({'haz': cpms['haz']}, nsample)
    #assert cpm_h_mcs.C.sum() == pytest.approx(nsample * cpms['haz'].p[1], abs=3)

    cpms_cp = list(cpms.values())
    elim_order = ['e1', 'e2', 'e3', 'e4', 'e5', 'e6'] # except 'haz' and 'od1'
    for _c, _p in zip(cpm_h_mcs.Cs, cpm_h_mcs.q):
        cpms_c = operation.condition(cpms, [varis['haz']], _c.tolist() )

        for i in elim_order:
            is_inscope = operation.isinscope([varis[i]], cpms_cp)
            cpm_sel = [y for x, y in zip(is_inscope, cpms_cp) if x]

            if cpm_sel:
                cpm_mult = cpm.product(cpm_sel)
                cpm_mult = cpm_mult.sum([varis[i]])

                cpms_cp = [y for x, y in zip(is_inscope, cpms_cp) if not x]
                cpms_cp.insert(0, cpm_mult)

    cpm_od1 = cpms_cp[0].product(cpms_cp[1])
    # prob disruption
    assert cpm_od1.p[(cpm_od1.C[:, 1] == 1) | (cpm_od1.C[:, 1] ==2)].sum() == pytest.approx(0.0634, abs=1.0e-4)
    # prob disconnection
    assert cpm_od1.p[cpm_od1.C[:, 1] == 2].sum() == pytest.approx(0.012, abs=1.0e-4)


