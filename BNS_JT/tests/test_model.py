import numpy as np
import pytest
import copy
import pdb
import itertools
from pathlib import Path

from BNS_JT import config, model, trans, variable, branch


HOME = Path(__file__).absolute().parent


@pytest.fixture(scope='session')
def setup_road():

    cfg = config.Config(HOME.joinpath('../demos/road/test.json'))
    cpms, varis = model.setup_model(cfg)

    return cpms, varis, cfg


def test_setup_model(setup_road, setup_bridge):

    d_cpms, d_varis, cfg = setup_road
    cpms = copy.deepcopy(d_cpms)
    varis = copy.deepcopy(d_varis)

    expected, _, _, _ = setup_bridge

    od_scen_pairs = itertools.product(cfg.infra['ODs'].keys(), cfg.scenarios['scenarios'].keys())

    for od, scen in od_scen_pairs:

        np.testing.assert_array_equal(cpms[(od, scen)][od].C, expected[od].C)


def test_compute_prob1(setup_road, expected_probs):

    d_cpms, d_varis, cfg = setup_road
    cpms = copy.deepcopy(d_cpms)
    varis = copy.deepcopy(d_varis)

    var_elim = list(cfg.infra['edges'].keys())

    od_scen_pairs = itertools.product(cfg.infra['ODs'].keys(), cfg.scenarios['scenarios'].keys())

    map_dic = {f'od{i+1}': i for i in range(4)}
    for od, scen in od_scen_pairs:

        prob, _ = model.compute_prob(cfg, cpms[(od, scen)], varis[(od, scen)], var_elim, od, 0, flag=True)
        print(prob, expected_probs['disconn'][map_dic[od]])
        assert expected_probs['disconn'][map_dic[od]] == pytest.approx(prob, abs=0.0001)


def test_compute_prob2(setup_road, expected_probs):

    d_cpms, d_varis, cfg = setup_road
    cpms = copy.deepcopy(d_cpms)
    varis = copy.deepcopy(d_varis)

    var_elim = list(cfg.infra['edges'].keys())

    od_scen_pairs = itertools.product(cfg.infra['ODs'].keys(), cfg.scenarios['scenarios'].keys())

    map_dic = {f'od{i+1}': i for i in range(4)}
    for od, scen in od_scen_pairs:

        prob, _ = model.compute_prob(cfg, cpms[(od, scen)], varis[(od, scen)], var_elim, od, 2, flag=False)

        assert expected_probs['delay'][map_dic[od]] == pytest.approx(prob, abs=0.0001)


def test_get_branches(setup_bridge):

    expected, _, _, _ = setup_bridge

    cfg = config.Config(HOME.joinpath('../demos/road/test.json'))

    path_times = trans.get_all_paths_and_times(cfg.infra['ODs'].values(), cfg.infra['G'], key='weight')

    branches = model.get_branches(cfg, path_times)

    assert len(branches) == len(cfg.infra['ODs'])

    # FIXME: only works for binary ATM
    varis = {}
    #B = np.vstack([np.eye(cfg.no_ds), np.ones(cfg.no_ds)])
    for k in cfg.infra['edges'].keys():
        B = [{i} for i in range(cfg.no_ds)]
        B.append({i for i in range(cfg.no_ds)})
        #varis[k] = variable.Variable(name=k, B=B.copy(), values=cfg.scenarios['damage_states'])
        varis[k] = variable.Variable(name=k, values=cfg.scenarios['damage_states'])

    for od, value in branches.items():
        values = sorted([y for _, y in path_times[cfg.infra['ODs'][od]]]) + [np.inf]
        #varis[od] = variable.Variable(name=od, B=[{i} for i in range(len(values))], values=values)
        varis[od] = variable.Variable(name=od, values=values)

        variables = {k: varis[k] for k in cfg.infra['edges'].keys()}
        c = branch.get_cmat_from_branches(value, variables)

        np.testing.assert_array_equal(c, expected[od].C)


def test_model_given_od_scen(setup_bridge):

    expected, _, _, _ = setup_bridge

    cfg = config.Config(HOME.joinpath('../demos/road/test.json'))

    path_times = trans.get_all_paths_and_times(cfg.infra['ODs'].values(), cfg.infra['G'], key='weight')

    branches = model.get_branches(cfg, path_times)

    od_scen_pairs = itertools.product(cfg.infra['ODs'].keys(), cfg.scenarios['scenarios'].keys())

    for od, scen in od_scen_pairs:

        cpms, varis = model.model_given_od_scen(cfg, path_times, od, scen, branches[od])

        assert len(cpms) == 7

        np.testing.assert_array_equal(cpms[od].C, expected[od].C)


