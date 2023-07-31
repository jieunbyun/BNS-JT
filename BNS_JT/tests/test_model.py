import numpy as np
import pytest
import itertools
from pathlib import Path

from BNS_JT import config, model, trans, variable, branch


HOME = Path(__file__).absolute().parent

expected_disconn = np.array([0.0096, 0.0011, 0.2102, 0.2102])
expected_delay = np.array([0.0583, 0.0052, 0.4795, 0.4382])
expected_damage = np.array([0.1610,  1,  1,  0.0002,   0,  0.4382])


@pytest.fixture()
def setup_road():

    cfg = config.Config(HOME.joinpath('./config_roads.json'))

    cpms, varis = model.setup_model(cfg)

    return cpms, varis, cfg


@pytest.fixture()
def cmatrix_road():

    expected={}
    expected['od1'] = np.array([[0,2,0,0,2,2,2],
                                [0,0,0,1,2,2,2],
                                [1,1,0,1,2,2,2],
                                [2,2,1,2,2,2,2]])

    expected['od2'] = np.array([[0,2,0,0,2,2,2],
				[0,0,1,0,2,2,2],
				[1,1,1,0,2,2,2],
				[2,2,2,1,2,2,2]])

    expected['od3'] = np.array([[0,2,2,2,2,0,0],
				[0,2,2,2,0,0,1],
				[1,2,2,2,1,0,1],
				[2,2,2,2,2,1,2]])

    expected['od4'] = np.array([[0,2,2,2,2,0,0],
				[0,2,2,2,0,1,0],
				[1,2,2,2,1,1,0],
				[2,2,2,2,2,2,1]])
    return expected


def test_setup_model(setup_road, cmatrix_road):

    cpms, varis, cfg = setup_road
    expected = cmatrix_road

    od_scen_pairs = itertools.product(cfg.infra['ODs'].keys(), cfg.scenarios['scenarios'].keys())

    for od, scen in od_scen_pairs:

        np.testing.assert_array_equal(cpms[(od, scen)][od].C, expected[od])


def test_compute_prob1(setup_road):

    cpms, varis, cfg = setup_road

    var_elim = list(cfg.infra['edges'].keys())

    for i, k in enumerate(['od1', 'od2', 'od3', 'od4']):
        prob, _ = model.compute_prob(cfg, cpms[(k, 's1')], varis[(k, 's1')], var_elim, k, 0, flag=True)

        assert expected_disconn[i] == pytest.approx(prob, abs=0.0001)


def test_compute_prob2(setup_road):

    cpms, varis, cfg = setup_road

    var_elim = list(cfg.infra['edges'].keys())

    for i, k in enumerate(['od1', 'od2', 'od3', 'od4']):
        prob, _ = model.compute_prob(cfg, cpms[(k, 's1')], varis[(k, 's1')], var_elim, k, 2, flag=False)

        assert expected_delay[i] == pytest.approx(prob, abs=0.0001)


def test_get_branches(cmatrix_road):

    expected = cmatrix_road
    cfg = config.Config(HOME.joinpath('./config_roads.json'))

    path_times = trans.get_all_paths_and_times(cfg.infra['ODs'].values(), cfg.infra['G'], key='time')

    branches = model.get_branches(cfg, path_times)

    assert len(branches) == len(cfg.infra['ODs'])

    # FIXME: only works for binary ATM
    varis = {}
    B = np.vstack([np.eye(cfg.no_ds), np.ones(cfg.no_ds)])
    for k in cfg.infra['edges'].keys():
        varis[k] = variable.Variable(name=k, B=B, values=cfg.scenarios['damage_states'])

    for od, value in branches.items():

        values = [np.inf] + sorted([y for _, y in path_times[cfg.infra['ODs'][od]]], reverse=True)
        varis[od] = variable.Variable(name=od, B=np.eye(len(values)), values=values)

        variables = {k: varis[k] for k in cfg.infra['edges'].keys()}
        c = branch.get_cmat_from_branches(value, variables)

        np.testing.assert_array_equal(c, expected[od])


def test_model_given_od_scen(cmatrix_road):

    expected = cmatrix_road

    cfg = config.Config(HOME.joinpath('./config_roads.json'))

    path_times = trans.get_all_paths_and_times(cfg.infra['ODs'].values(), cfg.infra['G'], key='time')

    branches = model.get_branches(cfg, path_times)

    od_scen_pairs = itertools.product(cfg.infra['ODs'].keys(), cfg.scenarios['scenarios'].keys())

    for od, scen in od_scen_pairs:

        cpms, varis = model.model_given_od_scen(cfg, path_times, od, scen, branches[od])

        assert len(cpms) == 7

        np.testing.assert_array_equal(cpms[od].C, expected[od])


