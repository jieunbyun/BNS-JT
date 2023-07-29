import numpy as np
import pytest

from pathlib import Path

from BNS_JT import config, model


HOME = Path(__file__).absolute().parent

expected_disconn = np.array([0.0096, 0.0011, 0.2102, 0.2102])
expected_delay = np.array([0.0583, 0.0052, 0.4795, 0.4382])
expected_damage = np.array([0.1610,  1,  1,  0.0002,   0,  0.4382])


@pytest.fixture()
def setup_road():

    cfg = config.Config(HOME.joinpath('./config_roads.json'))

    cpms, varis = model.setup_model(cfg)

    return cpms, varis, cfg


def test_setup_model(setup_road):

    cpms, varis, _ = setup_road

    expected = np.array([[0,2,0,0,2,2,2],
                         [0,0,0,1,2,2,2],
                         [1,1,0,1,2,2,2],
                         [2,2,1,2,2,2,2]])

    np.testing.assert_array_equal(cpms['od1'].C, expected)


def test_compute_prob1(setup_road):

    cpms, varis, cfg = setup_road

    var_elim = list(cfg.infra['edges'].keys())

    for i, k in enumerate(['od1', 'od2', 'od3', 'od4']):
        prob, _ = model.compute_prob(cfg, cpms, varis, var_elim, k, 0, flag=True)

        assert expected_disconn[i] == pytest.approx(prob, abs=0.0001)


def test_compute_prob2(setup_road):

    cpms, varis, cfg = setup_road

    var_elim = list(cfg.infra['edges'].keys())

    for i, k in enumerate(['od1', 'od2', 'od3', 'od4']):
        prob, _ = model.compute_prob(cfg, cpms, varis, var_elim, k, 2, flag=False)

        assert expected_delay[i] == pytest.approx(prob, abs=0.0001)


