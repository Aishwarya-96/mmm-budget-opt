import numpy as np
import pytest

from mmm.simulate import SimConfig, simulate
from mmm.transforms import geometric_adstock, hill_saturation


def test_adstock_zero_decay_is_identity():
    x = np.array([1.0, 0.0, 3.0, 2.0])
    np.testing.assert_allclose(geometric_adstock(x, 0.0), x)


def test_adstock_carries_a_single_pulse_forward():
    x = np.array([10.0, 0, 0, 0])
    out = geometric_adstock(x, 0.5, normalise=False)
    np.testing.assert_allclose(out, [10, 5, 2.5, 1.25])


def test_adstock_normalised_preserves_steady_state_level():
    x = np.full(200, 7.0)
    out = geometric_adstock(x, 0.7, max_lag=20)
    assert out[-1] == pytest.approx(7.0)


def test_adstock_rejects_bad_decay():
    with pytest.raises(ValueError):
        geometric_adstock(np.ones(5), 1.0)


def test_hill_is_half_at_half_saturation():
    assert hill_saturation(np.array([4.0]), 4.0, slope=2.0)[0] == pytest.approx(0.5)


def test_hill_is_monotone_and_bounded():
    y = hill_saturation(np.linspace(0, 100, 50), 10.0, slope=1.5)
    assert np.all(np.diff(y) >= 0)
    assert y.min() >= 0 and y.max() < 1


def test_simulation_shapes_and_positive_roi():
    obs, truth = simulate(SimConfig(n_geos=3, n_weeks=60))
    assert len(obs) == 3 * 60
    assert {"geo", "week", "population", "sales"}.issubset(obs.columns)
    assert (truth["true_roi"] > 0).all()


def test_endogenous_spend_correlates_with_lagged_sales():
    def corr(share):
        obs, _ = simulate(SimConfig(n_geos=5, n_weeks=120, endogenous_share=share))
        obs = obs.sort_values(["geo", "week"])
        obs["lag_sales"] = obs.groupby("geo")["sales"].shift()
        obs["spend_rel"] = obs["spend_search"] / obs.groupby("geo")["spend_search"].transform("mean")
        obs["sales_rel"] = obs["lag_sales"] / obs.groupby("geo")["sales"].transform("mean")
        return obs[["spend_rel", "sales_rel"]].dropna().corr().iloc[0, 1]

    assert corr(0.6) > corr(0.0) + 0.05
