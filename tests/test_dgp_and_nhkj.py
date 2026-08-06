"""Tests for the Table 1/2 designs and the smoothing-based benchmark."""

from __future__ import annotations

import numpy as np
import pytest

from drgct.dgp import (
    DGP_NAMES,
    PARAMETERS,
    POWER_DGPS,
    SIZE_DGPS,
    dgp_parameters,
    dgp_table,
    parameter_table,
    simulate_dgp,
)
from drgct.nhkj import gaussian_kernel, gaussian_kernel4, nhkj_test


# --------------------------------------------------------------------------- #
# Designs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", DGP_NAMES)
@pytest.mark.parametrize("lag", [1, 3, 5])
def test_every_design_simulates_finite_series(name, lag):
    s = simulate_dgp(name, n=300, lag=lag, rng=0)
    assert s.x.shape == (300,) and s.y.shape == (300,)
    assert np.all(np.isfinite(s.x)) and np.all(np.isfinite(s.y))
    assert s.causal == (name in POWER_DGPS)


def test_table2_parameters_match_the_paper():
    """Spot-check the Table 2 entries the paper prints explicitly."""
    p1 = dgp_parameters(1)
    assert p1["a"] == [1.0] and p1["b"] == [-1.0] and p1["c"] == [-1.0]
    assert p1["a_exp"] == 0.5 and p1["c_scalar"] == 1.0 and p1["a0"] == 0.5

    p2 = dgp_parameters(2)
    assert p2["a"] == [0.5, -0.5] and p2["b"] == [-0.5, 0.5]
    assert p2["a_exp"] == 0.25 and p2["c_scalar"] == 0.6 and p2["a0"] == 0.4

    p5 = dgp_parameters(5)
    assert p5["a"] == [0.25, -0.25, 0.25, 0.25, -0.25]
    assert p5["b"] == [-0.25, 0.25, 0.25, -0.25, 0.25]
    assert p5["c"] == [-0.25, 0.25, -0.25, 0.25, -0.25]
    assert p5["a_exp"] == 0.125 and p5["c_scalar"] == 0.5
    assert p5["a0"] == pytest.approx(1 / 3)

    for lag, block in PARAMETERS.items():
        assert len(block["a"]) == len(block["b"]) == len(block["c"]) == lag


def test_null_designs_have_no_x_in_the_y_equation():
    """S1 and S2 must give the same Y path whatever X does."""
    for name in SIZE_DGPS:
        a = simulate_dgp(name, n=200, lag=2, rng=np.random.default_rng(5))
        b = simulate_dgp(name, n=200, lag=2, rng=np.random.default_rng(5))
        np.testing.assert_allclose(a.y, b.y)
        # Y must be a deterministic function of its own past and eps2 only:
        # perturbing the b-coefficients (which drive X) leaves Y untouched.
        c = simulate_dgp(name, n=200, lag=2, rng=np.random.default_rng(5),
                         params={"b": [0.1, -0.1]})
        np.testing.assert_allclose(a.y, c.y)
        assert not np.allclose(a.x, c.x)


def test_alternative_designs_do_depend_on_x():
    for name in POWER_DGPS:
        a = simulate_dgp(name, n=200, lag=2, rng=np.random.default_rng(5))
        c = simulate_dgp(name, n=200, lag=2, rng=np.random.default_rng(5),
                         params={"b": [0.1, -0.1]})
        assert not np.allclose(a.y, c.y)


def test_burn_in_is_discarded():
    s = simulate_dgp("S1", n=100, lag=1, rng=0, burn=500)
    assert s.x.size == 100
    assert abs(s.x[0]) > 0  # not the zero initial condition


def test_innovation_scale_convention():
    a = simulate_dgp("S1", n=4000, lag=1, rng=1, sigma2=0.5, innovation_scale="variance")
    b = simulate_dgp("S1", n=4000, lag=1, rng=1, sigma2=0.5, innovation_scale="sd")
    assert a.x.std() > b.x.std()  # variance 0.5 -> sd 0.707 > 0.5


def test_descriptive_tables_build():
    assert len(dgp_table()) == len(DGP_NAMES)
    pt = parameter_table([1, 2, 3, 4, 5])
    assert len(pt) == 5 and "a0 (P4)" in pt.columns


def test_bad_inputs():
    with pytest.raises(ValueError):
        simulate_dgp("NOPE", n=100, lag=1)
    with pytest.raises(ValueError):
        simulate_dgp("S1", n=100, lag=2, params={"a": [0.5]})


# --------------------------------------------------------------------------- #
# Kernels and the benchmark test
# --------------------------------------------------------------------------- #
def test_kernels_integrate_to_one_and_have_the_right_moments():
    u = np.linspace(-12, 12, 400_001)
    du = u[1] - u[0]
    for k, second_moment in ((gaussian_kernel, 1.0), (gaussian_kernel4, 0.0)):
        vals = k(u)
        assert np.trapz(vals, dx=du) == pytest.approx(1.0, abs=1e-6)
        assert np.trapz(u * vals, dx=du) == pytest.approx(0.0, abs=1e-8)
        assert np.trapz(u**2 * vals, dx=du) == pytest.approx(second_moment, abs=1e-6)


def test_nhkj_detects_a_strong_nonlinear_alternative():
    rng = np.random.default_rng(3)
    n = 500
    x = rng.normal(size=n)
    y = np.r_[0.0, 1.5 * np.sin(2.0 * x[:-1])] + rng.normal(0, 0.4, n)
    assert nhkj_test(x, y, lag=1).pvalue < 0.05


def test_nhkj_does_not_reject_under_independence():
    rng = np.random.default_rng(17)
    n = 500
    x = rng.normal(size=n)
    y = np.zeros(n)
    e = rng.normal(0, 0.7, n)
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + e[t]
    assert nhkj_test(x, y, lag=1).pvalue > 0.05


def test_nhkj_bandwidth_schedule_matches_section_4():
    """h = c * n^{-0.15} with c = 2.5 / 3.0 / 3.5 by lag."""
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=300), rng.normal(size=300)
    assert nhkj_test(x, y, lag=1).bandwidth_const == 2.5
    assert nhkj_test(x, y, lag=2).bandwidth_const == 3.0
    assert nhkj_test(x, y, lag=4).bandwidth_const == 3.5
    r = nhkj_test(x, y, lag=1, bandwidth_const=2.5)
    assert r.bandwidth == pytest.approx(2.5 * r.n_eff**-0.15)


def test_nhkj_result_surface():
    rng = np.random.default_rng(1)
    r = nhkj_test(rng.normal(size=300), rng.normal(size=300), lag=2)
    assert 0.0 <= r.pvalue <= 1.0
    assert r.dim_w == 4
    assert "NHKJ" in r.summary()
    assert set(r.to_dict()) >= {"stat", "pvalue", "reject", "bandwidth"}
