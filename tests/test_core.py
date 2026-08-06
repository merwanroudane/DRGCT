"""Tests for the DRGCT itself: statistic construction, bootstrap, invariances."""

from __future__ import annotations

import numpy as np
import pytest

from drgct import DRGCTResult, drgc_test
from drgct.dgp import simulate_dgp
from drgct.nets import MDNConfig, MLPConfig

FAST = dict(
    B=199,
    L=8,
    M=8,
    G=4,
    mlp=MLPConfig(epochs=60, patience=25),
    mdn=MDNConfig(epochs=80, patience=30),
)


def test_result_shapes_and_contract():
    s = simulate_dgp("S1", n=200, lag=1, rng=0)
    r = drgc_test(s.x, s.y, lag=1, seed=0, **FAST)

    assert isinstance(r, DRGCTResult)
    assert r.n == 200 and r.n_eff == 199
    assert r.S_hat.shape == (FAST["L"],)
    assert r.influence.shape == (199, FAST["L"])
    assert r.boot_stats.shape == (FAST["B"],)
    assert r.mu.shape == (FAST["L"], 1) and r.nu.shape == (FAST["L"], 1)
    assert 0.0 <= r.pvalue <= 1.0
    assert r.ks_stat >= 0.0
    assert set(r.critical_values) == {0.10, 0.05, 0.01}
    assert "KS_n statistic" in r.summary()
    assert r.to_frame().shape[0] == 1


def test_statistic_matches_its_definition():
    """Recompute KS_n by hand from the stored pieces (equations (9)-(10))."""
    s = simulate_dgp("P1", n=200, lag=2, rng=1)
    r = drgc_test(s.x, s.y, lag=2, seed=3, **FAST)

    manual = r.influence.sum(axis=0) / np.sqrt(r.n_eff)
    np.testing.assert_allclose(manual, r.S_hat, rtol=1e-10, atol=1e-12)
    ks = np.max(np.maximum(np.abs(r.S_hat.real), np.abs(r.S_hat.imag)))
    np.testing.assert_allclose(ks, r.ks_stat, rtol=1e-12)


def test_pvalue_equals_bootstrap_exceedance_frequency():
    s = simulate_dgp("S2", n=200, lag=1, rng=2)
    r = drgc_test(s.x, s.y, lag=1, seed=5, **FAST)
    np.testing.assert_allclose(r.pvalue, np.mean(r.boot_stats >= r.ks_stat))
    np.testing.assert_allclose(r.critical_values[0.05], np.quantile(r.boot_stats, 0.95))


def test_seed_makes_the_test_exactly_reproducible():
    s = simulate_dgp("S1", n=200, lag=1, rng=4)
    a = drgc_test(s.x, s.y, lag=1, seed=123, **FAST)
    b = drgc_test(s.x, s.y, lag=1, seed=123, **FAST)
    assert a.ks_stat == pytest.approx(b.ks_stat, rel=1e-12)
    assert a.pvalue == pytest.approx(b.pvalue, rel=1e-12)


def test_pvalue_is_invariant_to_affine_rescaling():
    """Rescaling Y multiplies KS_n and every KS*_n by the same factor."""
    s = simulate_dgp("P2", n=250, lag=1, rng=6)
    a = drgc_test(s.x, s.y, lag=1, seed=9, **FAST)
    b = drgc_test(s.x, 7.5 * s.y + 3.0, lag=1, seed=9, **FAST)
    assert a.pvalue == pytest.approx(b.pvalue, abs=0.02)


def test_naive_variant_drops_the_correction_term():
    s = simulate_dgp("S1", n=200, lag=1, rng=8)
    r = drgc_test(s.x, s.y, lag=1, seed=11, doubly_robust=False, **FAST)
    assert r.doubly_robust is False
    np.testing.assert_allclose(r.phi_hat, 0.0)  # phihat dropped from equation (8)


def test_clear_alternative_is_rejected():
    """A strong, purely nonlinear dependence must be detected."""
    rng = np.random.default_rng(0)
    n = 600
    x = rng.normal(size=n)
    y = np.r_[0.0, 2.0 * np.sin(3.0 * x[:-1])] + rng.normal(0, 0.3, n)
    r = drgc_test(x, y, lag=1, seed=2, B=399, L=15, M=15, G=6)
    assert r.pvalue < 0.05


def test_independent_series_are_not_rejected():
    rng = np.random.default_rng(11)
    n = 600
    x = rng.normal(size=n)
    y = np.zeros(n)
    e = rng.normal(0, 0.7, n)
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + e[t]
    r = drgc_test(x, y, lag=1, seed=4, B=399, L=15, M=15, G=6)
    assert r.pvalue > 0.05


def test_input_validation():
    s = simulate_dgp("S1", n=120, lag=1, rng=0)
    with pytest.raises(ValueError):
        drgc_test(s.x, s.y)  # neither lag nor (p, q)
    with pytest.raises(ValueError):
        drgc_test(s.x, s.y[:-5], lag=1)  # length mismatch
    with pytest.raises(ValueError):
        drgc_test(s.x, s.y, lag=200)  # effective sample too small
