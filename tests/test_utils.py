"""Tests for the lag bookkeeping, scaling and multiplier utilities."""

from __future__ import annotations

import numpy as np
import pytest

from drgct.utils import (
    as_series,
    build_lag_design,
    check_stationarity,
    draw_multipliers,
    set_seed,
    zscore,
)


def test_as_series_rejects_bad_input():
    with pytest.raises(ValueError):
        as_series(np.arange(5))  # too short
    with pytest.raises(ValueError):
        as_series(np.r_[np.arange(20.0), np.nan])
    with pytest.raises(ValueError):
        as_series(np.zeros((20, 2)))


def test_lag_design_alignment():
    """Y_t, Y_{t-1}, X_{t-1} must line up exactly as in Algorithm 1 Step 1(a)."""
    n, p, q = 50, 2, 3
    x = np.arange(n, dtype=float)
    y = 100.0 + np.arange(n, dtype=float)
    d = build_lag_design(x, y, p=p, q=q)

    assert d.n_eff == n - q
    assert d.ylag.shape == (n - q, q)
    assert d.xlag.shape == (n - q, p)
    # first retained observation is t = q (0-based)
    assert d.y[0] == y[q]
    np.testing.assert_allclose(d.ylag[0], [y[q - 1], y[q - 2], y[q - 3]])
    np.testing.assert_allclose(d.xlag[0], [x[q - 1], x[q - 2]])
    # W = (X', Y')
    assert d.w.shape == (n - q, p + q)
    assert d.dim_w == p + q


def test_lag_design_p_greater_than_q():
    d = build_lag_design(np.arange(60.0), np.arange(60.0), p=5, q=2)
    assert d.n_eff == 55
    with pytest.raises(ValueError):
        build_lag_design(np.arange(60.0), np.arange(60.0), p=5, q=2, allow_p_gt_q=False)


def test_zscore():
    a = np.random.default_rng(0).normal(3.0, 7.0, (200, 3))
    z, m, s = zscore(a)
    np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(z.std(axis=0), 1, atol=1e-12)
    np.testing.assert_allclose(z * s + m, a)


@pytest.mark.parametrize("kind", ["rademacher", "mammen", "normal"])
def test_multipliers_have_zero_mean_unit_variance(kind):
    rng = np.random.default_rng(1)
    xi = draw_multipliers(rng, 400_000, kind)
    assert abs(xi.mean()) < 0.01
    assert abs(xi.var() - 1.0) < 0.02
    if kind in ("rademacher", "mammen"):
        assert np.isfinite(xi).all() and np.abs(xi).max() < 5.0  # bounded support


def test_set_seed_is_reproducible():
    a = set_seed(42).normal(size=5)
    b = set_seed(42).normal(size=5)
    np.testing.assert_allclose(a, b)


def test_check_stationarity_flags_a_random_walk():
    rng = np.random.default_rng(7)
    rw = np.cumsum(rng.normal(size=400))
    assert check_stationarity(rw)["stationary"] is False
    assert check_stationarity(rng.normal(size=400))["stationary"] is True
