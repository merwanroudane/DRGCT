"""Low-level helpers shared by every module of :mod:`drgct`.

The functions here implement the *bookkeeping* part of Algorithm 1 of
Hui, Liu and Song (2025): reorganising a bivariate time series
``{(X_t, Y_t)}_{t=1}^n`` into the regressor blocks

    Y_{t-1} = (Y_{t-1}, ..., Y_{t-q})'      (q lags of the caused series)
    X_{t-1} = (X_{t-1}, ..., X_{t-p})'      (p lags of the causing series)
    W_{t-1} = (X_{t-1}', Y_{t-1}')'         (the full information set)

for ``t = q+1, ..., n`` under the maintained convention ``p <= q``.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np

__all__ = [
    "set_seed",
    "as_series",
    "LagDesign",
    "build_lag_design",
    "zscore",
    "rademacher",
    "mammen",
    "draw_multipliers",
    "check_stationarity",
]


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int | None) -> np.random.Generator:
    """Seed Python, NumPy and (if available) PyTorch, and return a Generator.

    Parameters
    ----------
    seed : int or None
        Master seed.  ``None`` leaves the global generators untouched and
        returns a fresh unseeded :class:`numpy.random.Generator`.

    Returns
    -------
    numpy.random.Generator
        The generator that the estimation routines should use for every
        non-torch random draw (``(mu, nu)`` pairs, bootstrap multipliers,
        Monte-Carlo innovations).
    """
    if seed is None:
        return np.random.default_rng()

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:  # torch is a hard dependency, but keep utils importable without it
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch always present in practice
        pass
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- #
# Input coercion
# --------------------------------------------------------------------------- #
def as_series(v, name: str = "series") -> np.ndarray:
    """Coerce ``v`` to a finite 1-D float64 array.

    Accepts lists, tuples, NumPy arrays, ``pandas.Series`` and single-column
    ``pandas.DataFrame`` objects.
    """
    arr = np.asarray(getattr(v, "values", v), dtype=float)
    arr = np.squeeze(arr)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}.")
    if arr.size < 10:
        raise ValueError(f"{name} has only {arr.size} observations; need at least 10.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"{name} contains NaN or inf.  Clean the series before testing "
            "(drgct.datasets.to_percentage_changes drops non-finite rows for you)."
        )
    return np.ascontiguousarray(arr)


# --------------------------------------------------------------------------- #
# Lag design
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LagDesign:
    """Container for the reorganised sample of Step 1(a) / Step 2(a).

    Attributes
    ----------
    y : ndarray, shape (n_eff,)
        The response ``Y_t`` for ``t = q+1, ..., n``.
    ylag : ndarray, shape (n_eff, q)
        ``Y_{t-1} = (Y_{t-1}, ..., Y_{t-q})'`` stacked row-wise.
    xlag : ndarray, shape (n_eff, p)
        ``X_{t-1} = (X_{t-1}, ..., X_{t-p})'`` stacked row-wise.
    p, q : int
        Lag orders of the causing and caused series.
    n : int
        Length of the original sample.
    n_eff : int
        Effective sample size ``n - q`` used in the ``1/sqrt(n-q)`` scaling
        of equations (7)-(9) of the paper.
    t_index : ndarray of int
        0-based positions in the original series of the retained ``Y_t``.
    """

    y: np.ndarray
    ylag: np.ndarray
    xlag: np.ndarray
    p: int
    q: int
    n: int
    n_eff: int
    t_index: np.ndarray

    @property
    def w(self) -> np.ndarray:
        """``W_{t-1} = (X_{t-1}', Y_{t-1}')'`` -- shape ``(n_eff, p + q)``."""
        return np.hstack([self.xlag, self.ylag])

    @property
    def dim_w(self) -> int:
        return self.p + self.q


def build_lag_design(x, y, p: int, q: int, *, allow_p_gt_q: bool = True) -> LagDesign:
    """Reorganise ``{(X_t, Y_t)}`` into ``{Y_t, Y_{t-1}, X_{t-1}}_{t=q+1}^n``.

    This is Step 1(a) and Step 2(a) of Algorithm 1.

    Parameters
    ----------
    x, y : array_like
        The candidate causing series ``X_t`` and the caused series ``Y_t``.
        They must have the same length and be aligned in calendar time.
    p : int
        Number of lags of ``X`` entering the information set.
    q : int
        Number of lags of ``Y`` entering the information set.
    allow_p_gt_q : bool, default True
        The paper assumes ``p <= q``.  When ``p > q`` the effective sample is
        started at ``t = max(p, q) + 1`` instead, which keeps the estimator
        well defined; set to ``False`` to raise instead.

    Returns
    -------
    LagDesign
    """
    x = as_series(x, "x")
    y = as_series(y, "y")
    if x.size != y.size:
        raise ValueError(f"x and y must be the same length ({x.size} vs {y.size}).")
    p, q = int(p), int(q)
    if p < 1 or q < 1:
        raise ValueError("p and q must be >= 1.")
    if p > q and not allow_p_gt_q:
        raise ValueError("The paper maintains p <= q; pass allow_p_gt_q=True to relax.")

    n = x.size
    start = max(p, q)  # 0-based index of the first usable Y_t
    n_eff = n - start
    if n_eff < 30:
        raise ValueError(
            f"Effective sample size is {n_eff} (n={n}, max(p,q)={start}); "
            "at least 30 usable observations are required."
        )

    t_index = np.arange(start, n)
    ylag = np.column_stack([y[t_index - j] for j in range(1, q + 1)])
    xlag = np.column_stack([x[t_index - k] for k in range(1, p + 1)])
    return LagDesign(
        y=y[t_index].copy(),
        ylag=np.ascontiguousarray(ylag),
        xlag=np.ascontiguousarray(xlag),
        p=p,
        q=q,
        n=n,
        n_eff=n_eff,
        t_index=t_index,
    )


# --------------------------------------------------------------------------- #
# Scaling
# --------------------------------------------------------------------------- #
def zscore(a: np.ndarray, *, ddof: int = 0, eps: float = 1e-12):
    """Column-wise standardisation returning ``(z, mean, scale)``.

    Standardising the inputs stabilises the training of the MLP and the MDN.
    It leaves the *p-value* of the DRGCT unchanged: rescaling ``Y`` multiplies
    ``KS_n`` and every bootstrap replicate ``KS_n^*`` by the same constant,
    so the rank of ``KS_n`` in the bootstrap distribution is invariant.
    """
    a = np.atleast_2d(np.asarray(a, dtype=float))
    mean = a.mean(axis=0)
    scale = a.std(axis=0, ddof=ddof)
    scale = np.where(scale < eps, 1.0, scale)
    return (a - mean) / scale, mean, scale


# --------------------------------------------------------------------------- #
# Bootstrap multipliers  (Step 4(a) of Algorithm 1)
# --------------------------------------------------------------------------- #
def rademacher(rng: np.random.Generator, size) -> np.ndarray:
    """i.i.d. +/-1 with equal probability: mean 0, variance 1, bounded support."""
    return rng.integers(0, 2, size=size).astype(float) * 2.0 - 1.0


def mammen(rng: np.random.Generator, size) -> np.ndarray:
    """Mammen's (1993) two-point distribution: mean 0, variance 1, bounded, skewed."""
    root5 = np.sqrt(5.0)
    a, b = -(root5 - 1.0) / 2.0, (root5 + 1.0) / 2.0
    prob_b = (root5 - 1.0) / (2.0 * root5)
    u = rng.random(size)
    return np.where(u < prob_b, b, a)


def draw_multipliers(rng: np.random.Generator, size, kind: str = "rademacher") -> np.ndarray:
    """Draw the multipliers ``{xi_t}`` used in the bootstrap process (12).

    Theorem 4 requires ``xi_t`` i.i.d. with zero mean, unit variance and
    **bounded support**, independent of the data.  ``"rademacher"`` (default)
    and ``"mammen"`` satisfy all three; ``"normal"`` is offered for
    experimentation but violates the bounded-support condition.
    """
    kind = str(kind).lower()
    if kind in ("rademacher", "rad", "two-point"):
        return rademacher(rng, size)
    if kind == "mammen":
        return mammen(rng, size)
    if kind in ("normal", "gaussian"):
        return rng.standard_normal(size)
    raise ValueError(f"Unknown multiplier kind {kind!r}; use rademacher/mammen/normal.")


# --------------------------------------------------------------------------- #
# Pre-flight diagnostics
# --------------------------------------------------------------------------- #
def check_stationarity(series, name: str = "series", *, alpha: float = 0.05) -> dict:
    """Augmented Dickey-Fuller and KPSS pre-test for Assumption 1.

    Assumption 1 of the paper requires strict stationarity and exponential
    beta-mixing.  Neither is testable, but a unit root is the failure mode
    that actually bites in applied work, so we screen for it.

    Returns
    -------
    dict
        ``{'adf_stat', 'adf_pvalue', 'kpss_stat', 'kpss_pvalue',
        'stationary', 'message'}``.  ``stationary`` is ``True`` only when ADF
        rejects *and* KPSS fails to reject at level ``alpha``.
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    s = as_series(series, name)
    adf_stat, adf_p = adfuller(s, autolag="AIC")[:2]
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_stat, kpss_p = kpss(s, regression="c", nlags="auto")[:2]

    ok = (adf_p < alpha) and (kpss_p > alpha)
    if ok:
        msg = f"{name}: stationary (ADF p={adf_p:.3f}, KPSS p={kpss_p:.3f})."
    else:
        msg = (
            f"{name}: stationarity NOT established (ADF p={adf_p:.3f}, "
            f"KPSS p={kpss_p:.3f}).  Assumption 1 of Hui et al. (2025) may fail; "
            "consider differencing or a percentage-change transform."
        )
    return {
        "adf_stat": float(adf_stat),
        "adf_pvalue": float(adf_p),
        "kpss_stat": float(kpss_stat),
        "kpss_pvalue": float(kpss_p),
        "stationary": bool(ok),
        "message": msg,
    }
