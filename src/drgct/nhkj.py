r"""Smoothing-based nonparametric Granger causality test (the NHKJ benchmark).

Tables 3 and 4 of Hui, Liu and Song (2025) compare the DRGCT against the
consistent nonparametric causality test of

    Nishiyama, Y., Hitomi, K., Kawasaki, Y. and Jeong, K. (2011),
    "A consistent nonparametric test for nonlinear causality --
    Specification in time series regression",
    Journal of Econometrics 165, 112-127,

abbreviated **NHKJ** in the paper.

Implementation note (please read before quoting numbers)
--------------------------------------------------------
This module implements a *kernel-smoothing conditional-moment test of the
NHKJ class*, built in the standard Zheng (1996) / Fan and Li (1996) degenerate
U-statistic form, with the two configuration choices the DRGCT paper states
for its benchmark:

* a **fourth-order Gaussian kernel**, ``K4(u) = 0.5 (3 - u^2) phi(u)``, applied
  as a product kernel over the ``d = p + q`` coordinates of ``W_{t-1}``
  (needed to keep the bias under control when ``d`` is as large as 5), and
* the bandwidth family ``h = c * n^{-0.15}`` with ``c`` set per DGP and lag,
  exactly as tabulated in Section 4 of the paper.

It is a faithful member of the same family of tests and reproduces the
qualitative findings the paper reports for NHKJ (severe undersizing as ``d``
grows, and power collapsing with the lag order).  It is *not* a line-by-line
transcription of the estimator in the original NHKJ article, and small
numerical differences from Tables 3-4 should be expected.  Treat it as the
"smoothing-based nonparametric benchmark", which is the role it plays here.

The statistic
-------------
With ``epshat_t = Y_t - mhat_{-t}(Y_{t-1})`` the leave-one-out Nadaraya-Watson
residuals from regressing ``Y_t`` on ``Y_{t-1}`` only, and
``W_{t-1} = (X_{t-1}', Y_{t-1}')'``,

    Gammahat = 1 / (N (N-1) h^d) * sum_{t != s} epshat_t epshat_s K((W_t - W_s)/h)
    sigmahat^2 = 2 / (N (N-1) h^d) * sum_{t != s} epshat_t^2 epshat_s^2 K^2((W_t - W_s)/h)
    T_n = N h^{d/2} Gammahat / sigmahat   ->d   N(0, 1)   under H0,

and ``H0`` is rejected for large positive ``T_n`` (one-sided).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from .utils import as_series, build_lag_design, zscore

__all__ = ["NHKJResult", "gaussian_kernel", "gaussian_kernel4", "nhkj_test"]

_SQRT_2PI = np.sqrt(2.0 * np.pi)


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
def gaussian_kernel(u: np.ndarray) -> np.ndarray:
    """Second-order (standard) Gaussian kernel ``phi(u)``."""
    return np.exp(-0.5 * u * u) / _SQRT_2PI


def gaussian_kernel4(u: np.ndarray) -> np.ndarray:
    """Fourth-order Gaussian kernel ``K4(u) = 0.5 (3 - u^2) phi(u)``.

    Integrates to one, has zero second moment, and a non-zero fourth moment;
    this is the bias-reduction device the DRGCT paper adopts for the NHKJ
    benchmark when the conditioning dimension reaches 4 or 5.
    """
    return 0.5 * (3.0 - u * u) * gaussian_kernel(u)


_KERNELS = {2: gaussian_kernel, 4: gaussian_kernel4}


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class NHKJResult:
    """Output of :func:`nhkj_test`."""

    stat: float
    pvalue: float
    gamma: float
    sigma: float
    bandwidth: float
    bandwidth_const: float
    kernel_order: int
    p: int
    q: int
    lag: int
    n: int
    n_eff: int
    dim_w: int
    alpha: float
    reject: bool
    direction: str = "X -> Y"
    elapsed: float = 0.0
    settings: dict = field(default_factory=dict)

    @property
    def stars(self) -> str:
        if self.pvalue < 0.01:
            return "***"
        if self.pvalue < 0.05:
            return "**"
        if self.pvalue < 0.10:
            return "*"
        return ""

    def summary(self) -> str:
        return "\n".join(
            [
                "=" * 72,
                "  Smoothing-based nonparametric Granger causality test (NHKJ class)",
                "  Nishiyama, Hitomi, Kawasaki & Jeong (2011), J. Econometrics 165",
                "=" * 72,
                f"  H0 : {self.direction.split('->')[0].strip()} does not Granger-cause "
                f"{self.direction.split('->')[1].strip()} in mean",
                f"  Lag orders          : p = {self.p}, q = {self.q}   (dim W = {self.dim_w})",
                f"  Sample              : n = {self.n},  effective = {self.n_eff}",
                f"  Kernel              : order {self.kernel_order} Gaussian product kernel",
                f"  Bandwidth           : h = {self.bandwidth_const:g} * n^-0.15 "
                f"= {self.bandwidth:.5f}",
                "-" * 72,
                f"  Gamma_hat           : {self.gamma:.6e}",
                f"  sigma_hat           : {self.sigma:.6e}",
                f"  T_n  (~ N(0,1))     : {self.stat:.4f}",
                f"  one-sided p-value   : {self.pvalue:.4f} {self.stars}",
                f"  Decision at {100 * self.alpha:g}%    : "
                f"{'REJECT H0' if self.reject else 'DO NOT REJECT H0'}",
                "=" * 72,
            ]
        )

    def print(self) -> None:
        print(self.summary())

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "lag": self.lag,
            "p": self.p,
            "q": self.q,
            "n": self.n,
            "n_eff": self.n_eff,
            "stat": self.stat,
            "pvalue": self.pvalue,
            "reject": self.reject,
            "stars": self.stars,
            "bandwidth": self.bandwidth,
            "elapsed": self.elapsed,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<NHKJResult {self.direction} lag={self.lag} T={self.stat:.3f} p={self.pvalue:.4f}>"


# --------------------------------------------------------------------------- #
# Kernel machinery
# --------------------------------------------------------------------------- #
def _product_kernel_block(a: np.ndarray, b: np.ndarray, h: float, kfun) -> np.ndarray:
    """``K((a_i - b_j)/h)`` as a product kernel; returns an ``(len(a), len(b))`` block."""
    diff = (a[:, None, :] - b[None, :, :]) / h
    return np.prod(kfun(diff), axis=-1)


def _loo_nw_residuals(ylag: np.ndarray, y: np.ndarray, h: float, kfun, chunk: int) -> np.ndarray:
    """Leave-one-out Nadaraya-Watson residuals of ``y`` on ``ylag``."""
    n = ylag.shape[0]
    num = np.zeros(n)
    den = np.zeros(n)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        k = _product_kernel_block(ylag[start:stop], ylag, h, kfun)
        rows = np.arange(start, stop)
        k[rows - start, rows] = 0.0  # leave one out
        num[start:stop] = k @ y
        den[start:stop] = k.sum(axis=1)
    # A fourth-order kernel can produce near-zero (even negative) denominators;
    # trim those observations rather than letting them explode the residual.
    safe = np.abs(den) > 1e-8
    mhat = np.zeros(n)
    mhat[safe] = num[safe] / den[safe]
    mhat[~safe] = y.mean()
    return y - mhat


def _degenerate_ustat(w: np.ndarray, eps: np.ndarray, h: float, kfun, chunk: int):
    """Return ``(Gammahat, sigmahat)`` of the Zheng/Fan-Li statistic."""
    n, d = w.shape
    hd = h**d
    s_gamma = 0.0
    s_var = 0.0
    eps2 = eps * eps
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        k = _product_kernel_block(w[start:stop], w, h, kfun)
        rows = np.arange(start, stop)
        k[rows - start, rows] = 0.0
        s_gamma += float(eps[start:stop] @ (k @ eps))
        s_var += float(eps2[start:stop] @ ((k * k) @ eps2))
    denom = n * (n - 1) * hd
    gamma = s_gamma / denom
    sigma2 = 2.0 * s_var / denom
    return gamma, np.sqrt(max(sigma2, 1e-300))


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def nhkj_test(
    x,
    y,
    lag: int | None = None,
    *,
    p: int | None = None,
    q: int | None = None,
    bandwidth_const: float | None = None,
    bandwidth: float | None = None,
    bandwidth_exponent: float = -0.15,
    kernel_order: int = 4,
    alpha: float = 0.05,
    standardize: bool = True,
    chunk: int = 512,
    x_name: str = "X",
    y_name: str = "Y",
) -> NHKJResult:
    r"""Smoothing-based nonparametric test that ``x`` does not Granger-cause ``y``.

    Parameters
    ----------
    x, y : array_like
        Aligned stationary series.
    lag : int, optional
        Common lag order ``p = q = lag``.
    p, q : int, optional
        Separate lag orders.
    bandwidth_const : float, optional
        ``c`` in ``h = c * n^{-0.15}``.  If omitted, the paper's schedule is
        used: ``c = 2.5`` for ``lag = 1``, ``3.0`` for ``lag in {2, 3}``,
        ``3.5`` for ``lag in {4, 5}``, and ``4.0`` beyond.  (Section 4 uses
        exactly these for DGP S1/P1/P2 and shifts them up by 0.5 for the
        exponential-mean designs S2/P3; pass the value explicitly to match a
        specific row of Table 3.)
    bandwidth : float, optional
        Set the bandwidth directly, overriding ``bandwidth_const``.
    bandwidth_exponent : float, default -0.15
        The exponent in ``h = c * n^{exponent}``.
    kernel_order : {2, 4}, default 4
        Order of the Gaussian product kernel.
    alpha : float, default 0.05
    standardize : bool, default True
        z-score each coordinate of ``W_{t-1}`` before smoothing, so that a
        single scalar bandwidth is sensible.
    chunk : int, default 512
        Row-block size for the pairwise kernel computations; lower it if you
        run into memory pressure at large ``n``.
    x_name, y_name : str

    Returns
    -------
    NHKJResult

    Examples
    --------
    >>> import numpy as np
    >>> from drgct import nhkj_test
    >>> rng = np.random.default_rng(3)
    >>> n = 400
    >>> x = rng.normal(size=n)
    >>> y = np.r_[0.0, np.sin(x[:-1])] + rng.normal(0, 0.5, n)
    >>> nhkj_test(x, y, lag=1).pvalue < 0.10
    True
    """
    t0 = time.perf_counter()
    if lag is None and (p is None or q is None):
        raise ValueError("Provide either lag=... or both p=... and q=... .")
    if lag is not None:
        p = q = int(lag)
    p, q = int(p), int(q)
    lag = max(p, q)
    if kernel_order not in _KERNELS:
        raise ValueError("kernel_order must be 2 or 4.")
    kfun = _KERNELS[kernel_order]

    design = build_lag_design(as_series(x, "x"), as_series(y, "y"), p=p, q=q)
    n_eff = design.n_eff

    if standardize:
        ylag, _, _ = zscore(design.ylag)
        xlag, _, _ = zscore(design.xlag)
        yy = (design.y - design.y.mean()) / max(design.y.std(), 1e-12)
    else:
        ylag, xlag, yy = design.ylag, design.xlag, design.y
    w = np.hstack([xlag, ylag])

    if bandwidth is None:
        if bandwidth_const is None:
            bandwidth_const = 2.5 if lag == 1 else 3.0 if lag <= 3 else 3.5 if lag <= 5 else 4.0
        h = float(bandwidth_const) * n_eff**bandwidth_exponent
    else:
        h = float(bandwidth)
        bandwidth_const = h / (n_eff**bandwidth_exponent)

    eps = _loo_nw_residuals(ylag, yy, h, kfun, chunk)
    gamma, sigma = _degenerate_ustat(w, eps, h, kfun, chunk)
    d = w.shape[1]
    stat = float(n_eff * h ** (d / 2.0) * gamma / sigma)
    pvalue = float(norm.sf(stat))

    return NHKJResult(
        stat=stat,
        pvalue=pvalue,
        gamma=float(gamma),
        sigma=float(sigma),
        bandwidth=h,
        bandwidth_const=float(bandwidth_const),
        kernel_order=int(kernel_order),
        p=p,
        q=q,
        lag=lag,
        n=design.n,
        n_eff=n_eff,
        dim_w=d,
        alpha=float(alpha),
        reject=bool(pvalue < alpha),
        direction=f"{x_name} -> {y_name}",
        elapsed=time.perf_counter() - t0,
        settings={
            "standardize": standardize,
            "bandwidth_exponent": bandwidth_exponent,
            "chunk": chunk,
        },
    )
