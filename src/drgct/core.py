r"""The doubly robust Granger causality test (DRGCT).

Reference implementation of Algorithm 1 of

    Hui, Y., Liu, C. and Song, X. (2025),
    "Deep learning based doubly robust test for Granger causality",
    arXiv:2509.15798v2 [stat.ME].

Notation follows the paper exactly.

Hypotheses
----------
For a bivariate strictly stationary series ``{(X_t, Y_t)}`` satisfying the
Markov property (1), with ``W_{t-1} = (X_{t-1}', Y_{t-1}')'``,

    H0 :  E[ Y_t - m(Y_{t-1}) | W_{t-1} ] = 0   a.s.,      m(y) = E[Y_t | Y_{t-1} = y]

i.e. ``X`` does not Granger-cause ``Y`` in mean.  Using the generically
comprehensively revealing function ``phi(W, w) = exp(i w'W)`` of Stinchcombe
and White (1998), this is equivalent to equation (3),

    H0 :  E[ (Y_t - m(Y_{t-1})) exp(i w' W_{t-1}) ] = 0  for all w in W,

and -- by Proposition 1 of the paper -- to the *doubly robust* form (6),

    H0 :  E[ (Y_t - m(Y_{t-1})) exp(i mu' Y_{t-1})
              ( exp(i nu' X_{t-1}) - phi(nu | Y_{t-1}) ) ] = 0,

with ``phi(nu | Y_{t-1}) = E[ exp(i nu' X_{t-1}) | Y_{t-1} ]``.

Test statistic
--------------
With ``mhat`` from an MLP (Step 1) and ``phihat`` from an MDN generator
(Step 2), the feasible empirical process (9) is

    Shat_n(mu_l, nu_l) = (n-q)^{-1/2} sum_{t=q+1}^{n}
        (Y_t - mhat(Y_{t-1})) e^{i mu_l' Y_{t-1}}
        ( e^{i nu_l' X_{t-1}} - phihat(nu_l | Y_{t-1}) ),

and the Kolmogorov-Smirnov statistic (10) is

    KS_n = max_{l <= L} max( |Re Shat_n(mu_l, nu_l)| , |Im Shat_n(mu_l, nu_l)| ).

Critical values come from the multiplier bootstrap (12): only the multipliers
``xi_t`` are redrawn, so the two neural networks are trained exactly **once**
per test -- this is the computational pay-off of the doubly robust structure
emphasised in Section 3.4.

Why the ``- phihat`` term matters
---------------------------------
Dropping it gives the naive process (5).  Then ``Shat0_n - S0_n`` is driven by
the *first* power of the MLP error and does not vanish fast enough, so the
process fails to converge to a Gaussian limit and the type I error explodes
with ``n`` (Section 4 of the paper reports sizes of 0.151 at ``n = 1000`` and
0.321 at ``n = 2000`` for DGP S1 with lag 5).  Keeping it makes the bias
depend on the *product* of the two estimation errors -- double robustness.
Pass ``doubly_robust=False`` to reproduce the failure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .nets import (
    MDNConfig,
    MLPConfig,
    fit_conditional_density,
    fit_conditional_mean,
)
from .utils import LagDesign, build_lag_design, draw_multipliers, set_seed, zscore

__all__ = [
    "DRGCTResult",
    "drgc_test",
    "drgc_lag_scan",
    "drgc_both_directions",
    "drgc_stability",
]

_DEFAULT_ALPHAS = (0.10, 0.05, 0.01)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class DRGCTResult:
    """Everything the DRGCT produces, with printing and export helpers.

    Attributes
    ----------
    ks_stat : float
        ``KS_n`` of equation (10).
    pvalue : float
        Bootstrap p-value ``p*_n = B^{-1} sum_b 1{ KS*_{n,b} >= KS_n }``.
    critical_values : dict
        ``{alpha: quantile(KS*, 1-alpha)}`` for ``alpha`` in ``alphas``.
    reject : bool
        ``pvalue < alpha`` at the headline level.
    alpha : float
        Headline significance level.
    p, q, lag : int
        Lag orders actually used.
    n, n_eff : int
        Original and effective sample sizes.
    boot_stats : ndarray, shape (B,)
        The bootstrap replicates ``KS*_{n,b}``.
    S_hat : ndarray of complex, shape (L,)
        ``Shat_n(mu_l, nu_l)``.
    mu, nu : ndarray
        The ``L`` random directions, shapes ``(L, q)`` and ``(L, p)``.
    residuals : ndarray, shape (n_eff,)
        ``Y_t - mhat(Y_{t-1})`` on the standardised ``Y`` scale.
    m_hat, phi_hat : ndarray
        Conditional-mean fits and conditional characteristic function
        estimates (the latter of shape ``(n_eff, L)``, complex).
    influence : ndarray of complex, shape (n_eff, L)
        The summands ``z_{t,l}`` of ``Shat_n``; the bootstrap reuses these.
    direction : str
        Human-readable ``"X -> Y"`` label.
    doubly_robust : bool
    settings : dict
        Full record of hyper-parameters, for reproducibility.
    elapsed : float
        Wall-clock seconds.
    """

    ks_stat: float
    pvalue: float
    critical_values: dict
    reject: bool
    alpha: float
    p: int
    q: int
    lag: int
    n: int
    n_eff: int
    boot_stats: np.ndarray
    S_hat: np.ndarray
    mu: np.ndarray
    nu: np.ndarray
    residuals: np.ndarray
    m_hat: np.ndarray
    phi_hat: np.ndarray
    influence: np.ndarray
    direction: str = "X -> Y"
    doubly_robust: bool = True
    settings: dict = field(default_factory=dict)
    elapsed: float = 0.0

    # ------------------------------------------------------------------ #
    @property
    def stars(self) -> str:
        """Conventional significance markers ``***``/``**``/``*``."""
        if self.pvalue < 0.01:
            return "***"
        if self.pvalue < 0.05:
            return "**"
        if self.pvalue < 0.10:
            return "*"
        return ""

    @property
    def decision(self) -> str:
        return "reject H0" if self.reject else "do not reject H0"

    def summary(self) -> str:
        """A publication-ready text block."""
        cv = "  ".join(f"{100 * a:g}%: {v:.4f}" for a, v in sorted(self.critical_values.items()))
        kind = "doubly robust" if self.doubly_robust else "NAIVE (not doubly robust)"
        s = self.settings
        lines = [
            "=" * 72,
            "  Doubly Robust Granger Causality Test (DRGCT)",
            "  Hui, Liu & Song (2025), arXiv:2509.15798 -- Algorithm 1",
            "=" * 72,
            f"  H0 : {self.direction.split('->')[0].strip()} does not Granger-cause "
            f"{self.direction.split('->')[1].strip()} in mean",
            f"  Construction        : {kind}",
            f"  Lag orders          : p = {self.p} (cause), q = {self.q} (effect)",
            f"  Sample              : n = {self.n},  effective n - q = {self.n_eff}",
            f"  MDN components G    : {s.get('G')}",
            f"  (mu, nu) pairs L    : {s.get('L')}   drawn U[{s.get('w_lower')}, {s.get('w_upper')}]",
            f"  Pseudo-samples M    : {s.get('M')}",
            f"  Bootstrap B         : {s.get('B')}   multipliers: {s.get('multiplier')}",
            f"  Network (MLP/MDN)   : width {s.get('mlp_width')}/{s.get('mdn_width')}, "
            f"depth {s.get('mlp_depth')}/{s.get('mdn_depth')}",
            "-" * 72,
            f"  KS_n statistic      : {self.ks_stat:.6f}",
            f"  Bootstrap p-value   : {self.pvalue:.4f} {self.stars}",
            f"  Critical values     : {cv}",
            f"  Decision at {100 * self.alpha:g}%    : {self.decision.upper()}",
            "-" * 72,
            f"  Runtime             : {self.elapsed:.2f} s",
            "  Signif. codes: *** 1%   ** 5%   * 10%",
            "=" * 72,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<DRGCTResult {self.direction} lag={self.lag} "
            f"KS={self.ks_stat:.4f} p={self.pvalue:.4f}{self.stars}>"
        )

    def print(self) -> None:
        """Print :meth:`summary`."""
        print(self.summary())

    def to_dict(self) -> dict:
        """Flat record suitable for a tidy results table."""
        return {
            "direction": self.direction,
            "lag": self.lag,
            "p": self.p,
            "q": self.q,
            "n": self.n,
            "n_eff": self.n_eff,
            "ks_stat": self.ks_stat,
            "pvalue": self.pvalue,
            "reject": self.reject,
            "stars": self.stars,
            "doubly_robust": self.doubly_robust,
            **{f"cv_{int(100 * a)}": v for a, v in sorted(self.critical_values.items())},
            "elapsed": self.elapsed,
        }

    def to_frame(self):
        """One-row :class:`pandas.DataFrame` of :meth:`to_dict`."""
        import pandas as pd

        return pd.DataFrame([self.to_dict()])


# --------------------------------------------------------------------------- #
# Core routine
# --------------------------------------------------------------------------- #
def drgc_test(
    x,
    y,
    lag: int | None = None,
    *,
    p: int | None = None,
    q: int | None = None,
    G: int = 10,
    L: int = 20,
    M: int = 20,
    B: int = 1000,
    alpha: float = 0.05,
    w_lower: float = -1.0,
    w_upper: float = 1.0,
    multiplier: str = "rademacher",
    doubly_robust: bool = True,
    standardize: bool = True,
    mlp: MLPConfig | dict | None = None,
    mdn: MDNConfig | dict | None = None,
    seed: int | None = None,
    alphas: Sequence[float] = _DEFAULT_ALPHAS,
    x_name: str = "X",
    y_name: str = "Y",
    return_networks: bool = False,
) -> DRGCTResult:
    r"""Test whether ``x`` Granger-causes ``y`` in mean.

    This is the user-facing entry point implementing Steps 1-5 of Algorithm 1.

    Parameters
    ----------
    x, y : array_like
        Two aligned, *stationary* time series of equal length.  ``x`` plays
        the role of ``X_t`` (candidate cause), ``y`` the role of ``Y_t``.
        Use :func:`drgct.utils.check_stationarity` to pre-screen them.
    lag : int, optional
        Common lag order, i.e. ``p = q = lag`` -- the convention used
        throughout the paper's simulations and application.  Either ``lag``
        or both ``p`` and ``q`` must be given.
    p, q : int, optional
        Separate lag orders for ``X`` and ``Y``.  The paper maintains
        ``p <= q``; larger ``p`` is allowed here and simply shortens the
        effective sample.
    G : int, default 10
        Number of mixture components of the MDN.  Section 4: "In general,
        the setting ``G = 10`` is suitable for most scenarios."
    L : int, default 20
        Number of i.i.d. pairs ``(mu_l, nu_l)``.  Larger ``L`` raises power
        at linear computational cost; the paper fixes ``L = 20``.
    M : int, default 20
        Number of pseudo-samples drawn from the fitted conditional density
        per observation.  The test is insensitive to ``M``.
    B : int, default 1000
        Bootstrap replications.  Cheap: only the multipliers are redrawn.
    alpha : float, default 0.05
        Headline significance level for ``.reject``.
    w_lower, w_upper : float
        Bounds of the multivariate uniform from which ``(mu_l, nu_l)`` are
        drawn, i.e. the compact set ``W = W_1 x W_2``.
    multiplier : {'rademacher', 'mammen', 'normal'}
        Distribution of the bootstrap multipliers ``xi_t``.  The first two
        have bounded support as Theorem 4 requires.
    doubly_robust : bool, default True
        ``False`` builds the naive process (5) instead of (8).  Provided so
        that the size distortion documented in Section 4 can be reproduced;
        never use it for inference.
    standardize : bool, default True
        z-score ``X`` and ``Y`` before training.  The bootstrap p-value is
        invariant to this, but the networks train far more reliably.
    mlp, mdn : MLPConfig / MDNConfig or dict, optional
        Network hyper-parameters.  A dict is splatted into the config class.
    seed : int, optional
        Master seed for full reproducibility.
    alphas : sequence of float
        Levels at which bootstrap critical values are reported.
    x_name, y_name : str
        Labels used in the printed summary.
    return_networks : bool
        Attach the fitted ``MLP`` / ``MixtureDensityNetwork`` objects to
        ``result.settings['networks']`` (useful for the MDN diagnostic plot).

    Returns
    -------
    DRGCTResult

    Examples
    --------
    >>> import numpy as np
    >>> from drgct import drgc_test
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> x = np.zeros(n); e = rng.normal(0, 0.5, n)
    >>> for t in range(1, n):
    ...     x[t] = -0.5 * x[t - 1] + e[t]
    >>> y = 0.5 * np.roll(x, 1) ** 2 + rng.normal(0, 0.5, n)   # X causes Y
    >>> res = drgc_test(x, y, lag=1, B=199, seed=1)
    >>> res.pvalue < 0.10
    True

    Notes
    -----
    Runtime is dominated by training the two networks; the bootstrap adds
    milliseconds because it re-uses the stored influence terms ``z_{t,l}``.
    """
    t0 = time.perf_counter()

    if lag is None and (p is None or q is None):
        raise ValueError("Provide either lag=... or both p=... and q=... .")
    if lag is not None:
        p = q = int(lag)
    p, q = int(p), int(q)
    lag = max(p, q)

    if isinstance(mlp, dict):
        mlp = MLPConfig(**mlp)
    if isinstance(mdn, dict):
        mdn = MDNConfig(**mdn)
    mlp_cfg = mlp or MLPConfig()
    mdn_cfg = mdn or MDNConfig()
    mdn_cfg.n_components = int(G)

    rng = set_seed(seed)
    design: LagDesign = build_lag_design(x, y, p=p, q=q)

    # ---- scaling ---------------------------------------------------------- #
    if standardize:
        ylag_s, _, _ = zscore(design.ylag)
        xlag_s, _, _ = zscore(design.xlag)
        y_mean, y_scale = design.y.mean(), design.y.std()
        y_scale = y_scale if y_scale > 1e-12 else 1.0
        y_s = (design.y - y_mean) / y_scale
    else:
        ylag_s, xlag_s, y_s = design.ylag, design.xlag, design.y

    # ---- Step 1: conditional mean via MLP --------------------------------- #
    mean_fit = fit_conditional_mean(ylag_s, y_s, mlp_cfg, lag=lag)
    resid = y_s - mean_fit.fitted  # Y_t - mhat(Y_{t-1})

    # ---- Step 2(d): random directions (mu_l, nu_l) ------------------------ #
    mu = rng.uniform(w_lower, w_upper, size=(int(L), q))
    nu = rng.uniform(w_lower, w_upper, size=(int(L), p))

    # ---- Steps 2(b), 2(c), 2(e): conditional characteristic function ------ #
    if doubly_robust:
        dens_fit = fit_conditional_density(
            ylag_s,
            xlag_s,
            mdn_cfg,
            n_samples=int(M),
            lag=lag,
            seed=None if seed is None else int(seed) + 7919,
        )
        # samples: (N, M, p);  proj: (N, M, L)
        proj = np.einsum("nmp,lp->nml", dens_fit.samples, nu)
        phi_hat = np.exp(1j * proj).mean(axis=1)  # (N, L)
    else:
        dens_fit = None
        phi_hat = np.zeros((design.n_eff, int(L)), dtype=complex)

    # ---- Step 3: the empirical process and KS_n --------------------------- #
    a = np.exp(1j * (ylag_s @ mu.T))  # e^{i mu' Y_{t-1}}   (N, L)
    b = np.exp(1j * (xlag_s @ nu.T)) - phi_hat  # (N, L)
    z = resid[:, None] * a * b  # influence terms z_{t,l}
    root_n = np.sqrt(design.n_eff)
    S_hat = z.sum(axis=0) / root_n
    ks_stat = float(np.max(np.maximum(np.abs(S_hat.real), np.abs(S_hat.imag))))

    # ---- Step 4: multiplier bootstrap ------------------------------------- #
    B = int(B)
    xi = draw_multipliers(rng, (B, design.n_eff), multiplier)
    S_boot = (xi @ z) / root_n  # (B, L) complex
    boot_stats = np.max(np.maximum(np.abs(S_boot.real), np.abs(S_boot.imag)), axis=1)
    pvalue = float(np.mean(boot_stats >= ks_stat))
    crit = {float(a_): float(np.quantile(boot_stats, 1.0 - a_)) for a_ in alphas}

    settings = {
        "G": int(G),
        "L": int(L),
        "M": int(M),
        "B": B,
        "w_lower": w_lower,
        "w_upper": w_upper,
        "multiplier": multiplier,
        "standardize": standardize,
        "seed": seed,
        "mlp_width": mean_fit.width,
        "mlp_depth": mean_fit.depth,
        "mlp_loss": mlp_cfg.loss,
        "mdn_width": None if dens_fit is None else dens_fit.width,
        "mdn_depth": None if dens_fit is None else dens_fit.depth,
        "mlp_history": mean_fit.history,
        "mdn_history": None if dens_fit is None else dens_fit.history,
    }
    if return_networks:
        settings["networks"] = {
            "mlp": mean_fit.model,
            "mdn": None if dens_fit is None else dens_fit.model,
            "mdn_samples": None if dens_fit is None else dens_fit.samples,
            "ylag_std": ylag_s,
            "xlag_std": xlag_s,
        }

    return DRGCTResult(
        ks_stat=ks_stat,
        pvalue=pvalue,
        critical_values=crit,
        reject=bool(pvalue < alpha),
        alpha=float(alpha),
        p=p,
        q=q,
        lag=lag,
        n=design.n,
        n_eff=design.n_eff,
        boot_stats=boot_stats,
        S_hat=S_hat,
        mu=mu,
        nu=nu,
        residuals=resid,
        m_hat=mean_fit.fitted,
        phi_hat=phi_hat,
        influence=z,
        direction=f"{x_name} -> {y_name}",
        doubly_robust=bool(doubly_robust),
        settings=settings,
        elapsed=time.perf_counter() - t0,
    )


# --------------------------------------------------------------------------- #
# Convenience wrappers
# --------------------------------------------------------------------------- #
def drgc_lag_scan(
    x,
    y,
    lags: Iterable[int] = range(1, 11),
    *,
    progress: bool = True,
    seed: int | None = None,
    **kwargs,
):
    """Run :func:`drgc_test` over a grid of lag orders and tabulate the result.

    This reproduces the structure of Table 6 of the paper, where causality is
    reported "under specific lag orders" 1 to 10.

    Parameters
    ----------
    x, y : array_like
    lags : iterable of int, default ``range(1, 11)``
    progress : bool
        Print a one-line progress report per lag.
    seed : int, optional
        Base seed; lag ``k`` uses ``seed + k`` so that the runs are
        independent yet reproducible.
    **kwargs
        Forwarded to :func:`drgc_test`.

    Returns
    -------
    (pandas.DataFrame, dict)
        A tidy table with one row per lag, and a dict mapping lag ->
        :class:`DRGCTResult`.
    """
    import pandas as pd

    rows, results = [], {}
    for k in lags:
        res = drgc_test(x, y, lag=int(k), seed=None if seed is None else int(seed) + int(k), **kwargs)
        results[int(k)] = res
        rows.append(res.to_dict())
        if progress:
            print(
                f"  lag {k:>2d} | KS = {res.ks_stat:8.4f} | p = {res.pvalue:6.4f} "
                f"{res.stars:<3} | {res.elapsed:5.1f}s"
            )
    return pd.DataFrame(rows), results


def drgc_stability(
    x,
    y,
    lag: int,
    *,
    n_draws: int = 25,
    seed: int = 0,
    progress: bool = False,
    **kwargs,
):
    r"""Re-run the test over ``n_draws`` independent random-direction draws.

    **Why you should always do this before reporting a result.**
    Step 2(d) of Algorithm 1 draws ``L`` pairs ``(mu_l, nu_l)`` at random from
    a compact subset of ``R^{p+q}``.  With ``L = 20`` and a conditioning set of
    dimension ``p + q = 20``, those directions cover ``W`` thinly, so ``KS_n``
    -- and therefore the p-value -- carries a non-negligible amount of
    *simulation* noise on top of the sampling noise.  Two honest analysts using
    different seeds can land on either side of 0.05.  The asymptotic theory is
    unaffected (Assumption 7(iv) lets ``L`` grow polynomially in ``n``), but
    finite-sample practice is: report the spread, not one lucky draw.

    Three remedies, in order of preference: raise ``L``; report this
    distribution; use the merged p-value returned here.

    Parameters
    ----------
    x, y : array_like
    lag : int
    n_draws : int, default 25
        Number of independent (direction draw, network initialisation) pairs.
    seed : int
        Base seed; draw ``r`` uses ``seed + 7919 * r``.
    progress : bool
    **kwargs
        Forwarded to :func:`drgc_test`.

    Returns
    -------
    dict
        ``pvalues`` (ndarray), ``ks_stats`` (ndarray), ``median``, ``mean``,
        ``q05``, ``q95``, ``share_reject`` (fraction below ``alpha``),
        ``merged_pvalue`` and ``results`` (the list of
        :class:`DRGCTResult` objects).

        ``merged_pvalue`` is ``min(1, 2 * median(p_r))``.  Ruger's inequality
        -- see also Vovk and Wang (2020) on p-value merging -- guarantees that
        twice the median of arbitrarily dependent valid p-values is itself a
        valid p-value.  It is *conservative* by construction: rejecting on it
        is a strictly stronger statement than rejecting on any single draw.

    Examples
    --------
    >>> import numpy as np
    >>> from drgct import drgc_stability
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> x = rng.normal(size=n)
    >>> y = np.r_[0.0, 1.5 * np.sin(2 * x[:-1])] + rng.normal(0, 0.4, n)
    >>> out = drgc_stability(x, y, lag=1, n_draws=5, B=199, L=10, M=10, G=4)
    >>> out["merged_pvalue"] < 0.05
    True
    """
    alpha = float(kwargs.get("alpha", 0.05))
    results, pvals, stats = [], [], []
    for r in range(int(n_draws)):
        res = drgc_test(x, y, lag=lag, seed=int(seed) + 7919 * r, **kwargs)
        results.append(res)
        pvals.append(res.pvalue)
        stats.append(res.ks_stat)
        if progress:
            print(f"  draw {r + 1:>3d}/{n_draws}: KS = {res.ks_stat:7.4f}  p = {res.pvalue:.4f}")
    pvals = np.asarray(pvals, dtype=float)
    stats = np.asarray(stats, dtype=float)
    med = float(np.median(pvals))
    return {
        "pvalues": pvals,
        "ks_stats": stats,
        "median": med,
        "mean": float(pvals.mean()),
        "q05": float(np.quantile(pvals, 0.05)),
        "q95": float(np.quantile(pvals, 0.95)),
        "share_reject": float(np.mean(pvals < alpha)),
        "merged_pvalue": float(min(1.0, 2.0 * med)),
        "alpha": alpha,
        "lag": int(lag),
        "n_draws": int(n_draws),
        "direction": results[0].direction if results else "",
        "results": results,
    }


def drgc_both_directions(
    x,
    y,
    lag: int,
    *,
    x_name: str = "X",
    y_name: str = "Y",
    seed: int | None = None,
    **kwargs,
):
    """Test ``X -> Y`` and ``Y -> X`` at a common lag order.

    Returns
    -------
    dict
        ``{'x_to_y': DRGCTResult, 'y_to_x': DRGCTResult}``.
    """
    fwd = drgc_test(
        x, y, lag=lag, x_name=x_name, y_name=y_name, seed=seed, **kwargs
    )
    bwd = drgc_test(
        y,
        x,
        lag=lag,
        x_name=y_name,
        y_name=x_name,
        seed=None if seed is None else int(seed) + 104729,
        **kwargs,
    )
    return {"x_to_y": fwd, "y_to_x": bwd}
