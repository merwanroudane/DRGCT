r"""The six data generating processes of Section 4 of Hui, Liu and Song (2025).

Table 1 of the paper (reproduced verbatim below) fixes the dynamics of ``X_t``
to a linear AR(p) in every design and varies how ``{X_{t-1}, ..., X_{t-p}}``
enters ``Y_t``:

============  ====================================  ==========================================================
DGP           ``X_t``                               ``Y_t``
============  ====================================  ==========================================================
**S1** size   ``0.5 sum_k b_k X_{t-k} + e1_t``      ``0.5 sum_j a_j Y_{t-j} + e2_t``
**S2** size   ``0.5 sum_k b_k X_{t-k} + e1_t``      ``a sum_j exp(-0.5 Y_{t-j}^2) + e2_t``
**P1** power  ``0.5 sum_k b_k X_{t-k} + e1_t``      ``0.5 sum_j a_j Y_{t-j} + sin( sum_k c_k X_{t-k} ) + e2_t``
**P2** power  ``0.5 sum_k b_k X_{t-k} + e1_t``      ``0.5 sum_j a_j Y_{t-j} + 0.5 c sum_k X_{t-k}^2 + e2_t``
**P3** power  ``0.5 sum_k b_k X_{t-k} + e1_t``      ``a sum_j exp(-0.5 Y_{t-j}^2) + c sum_k cos(X_{t-k}) + e2_t``
**P4** power  ``0.5 sum_k b_k X_{t-k} + e1_t``      ``a0 sum_j ( X_{t-j} Y_{t-j} ) + e2_t``
============  ====================================  ==========================================================

``e1_t`` and ``e2_t`` are i.i.d. ``N(0, 0.5)`` and mutually independent.
S1 and S2 satisfy the null (no causality); P1-P4 satisfy the alternative.
Throughout, ``lag = p = q``, and the coefficient vectors for
``lag = 1, ..., 5`` are the ones tabulated in Table 2 of the paper -- they are
chosen to keep the multi-lag sums from diverging, not to flatter the test.

The first ``n0 = 500`` observations are discarded to remove the influence of
the initial values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "DGP_NAMES",
    "SIZE_DGPS",
    "POWER_DGPS",
    "PARAMETERS",
    "DGP_EQUATIONS",
    "dgp_parameters",
    "simulate_dgp",
    "parameter_table",
    "dgp_table",
]

DGP_NAMES = ("S1", "S2", "P1", "P2", "P3", "P4")
SIZE_DGPS = ("S1", "S2")
POWER_DGPS = ("P1", "P2", "P3", "P4")

#: LaTeX-ready equations, used by :func:`drgct.tables.table_dgp_definitions`.
DGP_EQUATIONS = {
    "S1": (r"X_t = 0.5\sum_{k=1}^{p} b_k X_{t-k} + \varepsilon_{1,t}",
           r"Y_t = 0.5\sum_{j=1}^{q} a_j Y_{t-j} + \varepsilon_{2,t}"),
    "S2": (r"X_t = 0.5\sum_{k=1}^{p} b_k X_{t-k} + \varepsilon_{1,t}",
           r"Y_t = a\sum_{j=1}^{q} \exp(-0.5 Y_{t-j}^2) + \varepsilon_{2,t}"),
    "P1": (r"X_t = 0.5\sum_{k=1}^{p} b_k X_{t-k} + \varepsilon_{1,t}",
           r"Y_t = 0.5\sum_{j=1}^{q} a_j Y_{t-j} + \sin\!\big(\sum_{k=1}^{p} c_k X_{t-k}\big) + \varepsilon_{2,t}"),
    "P2": (r"X_t = 0.5\sum_{k=1}^{p} b_k X_{t-k} + \varepsilon_{1,t}",
           r"Y_t = 0.5\sum_{j=1}^{q} a_j Y_{t-j} + 0.5c\sum_{k=1}^{p} X_{t-k}^2 + \varepsilon_{2,t}"),
    "P3": (r"X_t = 0.5\sum_{k=1}^{p} b_k X_{t-k} + \varepsilon_{1,t}",
           r"Y_t = a\sum_{j=1}^{q} \exp(-0.5 Y_{t-j}^2) + c\sum_{k=1}^{p}\cos(X_{t-k}) + \varepsilon_{2,t}"),
    "P4": (r"X_t = 0.5\sum_{k=1}^{p} b_k X_{t-k} + \varepsilon_{1,t}",
           r"Y_t = a_0\sum_{j=1}^{q} (X_{t-j} Y_{t-j}) + \varepsilon_{2,t}"),
}

_THIRD = 1.0 / 3.0

#: Table 2 of the paper, keyed by lag order.
PARAMETERS: dict[int, dict] = {
    1: {"a": [1.0], "b": [-1.0], "c": [-1.0], "a_exp": 0.5, "c_scalar": 1.0, "a0": 0.5},
    2: {
        "a": [0.5, -0.5],
        "b": [-0.5, 0.5],
        "c": [-0.5, 0.5],
        "a_exp": 0.25,
        "c_scalar": 0.6,
        "a0": 0.4,
    },
    3: {
        "a": [0.5, -0.5, 0.5],
        "b": [-0.5, 0.5, 0.5],
        "c": [-0.5, 0.5, -0.5],
        "a_exp": 0.25,
        "c_scalar": 0.5,
        "a0": _THIRD,
    },
    4: {
        "a": [0.25, -0.25, 0.25, 0.25],
        "b": [-0.25, 0.25, 0.25, -0.25],
        "c": [-0.25, 0.25, -0.25, 0.25],
        "a_exp": 0.125,
        "c_scalar": 0.5,
        "a0": _THIRD,
    },
    5: {
        "a": [0.25, -0.25, 0.25, 0.25, -0.25],
        "b": [-0.25, 0.25, 0.25, -0.25, 0.25],
        "c": [-0.25, 0.25, -0.25, 0.25, -0.25],
        "a_exp": 0.125,
        "c_scalar": 0.5,
        "a0": _THIRD,
    },
}


def dgp_parameters(lag: int) -> dict:
    """Return the Table 2 coefficient block for ``lag``.

    For ``lag > 5`` -- outside the paper's grid -- the lag-5 pattern is
    continued by alternating ``+/-0.25`` and keeping the scalars fixed, and a
    note is attached under the key ``'extrapolated'``.
    """
    lag = int(lag)
    if lag in PARAMETERS:
        return {k: (list(v) if isinstance(v, list) else v) for k, v in PARAMETERS[lag].items()}
    if lag < 1:
        raise ValueError("lag must be >= 1.")
    sign = np.where(np.arange(lag) % 2 == 0, 1.0, -1.0)
    scale = 1.0 / lag * 1.25
    return {
        "a": list(scale * sign),
        "b": list(-scale * sign),
        "c": list(-scale * sign),
        "a_exp": 0.5 / lag,
        "c_scalar": 0.5,
        "a0": 1.0 / lag,
        "extrapolated": True,
    }


@dataclass
class SimulatedSeries:
    """Container returned by :func:`simulate_dgp`."""

    x: np.ndarray
    y: np.ndarray
    dgp: str
    lag: int
    n: int
    causal: bool
    params: dict

    def as_tuple(self):
        return self.x, self.y


def simulate_dgp(
    dgp: str,
    n: int,
    lag: int,
    *,
    rng: np.random.Generator | int | None = None,
    burn: int = 500,
    sigma2: float = 0.5,
    innovation_scale: str = "variance",
    params: dict | None = None,
) -> SimulatedSeries:
    r"""Generate one realisation of a Table 1 design.

    Parameters
    ----------
    dgp : {'S1','S2','P1','P2','P3','P4'}
        Design name.  ``S*`` satisfy the null, ``P*`` the alternative.
    n : int
        Number of observations *kept* (the paper's ``n``).
    lag : int
        Common lag order ``p = q = lag``.
    rng : Generator, int or None
        Random state.
    burn : int, default 500
        Number of initial observations discarded (the paper's ``n0 = 500``).
    sigma2 : float, default 0.5
        Dispersion of the innovations, ``e1_t, e2_t ~ N(0, sigma2)``.
    innovation_scale : {'variance', 'sd'}, default ``'variance'``
        The paper writes ``N(0, 0.5)``.  Read as a *variance* by default
        (so the standard deviation is ``sqrt(0.5) ~ 0.707``); set to
        ``'sd'`` to read 0.5 as the standard deviation instead.
    params : dict, optional
        Override the Table 2 coefficients.  Keys ``a``, ``b``, ``c`` (lists of
        length ``lag``) and ``a_exp``, ``c_scalar``, ``a0`` (scalars).

    Returns
    -------
    SimulatedSeries
        ``.x`` and ``.y`` are length-``n`` arrays.

    Examples
    --------
    >>> from drgct.dgp import simulate_dgp
    >>> s = simulate_dgp("P2", n=300, lag=2, rng=0)
    >>> s.x.shape, s.y.shape, s.causal
    ((300,), (300,), True)
    """
    dgp = str(dgp).upper()
    if dgp not in DGP_NAMES:
        raise ValueError(f"dgp must be one of {DGP_NAMES}, got {dgp!r}.")
    lag = int(lag)
    n = int(n)
    burn = int(burn)
    if isinstance(rng, (int, np.integer)) or rng is None:
        rng = np.random.default_rng(rng)

    pr = dict(dgp_parameters(lag))
    if params:
        pr.update(params)
    a = np.asarray(pr["a"], dtype=float)
    b = np.asarray(pr["b"], dtype=float)
    c = np.asarray(pr["c"], dtype=float)
    a_exp = float(pr["a_exp"])
    c_scalar = float(pr["c_scalar"])
    a0 = float(pr["a0"])
    for name, vec in (("a", a), ("b", b), ("c", c)):
        if vec.size != lag:
            raise ValueError(f"params['{name}'] must have length lag={lag}, got {vec.size}.")

    sd = np.sqrt(sigma2) if innovation_scale == "variance" else float(sigma2)
    total = n + burn
    e1 = rng.normal(0.0, sd, total)
    e2 = rng.normal(0.0, sd, total)

    x = np.zeros(total)
    y = np.zeros(total)

    for t in range(lag, total):
        xl = x[t - lag : t][::-1]  # (X_{t-1}, ..., X_{t-lag})
        yl = y[t - lag : t][::-1]  # (Y_{t-1}, ..., Y_{t-lag})

        x[t] = 0.5 * float(b @ xl) + e1[t]

        if dgp == "S1":
            y[t] = 0.5 * float(a @ yl) + e2[t]
        elif dgp == "S2":
            y[t] = a_exp * float(np.sum(np.exp(-0.5 * yl**2))) + e2[t]
        elif dgp == "P1":
            y[t] = 0.5 * float(a @ yl) + np.sin(float(c @ xl)) + e2[t]
        elif dgp == "P2":
            y[t] = 0.5 * float(a @ yl) + 0.5 * c_scalar * float(np.sum(xl**2)) + e2[t]
        elif dgp == "P3":
            y[t] = (
                a_exp * float(np.sum(np.exp(-0.5 * yl**2)))
                + c_scalar * float(np.sum(np.cos(xl)))
                + e2[t]
            )
        else:  # P4
            y[t] = a0 * float(np.sum(xl * yl)) + e2[t]

        if not np.isfinite(x[t]) or not np.isfinite(y[t]):
            raise FloatingPointError(
                f"DGP {dgp} diverged at t={t} with lag={lag}. "
                "Reduce the coefficients (see the note in Section 6 of the paper)."
            )

    return SimulatedSeries(
        x=x[burn:],
        y=y[burn:],
        dgp=dgp,
        lag=lag,
        n=n,
        causal=dgp in POWER_DGPS,
        params=pr,
    )


# --------------------------------------------------------------------------- #
# Descriptive tables
# --------------------------------------------------------------------------- #
def dgp_table():
    """Table 1 of the paper as a :class:`pandas.DataFrame` of LaTeX equations."""
    import pandas as pd

    rows = []
    for name in DGP_NAMES:
        eq_x, eq_y = DGP_EQUATIONS[name]
        rows.append(
            {
                "DGP": name,
                "Hypothesis": "H0 (no causality)" if name in SIZE_DGPS else "H1 (causality)",
                "X_t": f"${eq_x}$",
                "Y_t": f"${eq_y}$",
            }
        )
    return pd.DataFrame(rows)


def parameter_table(lags: Sequence[int] = (1, 2, 3, 4, 5)):
    """Table 2 of the paper as a tidy :class:`pandas.DataFrame`."""
    import pandas as pd

    def fmt(v):
        if isinstance(v, list):
            return ", ".join(f"{z:.4g}" for z in v)
        return f"{v:.4g}"

    rows = []
    for lag in lags:
        pr = dgp_parameters(int(lag))
        rows.append(
            {
                "lag": int(lag),
                "a (S1,P1,P2)": fmt(pr["a"]),
                "b (all)": fmt(pr["b"]),
                "c (P1)": fmt(pr["c"]),
                "a (S2,P3)": fmt(pr["a_exp"]),
                "c (P2,P3)": fmt(pr["c_scalar"]),
                "a0 (P4)": fmt(pr["a0"]),
            }
        )
    return pd.DataFrame(rows)
