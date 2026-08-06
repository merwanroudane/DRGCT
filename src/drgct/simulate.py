r"""Monte-Carlo machinery reproducing Tables 3 and 4 of Hui, Liu and Song (2025).

The paper's headline experiment is 6 designs x 5 lag orders x 3 sample sizes
x 1000 replications x 1000 bootstrap draws.  That is 90,000 fitted network
pairs and is a cluster-scale job; :func:`monte_carlo` is written so that the
same code runs at any scale you choose, from a 20-replication smoke test to
the full grid, with process-level parallelism.

Two rejection-frequency summaries are provided:

* :func:`summarize` -- the long-format rejection rates, one row per
  (design, n, lag, method);
* :func:`size_power_tables` -- the same numbers reshaped into the exact
  layout of Tables 3 and 4.
"""

from __future__ import annotations

import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, Sequence

import numpy as np

from .dgp import POWER_DGPS, SIZE_DGPS, simulate_dgp

__all__ = [
    "run_replication",
    "monte_carlo",
    "summarize",
    "size_power_tables",
    "PAPER_BANDWIDTH_CONST",
]

#: Section 4 bandwidth constants ``c`` in ``h = c n^{-0.15}`` for the NHKJ
#: benchmark, keyed by ``(design family, lag)``.  ``'linear'`` covers S1/P1/P2
#: (linear autoregressive mean), ``'exp'`` covers S2/P3 (exponential mean).
PAPER_BANDWIDTH_CONST = {
    ("linear", 1): 2.5, ("linear", 2): 3.0, ("linear", 3): 3.0,
    ("linear", 4): 3.5, ("linear", 5): 3.5,
    ("exp", 1): 3.0, ("exp", 2): 3.5, ("exp", 3): 3.5,
    ("exp", 4): 4.0, ("exp", 5): 4.0,
}


def _family(dgp: str) -> str:
    return "exp" if dgp.upper() in ("S2", "P3") else "linear"


def paper_bandwidth_const(dgp: str, lag: int) -> float:
    """The bandwidth constant the paper uses for NHKJ at ``(dgp, lag)``."""
    fam = _family(dgp)
    return PAPER_BANDWIDTH_CONST.get((fam, int(lag)), 4.0 if fam == "exp" else 3.5)


# --------------------------------------------------------------------------- #
# One replication  (module level so that it is picklable on Windows)
# --------------------------------------------------------------------------- #
def run_replication(
    dgp: str,
    n: int,
    lag: int,
    rep: int,
    *,
    seed: int = 0,
    methods: Sequence[str] = ("drgc", "nhkj"),
    alpha: float = 0.05,
    drgc_kwargs: dict | None = None,
    nhkj_kwargs: dict | None = None,
    burn: int = 500,
    torch_threads: int = 1,
) -> list[dict]:
    """Simulate one dataset and apply every requested test to it.

    Parameters
    ----------
    dgp, n, lag : the design point.
    rep : int
        Replication index; combined with ``seed`` into a unique stream.
    seed : int
        Master seed for the whole experiment.
    methods : sequence of {'drgc', 'drgc_naive', 'nhkj'}
        ``'drgc_naive'`` runs the test *without* the doubly robust correction
        (equation (5)), which is how the paper documents the type-I-error
        blow-up of a naive deep-learning plug-in.
    alpha : float
    drgc_kwargs, nhkj_kwargs : dict, optional
        Passed through to :func:`drgct.drgc_test` / :func:`drgct.nhkj_test`.
    burn : int
    torch_threads : int
        ``torch.set_num_threads`` inside the worker; 1 is right when the
        outer loop is already parallelised over processes.

    Returns
    -------
    list of dict
        One record per method.
    """
    import torch

    torch.set_num_threads(int(torch_threads))
    from .core import drgc_test
    from .nhkj import nhkj_test

    drgc_kwargs = dict(drgc_kwargs or {})
    nhkj_kwargs = dict(nhkj_kwargs or {})
    # Silence the per-fit convergence warning: inside a replication loop it
    # would fire thousands of times and drown the progress report.
    from .nets import MDNConfig, MLPConfig

    if not isinstance(drgc_kwargs.get("mlp"), MLPConfig):
        drgc_kwargs["mlp"] = MLPConfig(**(drgc_kwargs.get("mlp") or {}),
                                       **{"warn_convergence": False})
    else:
        drgc_kwargs["mlp"].warn_convergence = False
    if not isinstance(drgc_kwargs.get("mdn"), MDNConfig):
        drgc_kwargs["mdn"] = MDNConfig(**(drgc_kwargs.get("mdn") or {}),
                                       **{"warn_convergence": False})
    else:
        drgc_kwargs["mdn"].warn_convergence = False

    # Deterministic, collision-free stream per design point and replication.
    key = abs(hash((dgp, int(n), int(lag), int(rep), int(seed)))) % (2**31 - 1)
    rng = np.random.default_rng(key)
    sim = simulate_dgp(dgp, n=n, lag=lag, rng=rng, burn=burn)

    out: list[dict] = []
    base = {"dgp": dgp, "n": int(n), "lag": int(lag), "rep": int(rep),
            "causal": dgp.upper() in POWER_DGPS}

    for method in methods:
        t0 = time.perf_counter()
        if method in ("drgc", "drgc_naive"):
            res = drgc_test(
                sim.x,
                sim.y,
                lag=lag,
                alpha=alpha,
                seed=int(key % 2**31),
                doubly_robust=(method == "drgc"),
                **drgc_kwargs,
            )
            rec = {"method": method, "stat": res.ks_stat, "pvalue": res.pvalue}
        elif method == "nhkj":
            kw = dict(nhkj_kwargs)
            kw.setdefault("bandwidth_const", paper_bandwidth_const(dgp, lag))
            res = nhkj_test(sim.x, sim.y, lag=lag, alpha=alpha, **kw)
            rec = {"method": method, "stat": res.stat, "pvalue": res.pvalue}
        else:
            raise ValueError(f"Unknown method {method!r}.")
        rec["reject"] = bool(rec["pvalue"] < alpha)
        rec["elapsed"] = time.perf_counter() - t0
        out.append({**base, **rec})
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def monte_carlo(
    dgps: Iterable[str] = ("S1", "S2", "P1", "P2", "P3", "P4"),
    ns: Iterable[int] = (500, 1000, 2000),
    lags: Iterable[int] = (1, 2, 3, 4, 5),
    reps: int = 1000,
    *,
    methods: Sequence[str] = ("drgc", "nhkj"),
    alpha: float = 0.05,
    seed: int = 20250915,
    n_jobs: int = 1,
    drgc_kwargs: dict | None = None,
    nhkj_kwargs: dict | None = None,
    burn: int = 500,
    progress: bool = True,
    out_csv: str | os.PathLike | None = None,
    flush_every: int = 200,
):
    """Run the full replication grid and return a tidy long DataFrame.

    Parameters
    ----------
    dgps, ns, lags : iterables
        The design grid.  Defaults reproduce Section 4 of the paper.
    reps : int, default 1000
        Monte-Carlo replications per design point.
    methods : sequence of str
        See :func:`run_replication`.
    alpha : float, default 0.05
        Nominal level of the reported rejection frequencies.
    seed : int
    n_jobs : int, default 1
        Worker processes.  ``-1`` uses ``os.cpu_count() - 1``.
    drgc_kwargs, nhkj_kwargs : dict, optional
    burn : int, default 500
    progress : bool
        Print a running ETA.
    out_csv : path, optional
        Stream partial results to disk every ``flush_every`` completed jobs,
        so a long run survives an interruption.
    flush_every : int

    Returns
    -------
    pandas.DataFrame
        Columns ``dgp, n, lag, rep, causal, method, stat, pvalue, reject, elapsed``.

    Notes
    -----
    Cost scales as ``len(dgps) * len(ns) * len(lags) * reps``.  On a laptop,
    one DRGCT at ``n = 500``, ``lag = 3`` takes roughly 2-5 seconds, so the
    paper's full grid is a multi-day single-core job -- use ``n_jobs`` and/or
    trim the grid.  A 100-replication run already pins the empirical size to
    about +/-0.02, which is enough to see the pattern.
    """
    import pandas as pd

    dgps = list(dgps)
    ns = list(ns)
    lags = list(lags)
    grid = [
        (d, n, k, r)
        for d, n, k in itertools.product(dgps, ns, lags)
        for r in range(int(reps))
    ]
    total = len(grid)
    if n_jobs == -1:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)

    if progress:
        print(
            f"[drgct] Monte-Carlo: {len(dgps)} DGPs x {len(ns)} sample sizes x "
            f"{len(lags)} lags x {reps} reps = {total} jobs, methods={list(methods)}, "
            f"n_jobs={n_jobs}"
        )

    records: list[dict] = []
    t0 = time.perf_counter()
    done = 0

    def _report():
        if not progress:
            return
        el = time.perf_counter() - t0
        rate = done / max(el, 1e-9)
        eta = (total - done) / max(rate, 1e-9)
        sys.stdout.write(
            f"\r  {done:>6d}/{total} jobs  |  {el / 60:6.1f} min elapsed  |  "
            f"ETA {eta / 60:6.1f} min   "
        )
        sys.stdout.flush()

    def _maybe_flush():
        if out_csv and done % max(flush_every, 1) == 0:
            pd.DataFrame(records).to_csv(out_csv, index=False)

    kwargs = dict(
        seed=seed,
        methods=tuple(methods),
        alpha=alpha,
        drgc_kwargs=drgc_kwargs,
        nhkj_kwargs=nhkj_kwargs,
        burn=burn,
    )

    if n_jobs <= 1:
        for d, n, k, r in grid:
            records.extend(run_replication(d, n, k, r, **kwargs))
            done += 1
            _report()
            _maybe_flush()
    else:
        with ProcessPoolExecutor(max_workers=int(n_jobs)) as pool:
            futures = {pool.submit(run_replication, d, n, k, r, **kwargs): (d, n, k, r)
                       for d, n, k, r in grid}
            for fut in as_completed(futures):
                records.extend(fut.result())
                done += 1
                _report()
                _maybe_flush()

    if progress:
        print(f"\n[drgct] finished in {(time.perf_counter() - t0) / 60:.1f} min")

    df = pd.DataFrame(records)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def summarize(df, *, alpha: float | None = None):
    """Collapse replication-level output into rejection frequencies.

    Parameters
    ----------
    df : DataFrame
        Output of :func:`monte_carlo`.
    alpha : float, optional
        Recompute rejections at a different level from the stored p-values.

    Returns
    -------
    pandas.DataFrame
        Columns ``dgp, n, lag, method, reps, rejection_rate, mc_se, causal``.
        ``mc_se`` is the Monte-Carlo standard error
        ``sqrt(phat (1 - phat) / reps)`` -- worth reporting next to any
        empirical size.
    """
    import pandas as pd

    d = df.copy()
    if alpha is not None:
        d["reject"] = d["pvalue"] < float(alpha)
    g = (
        d.groupby(["dgp", "n", "lag", "method"], as_index=False)
        .agg(reps=("reject", "size"), rejection_rate=("reject", "mean"),
             mean_pvalue=("pvalue", "mean"), causal=("causal", "first"))
    )
    g["mc_se"] = np.sqrt(g["rejection_rate"] * (1 - g["rejection_rate"]) / g["reps"])
    return g.sort_values(["dgp", "lag", "n", "method"]).reset_index(drop=True)


def size_power_tables(df, *, alpha: float | None = None, methods=("drgc", "nhkj")):
    """Reshape :func:`summarize` output into the layout of Tables 3 and 4.

    Returns
    -------
    dict
        ``{'size': DataFrame, 'power': DataFrame}``.  Each has a
        ``(lag, n)`` MultiIndex on the rows and a ``(DGP, method)``
        MultiIndex on the columns, exactly as printed in the paper.
    """
    import pandas as pd

    s = summarize(df, alpha=alpha)
    s = s[s["method"].isin(methods)]
    label = {"drgc": "DRGC", "drgc_naive": "DRGC (naive)", "nhkj": "NHKJ"}
    s = s.assign(method=s["method"].map(lambda m: label.get(m, m)))

    def _pivot(names):
        sub = s[s["dgp"].isin(names)]
        if sub.empty:
            return pd.DataFrame()
        out = sub.pivot_table(
            index=["lag", "n"], columns=["dgp", "method"], values="rejection_rate"
        )
        order = [d for d in names if d in out.columns.get_level_values(0)]
        return out.reindex(columns=order, level=0).sort_index()

    return {"size": _pivot(list(SIZE_DGPS)), "power": _pivot(list(POWER_DGPS))}
