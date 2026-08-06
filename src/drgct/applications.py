r"""The empirical study of Section 5: price-volume causality, and extensions.

Section 5 of Hui, Liu and Song (2025) asks whether daily *percentage changes*
in an index level (``P_t``) Granger-cause percentage changes in trading volume
(``V_t``) and vice versa, for three markets, across three overlapping
three-year sub-samples, at every lag order from 1 to 10.  That is
3 indices x 3 periods x 2 directions x 10 lags = **180 tests**, which is what
:func:`price_volume_study` runs.

Two extensions that the paper points to but does not carry out are provided
here as well:

* :func:`rolling_causality` -- slide a fixed-length window through calendar
  time and record the p-value at each position, turning the three static
  sub-samples into a continuous picture of when causality switches on;
* :func:`lag_scan_frame` -- the same machinery for a user's own pair of
  series, so the workflow generalises beyond the price-volume application.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, Sequence

import numpy as np

from .datasets import (
    INDEX_KEYS,
    INDEX_LABELS,
    MACRO_PERIODS,
    PAPER_PERIODS,
    load_index,
    load_macro,
    subsample,
    to_percentage_changes,
)

__all__ = [
    "DIRECTIONS",
    "MACRO_RELATIONS",
    "macro_study",
    "run_one_test",
    "price_volume_study",
    "rolling_causality",
    "lag_scan_frame",
]

#: ``P -> V`` (prices cause volumes) and ``V -> P`` (volumes cause prices).
DIRECTIONS = ("P_t -> V_t", "V_t -> P_t")


# --------------------------------------------------------------------------- #
# Worker (module level so that it pickles on Windows)
# --------------------------------------------------------------------------- #
def run_one_test(
    x,
    y,
    lag: int,
    *,
    meta: dict,
    alpha: float = 0.05,
    drgc_kwargs: dict | None = None,
    seed: int | None = None,
    torch_threads: int = 1,
) -> dict:
    """Run a single DRGCT and return a flat record tagged with ``meta``."""
    import torch

    torch.set_num_threads(int(torch_threads))
    from .core import drgc_test

    kw = dict(drgc_kwargs or {})
    res = drgc_test(x, y, lag=int(lag), alpha=alpha, seed=seed, **kw)
    return {
        **meta,
        "lag": int(lag),
        "ks_stat": res.ks_stat,
        "pvalue": res.pvalue,
        "reject": bool(res.pvalue < alpha),
        "stars": res.stars,
        "cv_5": res.critical_values.get(0.05, np.nan),
        "n_eff": res.n_eff,
        "elapsed": res.elapsed,
    }


def _progress(done, total, t0, prefix="  "):
    el = time.perf_counter() - t0
    rate = done / max(el, 1e-9)
    eta = (total - done) / max(rate, 1e-9)
    sys.stdout.write(
        f"\r{prefix}{done:>4d}/{total} tests | {el / 60:5.1f} min elapsed | ETA {eta / 60:5.1f} min   "
    )
    sys.stdout.flush()


# --------------------------------------------------------------------------- #
# Table 5 / Table 6 driver
# --------------------------------------------------------------------------- #
def price_volume_study(
    indices: Sequence[str] = INDEX_KEYS,
    periods: dict | Sequence[str] = tuple(PAPER_PERIODS),
    lags: Iterable[int] = range(1, 11),
    *,
    directions: Sequence[str] = DIRECTIONS,
    alpha: float = 0.05,
    drgc_kwargs: dict | None = None,
    volume_divisor: float = 10.0,
    seed: int = 20240926,
    n_jobs: int = 1,
    progress: bool = True,
    out_csv: str | os.PathLike | None = None,
):
    """Reproduce the Section 5 grid of DRGCTs and return tidy long results.

    Parameters
    ----------
    indices : sequence of str
        Dataset keys, see :data:`drgct.datasets.INDEX_KEYS`.
    periods : dict or sequence of str
        Either keys of :data:`drgct.datasets.PAPER_PERIODS` or an explicit
        ``{label: (start, end)}`` mapping.  Pass ``["full"]`` together with a
        custom mapping to use the whole sample.
    lags : iterable of int, default ``range(1, 11)``
    directions : sequence of str
        Subset of :data:`DIRECTIONS`.
    alpha : float, default 0.05
        "The upper 5% critical value is employed as the threshold."
    drgc_kwargs : dict, optional
        Forwarded to :func:`drgct.drgc_test` (e.g. ``{'G': 10, 'L': 20,
        'M': 20, 'B': 1000}``, which are already the defaults).
    volume_divisor : float, default 10.0
        Section 5 divides the volume percentage change by 10.
    seed : int
    n_jobs : int, default 1
        Worker processes; ``-1`` uses ``cpu_count() - 1``.
    progress : bool
    out_csv : path, optional
        Write the tidy results as they finish.

    Returns
    -------
    pandas.DataFrame
        Columns ``index_key, index_label, period, direction, lag, ks_stat,
        pvalue, reject, stars, cv_5, n_eff, elapsed, start, end``.

    Examples
    --------
    A five-minute version of the paper's study::

        from drgct.applications import price_volume_study
        df = price_volume_study(
            indices=["spx500"], periods=["2021-2024"], lags=range(1, 6),
            drgc_kwargs=dict(B=499), n_jobs=4,
        )
    """
    import pandas as pd

    if isinstance(periods, dict):
        period_map = dict(periods)
    else:
        period_map = {k: PAPER_PERIODS[k] for k in periods}
    lags = [int(k) for k in lags]
    if n_jobs == -1:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)

    # Build the full job list up front so the ETA is honest.
    jobs = []
    for key in indices:
        raw = load_index(key)
        pv_full = to_percentage_changes(raw, volume_divisor=volume_divisor)
        label = INDEX_LABELS.get(key, key)
        for pname, (lo, hi) in period_map.items():
            pv = subsample(pv_full, (lo, hi))
            P = pv["P"].to_numpy()
            V = pv["V"].to_numpy()
            for direction in directions:
                cause, effect = (P, V) if direction == "P_t -> V_t" else (V, P)
                for lag in lags:
                    meta = {
                        "index_key": key,
                        "index_label": label,
                        "period": pname,
                        "start": lo,
                        "end": hi,
                        "direction": direction,
                        "n_obs": int(len(pv)),
                    }
                    sub_seed = (
                        int(seed)
                        + 1009 * (list(indices).index(key) + 1)
                        + 101 * (list(period_map).index(pname) + 1)
                        + 13 * (list(directions).index(direction) + 1)
                        + lag
                    )
                    jobs.append((cause, effect, lag, meta, sub_seed))

    total = len(jobs)
    if progress:
        print(
            f"[drgct] price-volume study: {len(indices)} indices x {len(period_map)} periods "
            f"x {len(directions)} directions x {len(lags)} lags = {total} DRGCTs "
            f"(n_jobs={n_jobs})"
        )

    records: list[dict] = []
    t0 = time.perf_counter()
    kw = dict(alpha=alpha, drgc_kwargs=drgc_kwargs)

    if n_jobs <= 1:
        for i, (c, e, lag, meta, s) in enumerate(jobs, start=1):
            records.append(run_one_test(c, e, lag, meta=meta, seed=s, **kw))
            if progress:
                _progress(i, total, t0)
            if out_csv:
                pd.DataFrame(records).to_csv(out_csv, index=False)
    else:
        with ProcessPoolExecutor(max_workers=int(n_jobs)) as pool:
            futs = [
                pool.submit(run_one_test, c, e, lag, meta=meta, seed=s, **kw)
                for c, e, lag, meta, s in jobs
            ]
            for i, fut in enumerate(as_completed(futs), start=1):
                records.append(fut.result())
                if progress:
                    _progress(i, total, t0)
        if out_csv:
            pd.DataFrame(records).to_csv(out_csv, index=False)

    if progress:
        print(f"\n[drgct] done in {(time.perf_counter() - t0) / 60:.1f} min")

    df = pd.DataFrame(records)
    order_dir = {d: i for i, d in enumerate(directions)}
    order_idx = {INDEX_LABELS.get(k, k): i for i, k in enumerate(indices)}
    df = (
        df.assign(_d=df["direction"].map(order_dir), _i=df["index_label"].map(order_idx))
        .sort_values(["_d", "_i", "period", "lag"])
        .drop(columns=["_d", "_i"])
        .reset_index(drop=True)
    )
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


# --------------------------------------------------------------------------- #
# Rolling window
# --------------------------------------------------------------------------- #
def rolling_causality(
    x,
    y,
    *,
    lag: int = 5,
    window: int = 750,
    step: int = 21,
    dates=None,
    alpha: float = 0.05,
    both_directions: bool = True,
    x_name: str = "P_t",
    y_name: str = "V_t",
    drgc_kwargs: dict | None = None,
    seed: int = 7,
    n_jobs: int = 1,
    progress: bool = True,
):
    """Slide a fixed window through the sample and test causality at each stop.

    The paper's three overlapping sub-samples are a coarse version of this.
    A rolling profile is the natural way to answer "*when* did prices start
    to drive volumes?", which is the substantive question behind the CSI 300
    result in Table 6.

    Parameters
    ----------
    x, y : array_like
        Candidate cause and effect.
    lag : int, default 5
    window : int, default 750
        Observations per window -- the paper's three-year sub-samples are
        about this size.
    step : int, default 21
        Advance between windows (21 trading days ~ one month).
    dates : DatetimeIndex, optional
        Used to label each window; falls back to integer positions.
    alpha : float
    both_directions : bool
        Also test ``y -> x``.
    x_name, y_name : str
    drgc_kwargs : dict, optional
    seed : int
    n_jobs : int, default 1
    progress : bool

    Returns
    -------
    pandas.DataFrame
        Columns ``direction, start_idx, end_idx, start_date, end_date, lag,
        ks_stat, pvalue, reject``.
    """
    import pandas as pd

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if window >= n:
        raise ValueError(f"window ({window}) must be smaller than the sample ({n}).")
    starts = list(range(0, n - window + 1, int(step)))
    if n_jobs == -1:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)

    pairs = [(x, y, f"{x_name} -> {y_name}")]
    if both_directions:
        pairs.append((y, x, f"{y_name} -> {x_name}"))

    jobs = []
    for pi, (a, b, dname) in enumerate(pairs):
        for si, s0 in enumerate(starts):
            s1 = s0 + window
            meta = {
                "direction": dname,
                "start_idx": s0,
                "end_idx": s1 - 1,
                "start_date": None if dates is None else dates[s0],
                "end_date": None if dates is None else dates[s1 - 1],
            }
            jobs.append((a[s0:s1], b[s0:s1], lag, meta, seed + 1000 * pi + si))

    total = len(jobs)
    if progress:
        print(f"[drgct] rolling study: {total} windows of {window} obs, step {step}, lag {lag}")

    records = []
    t0 = time.perf_counter()
    kw = dict(alpha=alpha, drgc_kwargs=drgc_kwargs)
    if n_jobs <= 1:
        for i, (a, b, lg, meta, s) in enumerate(jobs, start=1):
            records.append(run_one_test(a, b, lg, meta=meta, seed=s, **kw))
            if progress:
                _progress(i, total, t0)
    else:
        with ProcessPoolExecutor(max_workers=int(n_jobs)) as pool:
            futs = [pool.submit(run_one_test, a, b, lg, meta=meta, seed=s, **kw)
                    for a, b, lg, meta, s in jobs]
            for i, fut in enumerate(as_completed(futs), start=1):
                records.append(fut.result())
                if progress:
                    _progress(i, total, t0)
    if progress:
        print(f"\n[drgct] done in {(time.perf_counter() - t0) / 60:.1f} min")

    return pd.DataFrame(records).sort_values(["direction", "start_idx"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Generic helper
# --------------------------------------------------------------------------- #
def lag_scan_frame(
    x,
    y,
    lags: Iterable[int] = range(1, 11),
    *,
    x_name: str = "X",
    y_name: str = "Y",
    both_directions: bool = True,
    alpha: float = 0.05,
    drgc_kwargs: dict | None = None,
    seed: int = 11,
    n_jobs: int = 1,
    progress: bool = True,
):
    """Lag scan for an arbitrary pair of series, in the tidy layout the
    table and plot builders expect (``index_label``/``period`` set to
    placeholders so :func:`drgct.tables.table_lag_orders` and
    :func:`drgct.plots.plot_pvalue_heatmap` work unchanged).
    """
    import pandas as pd

    pairs = [(x, y, f"{x_name} -> {y_name}")]
    if both_directions:
        pairs.append((y, x, f"{y_name} -> {x_name}"))
    if n_jobs == -1:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)

    jobs = []
    for pi, (a, b, dname) in enumerate(pairs):
        for lag in lags:
            meta = {"index_key": "user", "index_label": f"{x_name} / {y_name}",
                    "period": "full sample", "direction": dname}
            jobs.append((a, b, int(lag), meta, seed + 977 * pi + int(lag)))

    records = []
    t0 = time.perf_counter()
    kw = dict(alpha=alpha, drgc_kwargs=drgc_kwargs)
    if n_jobs <= 1:
        for i, (a, b, lg, meta, s) in enumerate(jobs, start=1):
            records.append(run_one_test(a, b, lg, meta=meta, seed=s, **kw))
            if progress:
                _progress(i, len(jobs), t0)
    else:
        with ProcessPoolExecutor(max_workers=int(n_jobs)) as pool:
            futs = [pool.submit(run_one_test, a, b, lg, meta=meta, seed=s, **kw)
                    for a, b, lg, meta, s in jobs]
            for i, fut in enumerate(as_completed(futs), start=1):
                records.append(fut.result())
                if progress:
                    _progress(i, len(jobs), t0)
    if progress:
        print()
    return pd.DataFrame(records).sort_values(["direction", "lag"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# US macroeconomic application
# --------------------------------------------------------------------------- #
#: The six macroeconomic relations tested in the macro application.  Each entry
#: is ``(cause, effect, short label, the question it answers)`` where *cause*
#: and *effect* are column labels of :func:`drgct.datasets.load_macro`.
MACRO_RELATIONS = [
    ("Fed funds rate", "Industrial production", "MP to output",
     "Does monetary policy move real activity, and after how many months? "
     "Friedman's (1961) long and variable lags."),
    ("Fed funds rate", "CPI inflation", "MP to prices",
     "The price leg of monetary transmission; the lag is conventionally "
     "thought to be longer than the output leg."),
    ("M2 money growth", "CPI inflation", "Money to prices",
     "The quantity theory in its Granger-causal form."),
    ("WTI oil price", "CPI inflation", "Oil to prices",
     "Pass-through of energy supply shocks into headline inflation."),
    ("WTI oil price", "Industrial production", "Oil to output",
     "Hamilton's (1983) oil-shock channel; the response is famously "
     "asymmetric, so a linear test is the wrong instrument."),
    ("Industrial production", "Unemployment rate", "Output to unemployment",
     "Okun's law read as a dynamic, one-directional forecasting statement."),
]


def macro_study(
    relations=MACRO_RELATIONS,
    periods=("Full sample",),
    lags: Iterable[int] = (1, 3, 6, 9, 12, 18),
    *,
    both_directions: bool = True,
    alpha: float = 0.05,
    drgc_kwargs: dict | None = None,
    seed: int = 19590201,
    n_jobs: int = 1,
    progress: bool = True,
    out_csv: str | os.PathLike | None = None,
):
    """Run the DRGCT over a grid of macroeconomic relations, periods and lags.

    Parameters
    ----------
    relations : sequence
        ``(cause, effect, short_label, question)`` tuples; see
        :data:`MACRO_RELATIONS`.  ``cause`` and ``effect`` must be column
        labels of :func:`drgct.datasets.load_macro`.
    periods : sequence of str or dict
        Keys of :data:`drgct.datasets.MACRO_PERIODS`, or an explicit
        ``{label: (start, end)}`` mapping.
    lags : iterable of int
        Lag orders in months.  Monetary transmission is usually placed at
        6-18 months, which is precisely the range kernel-smoothing causality
        tests cannot reach.
    both_directions : bool
        Also test effect -> cause, so that a one-directional finding can be
        distinguished from a feedback loop.
    alpha, drgc_kwargs, seed, n_jobs, progress, out_csv
        As in :func:`price_volume_study`.

    Returns
    -------
    pandas.DataFrame
        Long format with ``index_label`` (the relation), ``period``,
        ``direction``, ``lag``, ``ks_stat``, ``pvalue``, ``reject``, so the
        table and figure builders accept it unchanged.
    """
    import pandas as pd

    period_map = dict(periods) if isinstance(periods, dict) else {
        k: MACRO_PERIODS[k] for k in periods
    }
    lags = [int(k) for k in lags]
    if n_jobs == -1:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)

    macro = load_macro()
    jobs = []
    for ri, (cause, effect, short, _question) in enumerate(relations):
        for c in (cause, effect):
            if c not in macro.columns:
                raise ValueError(f"{c!r} not in the macro dataset: {list(macro.columns)}")
        for pi, (pname, (lo, hi)) in enumerate(period_map.items()):
            sub = macro[(macro.index >= pd.Timestamp(lo)) & (macro.index <= pd.Timestamp(hi))]
            a, b = sub[cause].to_numpy(), sub[effect].to_numpy()
            pairs = [(a, b, f"{cause} -> {effect}")]
            if both_directions:
                pairs.append((b, a, f"{effect} -> {cause}"))
            for di, (u, v, dname) in enumerate(pairs):
                for lag in lags:
                    meta = {
                        "index_label": short,
                        "relation": f"{cause} -> {effect}",
                        "cause": cause,
                        "effect": effect,
                        "period": pname,
                        "start": lo,
                        "end": hi,
                        "direction": dname,
                        "forward": di == 0,
                        "n_obs": int(len(sub)),
                    }
                    jobs.append((u, v, lag, meta,
                                 seed + 7919 * ri + 613 * pi + 97 * di + lag))

    total = len(jobs)
    if progress:
        print(f"[drgct] macro study: {len(relations)} relations x {len(period_map)} periods "
              f"x {1 + int(both_directions)} directions x {len(lags)} lags = {total} DRGCTs "
              f"(n_jobs={n_jobs})")

    records: list[dict] = []
    t0 = time.perf_counter()
    kw = dict(alpha=alpha, drgc_kwargs=drgc_kwargs)
    if n_jobs <= 1:
        for i, (u, v, lag, meta, sd) in enumerate(jobs, start=1):
            records.append(run_one_test(u, v, lag, meta=meta, seed=sd, **kw))
            if progress:
                _progress(i, total, t0)
    else:
        with ProcessPoolExecutor(max_workers=int(n_jobs)) as pool:
            futs = [pool.submit(run_one_test, u, v, lag, meta=meta, seed=sd, **kw)
                    for u, v, lag, meta, sd in jobs]
            for i, fut in enumerate(as_completed(futs), start=1):
                records.append(fut.result())
                if progress:
                    _progress(i, total, t0)
    if progress:
        print(f"\n[drgct] done in {(time.perf_counter() - t0) / 60:.1f} min")

    order = {r[2]: i for i, r in enumerate(relations)}
    df = pd.DataFrame(records)
    df = (df.assign(_o=df["index_label"].map(order))
            .sort_values(["_o", "period", "forward", "lag"], ascending=[True, True, False, True])
            .drop(columns="_o").reset_index(drop=True))
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df
