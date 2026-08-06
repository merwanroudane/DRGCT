#!/usr/bin/env python
"""Example 4 -- A template for your own two series.

Edit the CONFIG block at the top, run the file, and you get the whole
apparatus: stationarity screen, descriptive statistics, a two-directional lag
scan, a stability check, benchmarks, and the full set of journal-ready tables
and figures.

The example ships with a synthetic macro-style dataset (a policy-rate shock
that affects output growth with a 6-month delay through a nonlinear channel)
so that it runs out of the box. Point ``CSV_PATH`` at your own file to use
real data.

Run:
    python examples/04_your_own_data.py
"""

from __future__ import annotations

import pathlib
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

from drgct import drgc_lag_scan, drgc_stability, drgc_test, nhkj_test
from drgct.plots import (
    plot_bootstrap_distribution,
    plot_empirical_process,
    plot_lag_profile,
    plot_pvalue_heatmap,
    plot_stability,
    save_figure,
    use_journal_style,
)
from drgct.tables import (
    export_table,
    table_descriptives,
    table_hyperparameters,
    table_lag_orders,
    table_pvalues,
)
from drgct.utils import check_stationarity

# ======================================================================= #
# CONFIG -- edit this block only
# ======================================================================= #
CSV_PATH = None                # e.g. "my_data.csv"; None uses the synthetic demo
X_COL, Y_COL = "policy", "output"      # column names: X is the candidate cause
X_NAME, Y_NAME = "Policy rate", "Output growth"
TRANSFORM = None               # e.g. "diff", "logdiff", "pct"; None = use as is
LAGS = range(1, 13)            # monthly data -> scan a year of lags
SETTINGS = dict(G=10, L=40, M=20, B=999)   # L raised because p + q reaches 24
SEED = 12345
ALPHA = 0.05
OUT = pathlib.Path("results_example4")
# ======================================================================= #


def synthetic_macro(n=420, seed=3):
    """Policy shocks feed into output growth after 6 months, nonlinearly."""
    rng = np.random.default_rng(seed)
    pol = np.zeros(n)
    out = np.zeros(n)
    e1 = rng.normal(0, 0.4, n)
    e2 = rng.normal(0, 0.4, n)
    for t in range(6, n):
        pol[t] = 0.8 * pol[t - 1] - 0.2 * pol[t - 2] + e1[t]
        # A contraction hurts output more than an easing helps it: an
        # asymmetric, purely nonlinear transmission at lag 6.
        shock = pol[t - 6]
        out[t] = 0.5 * out[t - 1] - 0.6 * np.maximum(shock, 0.0) ** 2 + e2[t]
    dates = pd.date_range("1990-01-31", periods=n, freq="ME")
    return pd.DataFrame({"policy": pol, "output": out}, index=dates)


def transform(s, how):
    s = np.asarray(s, dtype=float)
    if how is None:
        return s
    if how == "diff":
        return np.diff(s)
    if how == "logdiff":
        return 100.0 * np.diff(np.log(s))
    if how == "pct":
        return 100.0 * (s[1:] / s[:-1] - 1.0)
    raise ValueError(f"unknown transform {how!r}")


def linear_granger(cause, effect, lag):
    n = len(effect)
    Yl = np.column_stack([effect[lag - j - 1: n - j - 1] for j in range(lag)])
    Xl = np.column_stack([cause[lag - j - 1: n - j - 1] for j in range(lag)])
    yy = effect[lag:]
    r = sm.OLS(yy, sm.add_constant(Yl)).fit()
    u = sm.OLS(yy, sm.add_constant(np.hstack([Yl, Xl]))).fit()
    return u.compare_f_test(r)[1]


def main() -> None:
    use_journal_style()
    tdir, fdir = OUT / "tables", OUT / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load and align.  Inner join first, transform second.
    # ------------------------------------------------------------------ #
    if CSV_PATH is None:
        panel = synthetic_macro()
        print("Using the built-in synthetic macro dataset "
              "(set CSV_PATH to use your own).")
    else:
        panel = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    panel = panel[[X_COL, Y_COL]].dropna().sort_index()

    x = transform(panel[X_COL].to_numpy(), TRANSFORM)
    y = transform(panel[Y_COL].to_numpy(), TRANSFORM)
    idx = panel.index[len(panel) - len(x):]

    print(f"\n=== Sample ===")
    print(f"  {len(x)} observations, {idx.min().date()} to {idx.max().date()}")
    print(f"  X = {X_NAME} ({X_COL}),  Y = {Y_NAME} ({Y_COL}),  transform = {TRANSFORM}")

    # ------------------------------------------------------------------ #
    # 2. Stationarity screen -- Assumption 1 of the paper
    # ------------------------------------------------------------------ #
    print("\n=== Stationarity ===")
    ok = True
    for name, s in ((X_NAME, x), (Y_NAME, y)):
        chk = check_stationarity(s, name)
        print("  " + chk["message"])
        ok &= chk["stationary"]
    if not ok:
        print("\n  WARNING: at least one series failed the screen.  The asymptotics of\n"
              "  Hui, Liu and Song (2025) assume stationarity -- consider differencing\n"
              "  before you interpret anything below.\n")

    desc = table_descriptives({X_NAME: x, Y_NAME: y})
    print("\n" + desc.to_markdown(index=False))
    export_table(desc, "table0_descriptives", tdir,
                 caption="Descriptive statistics", label="tab:descriptives")

    # ------------------------------------------------------------------ #
    # 3. Two-directional lag scan
    # ------------------------------------------------------------------ #
    print(f"\n=== DRGCT lag scan, {X_NAME} -> {Y_NAME} ===")
    scan_xy, _ = drgc_lag_scan(x, y, lags=LAGS, seed=SEED, alpha=ALPHA,
                               x_name=X_NAME, y_name=Y_NAME, **SETTINGS)
    print(f"\n=== DRGCT lag scan, {Y_NAME} -> {X_NAME} ===")
    scan_yx, _ = drgc_lag_scan(y, x, lags=LAGS, seed=SEED + 1, alpha=ALPHA,
                               x_name=Y_NAME, y_name=X_NAME, **SETTINGS)

    long = pd.concat([
        scan_xy.assign(index_label=f"{X_NAME} / {Y_NAME}", period="full sample",
                       direction=f"{X_NAME} -> {Y_NAME}"),
        scan_yx.assign(index_label=f"{X_NAME} / {Y_NAME}", period="full sample",
                       direction=f"{Y_NAME} -> {X_NAME}"),
    ], ignore_index=True)
    long.to_csv(tdir / "raw_results.csv", index=False)

    print("\n=== Decisions by lag order ===")
    t6 = table_lag_orders(long, alpha=ALPHA, lags=LAGS)
    print(t6.to_markdown(index=False))
    export_table(t6, "table6_lag_orders", tdir,
                 caption="Granger causality under specific lag orders",
                 label="tab:lagorders")
    export_table(table_pvalues(long, lags=LAGS), "table6b_pvalues", tdir,
                 caption="Bootstrap p-values by lag order", label="tab:pvalues")

    # ------------------------------------------------------------------ #
    # 4. Benchmarks -- always report at least the linear one
    # ------------------------------------------------------------------ #
    print(f"\n=== Benchmarks, {X_NAME} -> {Y_NAME} ===")
    print(f"  {'lag':>3s}  {'DRGC':>8s}  {'NHKJ':>8s}  {'linear VAR':>10s}")
    rows = []
    for lag in sorted(set(list(LAGS)[:: max(1, len(list(LAGS)) // 5)]) | {max(LAGS)}):
        d = float(scan_xy.loc[scan_xy["lag"] == lag, "pvalue"].iloc[0])
        k = nhkj_test(x, y, lag=lag).pvalue
        f = linear_granger(x, y, lag)
        rows.append({"lag": lag, "DRGC": d, "NHKJ": k, "linear VAR": f})
        print(f"  {lag:>3d}  {d:>8.4f}  {k:>8.4f}  {f:>10.4f}")
    export_table(pd.DataFrame(rows).round(4), "table_benchmarks", tdir,
                 caption="DRGCT against smoothing-based and linear benchmarks",
                 label="tab:bench")

    # ------------------------------------------------------------------ #
    # 5. Stability -- never report a borderline p-value without this
    # ------------------------------------------------------------------ #
    head_lag = int(scan_xy.sort_values("pvalue")["lag"].iloc[0])
    print(f"\n=== Stability at the most significant lag ({head_lag}) ===")
    stab = drgc_stability(x, y, lag=head_lag, n_draws=25, seed=SEED, alpha=ALPHA,
                          x_name=X_NAME, y_name=Y_NAME, **SETTINGS)
    print(f"  median p        {stab['median']:.4f}")
    print(f"  5th-95th pct    [{stab['q05']:.4f}, {stab['q95']:.4f}]")
    print(f"  share rejecting {stab['share_reject']:.1%}")
    print(f"  merged p-value  {stab['merged_pvalue']:.4f}")
    save_figure(plot_stability(stab, label=f"{X_NAME} $\\to$ {Y_NAME}"),
                "fig_stability", fdir)

    # ------------------------------------------------------------------ #
    # 6. Figures and the hyper-parameter record
    # ------------------------------------------------------------------ #
    print("\n=== Figures ===")
    save_figure(plot_pvalue_heatmap(long, alpha=ALPHA), "fig_pvalue_heatmap", fdir)
    save_figure(plot_lag_profile(scan_xy, alpha=ALPHA,
                                 label=f"{X_NAME} $\\to$ {Y_NAME}"),
                "fig_lagprofile_x2y", fdir)
    save_figure(plot_lag_profile(scan_yx, alpha=ALPHA,
                                 label=f"{Y_NAME} $\\to$ {X_NAME}"),
                "fig_lagprofile_y2x", fdir)

    diag = drgc_test(x, y, lag=head_lag, seed=SEED, alpha=ALPHA,
                     x_name=X_NAME, y_name=Y_NAME, return_networks=True, **SETTINGS)
    diag.print()
    (tdir / "headline_summary.txt").write_text(diag.summary(), encoding="utf-8")
    export_table(table_hyperparameters(diag), "table7_hyperparameters", tdir,
                 caption="Hyper-parameters of the reported test", label="tab:hyper")
    save_figure(plot_bootstrap_distribution(diag), "fig_bootstrap_null", fdir)
    save_figure(plot_empirical_process(diag), "fig_empirical_process", fdir)

    print(f"""
=== What the synthetic demo should show ===

The transmission is at lag 6 and is purely nonlinear (a squared positive part),
so:
  * the linear VAR F-test should struggle at every lag;
  * the DRGCT should be insignificant at lags 1-5 and reject from lag 6 onward;
  * the reverse direction (output -> policy) should never reject.

That is the whole selling point: a delayed, asymmetric channel that a lag-1
test and a linear test both miss.

Output written to {OUT.resolve()}
""")


if __name__ == "__main__":
    main()
