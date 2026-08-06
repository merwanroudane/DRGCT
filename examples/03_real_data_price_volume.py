#!/usr/bin/env python
"""Example 3 -- The price-volume application on real market data.

A single-index walk-through of Section 5, short enough to read in one sitting:

1. load the CSI 300, transform to percentage changes, screen for stationarity;
2. scan lag orders 1-10 in both directions;
3. compare against a linear VAR F-test and the smoothing-based benchmark;
4. check whether the conclusion survives the random draw of directions;
5. write the tables and figures.

About 4 minutes on 8 cores. ``scripts/run_application.py`` does all three
indices, all three sub-samples, and the rolling-window study.

Run:
    python examples/03_real_data_price_volume.py
"""

from __future__ import annotations

import pathlib
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

from drgct import drgc_lag_scan, drgc_stability, drgc_test, nhkj_test
from drgct.datasets import load_index, subsample, to_percentage_changes
from drgct.plots import (
    plot_bootstrap_distribution,
    plot_empirical_process,
    plot_lag_profile,
    plot_mdn_fit,
    plot_pvalue_heatmap,
    plot_series_overview,
    plot_stability,
    save_figure,
    use_journal_style,
)
from drgct.tables import export_table, table_descriptives, table_lag_orders, table_pvalues
from drgct.utils import check_stationarity

INDEX = "csi300"
LABEL = "CSI 300"
PERIOD = "2021-2024"
LAGS = range(1, 11)
SETTINGS = dict(G=10, L=20, M=20, B=999)
SEED = 20240926
OUT = pathlib.Path("results_example3")


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
    # 1. Data
    # ------------------------------------------------------------------ #
    raw = load_index(INDEX)
    pv_full = to_percentage_changes(raw)          # P = % price change, V = % volume change / 10
    pv = subsample(pv_full, PERIOD)
    P, V = pv["P"].to_numpy(), pv["V"].to_numpy()

    print(f"=== {LABEL}, {PERIOD} ===")
    print(f"  {len(pv)} daily observations, "
          f"{pv.index.min().date()} to {pv.index.max().date()}")
    for name, s in (("P_t", P), ("V_t", V)):
        print("  " + check_stationarity(s, name)["message"])

    desc = table_descriptives({f"{LABEL} $P_t$": P, f"{LABEL} $V_t$": V})
    print("\n" + desc.to_markdown(index=False))
    export_table(desc, "table0_descriptives", tdir,
                 caption=f"Descriptive statistics, {LABEL}, {PERIOD}",
                 label="tab:descriptives")
    save_figure(plot_series_overview({LABEL: raw}, {LABEL: pv_full}),
                "fig_data_overview", fdir)

    # ------------------------------------------------------------------ #
    # 2. Lag scans, both directions
    # ------------------------------------------------------------------ #
    print(f"\n=== DRGCT lag scan, P_t -> V_t  (n = {len(pv)}) ===")
    scan_pv, results_pv = drgc_lag_scan(P, V, lags=LAGS, seed=SEED,
                                        x_name="P_t", y_name="V_t", **SETTINGS)
    print(f"\n=== DRGCT lag scan, V_t -> P_t ===")
    scan_vp, _ = drgc_lag_scan(V, P, lags=LAGS, seed=SEED + 1,
                               x_name="V_t", y_name="P_t", **SETTINGS)

    long = pd.concat([
        scan_pv.assign(index_label=LABEL, period=PERIOD, direction="P_t -> V_t"),
        scan_vp.assign(index_label=LABEL, period=PERIOD, direction="V_t -> P_t"),
    ], ignore_index=True)
    long.to_csv(tdir / "raw_results.csv", index=False)

    print("\n=== Table 6 layout ===")
    t6 = table_lag_orders(long, lags=LAGS)
    print(t6.to_markdown(index=False))
    export_table(t6, "table6_lag_orders", tdir,
                 caption=f"Granger causality under specific lag orders, {LABEL}, {PERIOD}",
                 label="tab:lagorders")
    export_table(table_pvalues(long, lags=LAGS), "table6b_pvalues", tdir,
                 caption="Bootstrap p-values by lag order", label="tab:pvalues")

    # ------------------------------------------------------------------ #
    # 3. Benchmarks
    # ------------------------------------------------------------------ #
    print("\n=== Benchmarks (p-values) ===")
    print(f"  {'lag':>3s}  {'DRGC':>8s}  {'NHKJ':>8s}  {'linear VAR':>10s}")
    bench = []
    for lag in (1, 2, 3, 5, 10):
        d = float(scan_pv.loc[scan_pv["lag"] == lag, "pvalue"].iloc[0])
        k = nhkj_test(P, V, lag=lag).pvalue
        f = linear_granger(P, V, lag)
        bench.append({"lag": lag, "DRGC": d, "NHKJ": k, "linear VAR": f})
        print(f"  {lag:>3d}  {d:>8.4f}  {k:>8.4f}  {f:>10.4f}")
    export_table(pd.DataFrame(bench).round(4), "table_benchmarks", tdir,
                 caption="DRGCT against a smoothing-based and a linear benchmark, "
                         "$P_t \\to V_t$",
                 label="tab:bench",
                 notes="Smoothing-based tests are badly undersized from lag 3 onward "
                       "(Table 3 of Hui, Liu and Song, 2025), so a non-rejection by "
                       "NHKJ at high lag orders is weak evidence.")

    # ------------------------------------------------------------------ #
    # 4. Is the conclusion robust to the random directions?
    # ------------------------------------------------------------------ #
    head_lag = int(scan_pv.sort_values("pvalue")["lag"].iloc[0])
    print(f"\n=== Stability check at the most significant lag ({head_lag}) ===")
    stab = drgc_stability(P, V, lag=head_lag, n_draws=25, seed=SEED,
                          x_name="P_t", y_name="V_t", **SETTINGS)
    print(f"  median p        {stab['median']:.4f}")
    print(f"  5th-95th pct    [{stab['q05']:.4f}, {stab['q95']:.4f}]")
    print(f"  share rejecting {stab['share_reject']:.1%}")
    print(f"  merged p-value  {stab['merged_pvalue']:.4f}   (2 x median; conservative but valid)")
    save_figure(plot_stability(stab, label=f"{LABEL}, {PERIOD}, $P_t \\to V_t$"),
                "fig_stability", fdir)

    # ------------------------------------------------------------------ #
    # 5. Figures
    # ------------------------------------------------------------------ #
    print("\n=== Figures ===")
    save_figure(plot_pvalue_heatmap(long), "fig_pvalue_heatmap", fdir)
    save_figure(plot_lag_profile(scan_pv, label=f"{LABEL}, {PERIOD}, $P_t \\to V_t$"),
                "fig_lagprofile_p2v", fdir)
    save_figure(plot_lag_profile(scan_vp, label=f"{LABEL}, {PERIOD}, $V_t \\to P_t$"),
                "fig_lagprofile_v2p", fdir)

    diag = drgc_test(P, V, lag=head_lag, seed=SEED, x_name="P_t", y_name="V_t",
                     return_networks=True, **SETTINGS)
    save_figure(plot_bootstrap_distribution(diag), "fig_bootstrap_null", fdir)
    save_figure(plot_empirical_process(diag), "fig_empirical_process", fdir)
    save_figure(plot_mdn_fit(diag), "fig_mdn_fit", fdir)

    print(f"\nOutput written to {OUT.resolve()}")


if __name__ == "__main__":
    main()
