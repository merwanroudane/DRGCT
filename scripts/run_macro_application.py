#!/usr/bin/env python
"""A full macroeconomic application of the DRGCT on real US data.

Monthly FRED series, 1959-2025, and six relations that the macro literature
argues about:

    1. Fed funds rate        -> Industrial production   (Friedman's long and variable lags)
    2. Fed funds rate        -> CPI inflation           (the price leg of transmission)
    3. M2 money growth       -> CPI inflation           (the quantity theory)
    4. WTI oil price         -> CPI inflation           (energy pass-through)
    5. WTI oil price         -> Industrial production   (Hamilton's oil-shock channel)
    6. Industrial production -> Unemployment rate       (Okun's law, dynamically)

Each is tested in both directions at lag orders 1, 3, 6, 9, 12 and 18 months.
The 6-18 month range is exactly where kernel-smoothing causality tests fail and
where monetary transmission is thought to operate, so it is the natural place
to put a test that survives high-dimensional conditioning.

What the script produces
------------------------
* a series/transformation table and a descriptive-statistics table with
  stationarity screens;
* the main causality grid (ticks and p-values by lag order, both directions);
* a three-way comparison against a linear VAR F-test and the smoothing-based
  nonparametric benchmark;
* a Great Inflation (1959-1983) versus Great Moderation (1984-2025) split for
  the three monetary relations;
* a direction-draw stability check on the headline relation;
* nine figures.

Usage
-----
    python scripts/run_macro_application.py --jobs -1
    python scripts/run_macro_application.py --quick
    python scripts/run_macro_application.py --lags 1 3 6 12 --skip-subsamples
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# --------------------------------------------------------------------------- #
def linear_granger(cause, effect, lag):
    """Textbook VAR F-test, as the point of comparison every referee knows."""
    import numpy as np
    import statsmodels.api as sm

    n = len(effect)
    Yl = np.column_stack([effect[lag - j - 1: n - j - 1] for j in range(lag)])
    Xl = np.column_stack([cause[lag - j - 1: n - j - 1] for j in range(lag)])
    yy = effect[lag:]
    r = sm.OLS(yy, sm.add_constant(Yl)).fit()
    u = sm.OLS(yy, sm.add_constant(np.hstack([Yl, Xl]))).fit()
    return float(u.compare_f_test(r)[1])


def main(argv=None) -> int:
    import numpy as np
    import pandas as pd

    from drgct import drgc_stability, drgc_test, nhkj_test
    from drgct.applications import MACRO_RELATIONS, macro_study
    from drgct.datasets import MACRO_PERIODS, MACRO_SERIES, load_macro
    from drgct.plots import (
        PALETTE,
        plot_bootstrap_distribution,
        plot_empirical_process,
        plot_lag_profile,
        plot_mdn_fit,
        plot_pvalue_heatmap,
        plot_stability,
        save_figure,
        use_journal_style,
    )
    from drgct.tables import export_table, table_descriptives, table_lag_orders, table_pvalues
    from drgct.utils import check_stationarity

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lags", nargs="+", type=int, default=[1, 3, 6, 9, 12, 18])
    ap.add_argument("--sub-lags", nargs="+", type=int, default=[3, 6, 12])
    ap.add_argument("-B", type=int, default=999)
    ap.add_argument("-G", type=int, default=10)
    ap.add_argument("-L", type=int, default=60,
                    help="random directions; raised from the paper's 20 because "
                         "p + q reaches 36 at lag 18")
    ap.add_argument("-M", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=19590201)
    ap.add_argument("--stability-draws", type=int, default=30)
    ap.add_argument("--skip-subsamples", action="store_true")
    ap.add_argument("--outdir", default=str(ROOT / "results" / "macro"))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)

    if a.quick:
        a.lags, a.sub_lags, a.B, a.L = [1, 6], [6], 299, 20
        a.stability_draws, a.skip_subsamples = 4, True

    out = pathlib.Path(a.outdir)
    tdir, fdir = out / "tables", out / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    use_journal_style()
    drgc_kwargs = dict(B=a.B, G=a.G, L=a.L, M=a.M)

    # ------------------------------------------------------------------ #
    # 1. Data
    # ------------------------------------------------------------------ #
    print("\n=== Data: monthly US macro series from FRED ===")
    raw = load_macro(transform=False)
    macro = load_macro()
    print(f"  {len(macro)} monthly observations, "
          f"{macro.index.min().date()} to {macro.index.max().date()}")

    rows = []
    for fred_id, (label, how) in MACRO_SERIES.items():
        if label not in macro.columns:
            continue
        chk = check_stationarity(macro[label], label)
        rows.append({
            "FRED id": fred_id,
            "Series": label,
            "Transformation": {"logdiff": "100 x dlog", "diff": "first difference",
                               "level": "level"}[how],
            "Obs.": int(macro[label].notna().sum()),
            "ADF p": round(chk["adf_pvalue"], 4),
            "KPSS p": round(chk["kpss_pvalue"], 4),
            "Stationary": "yes" if chk["stationary"] else "no",
        })
        print(f"  {label:<24s} {chk['message'].split(': ', 1)[1]}")
    export_table(
        pd.DataFrame(rows), "macro_table1_series", tdir,
        caption="US macroeconomic series, transformations and stationarity screens",
        label="tab:macroseries",
        notes="Monthly data from FRED (Federal Reserve Bank of St. Louis), "
              "February 1959 to December 2025.  ADF reports the p-value of the "
              "augmented Dickey--Fuller unit-root null; KPSS the p-value of the "
              "stationarity null.  A series is marked stationary only when ADF "
              "rejects and KPSS does not.  Assumption 1 of Hui, Liu and Song "
              "(2025) requires stationarity.",
    )
    export_table(
        table_descriptives({c: macro[c].to_numpy() for c in macro.columns}),
        "macro_table2_descriptives", tdir,
        caption="Descriptive statistics of the transformed macroeconomic series",
        label="tab:macrodesc",
        notes="Ljung--Box statistics use 10 lags.  All series are transformed as "
              "in the previous table.",
    )

    # Figure: levels and transformed series side by side.
    import matplotlib.pyplot as plt

    cols = [c for c in macro.columns if c != "PCE inflation"]
    fig, axes = plt.subplots(len(cols), 2, figsize=(9.4, 1.5 * len(cols) + 0.8),
                             sharex=True)
    inv = {v[0]: k for k, v in MACRO_SERIES.items()}
    for i, c in enumerate(cols):
        lv = raw[inv[c]]
        axes[i][0].plot(lv.index, lv.to_numpy(), color=PALETTE["blue"], lw=0.9)
        axes[i][1].plot(macro.index, macro[c].to_numpy(), color=PALETTE["terracotta"],
                        lw=0.5, alpha=0.9)
        axes[i][0].set_ylabel(c, fontsize=8)
        for j in (0, 1):
            axes[i][j].grid(axis="x", visible=False)
        if i == 0:
            axes[i][0].set_title("level", loc="left")
            axes[i][1].set_title("transformed (stationary)", loc="left")
    fig.suptitle("US macroeconomic data, monthly, 1959-2025 (FRED)", y=1.0, fontsize=11)
    fig.tight_layout()
    save_figure(fig, "macro_fig1_series", fdir)

    # ------------------------------------------------------------------ #
    # 2. Main causality grid
    # ------------------------------------------------------------------ #
    print("\n=== Main grid: six relations, both directions, full sample ===")
    df = macro_study(
        MACRO_RELATIONS, periods=["Full sample"], lags=a.lags,
        alpha=a.alpha, drgc_kwargs=drgc_kwargs, seed=a.seed, n_jobs=a.jobs,
        out_csv=tdir / "macro_raw.csv",
    )

    export_table(
        table_lag_orders(df, alpha=a.alpha, lags=a.lags), "macro_table3_lag_orders", tdir,
        caption="Granger causality in mean among US macroeconomic series, by lag order",
        label="tab:macrolags",
        notes=f"Ticks mark rejection of the null of non-causality at the "
              f"{100 * a.alpha:g}\\% level, bootstrap critical values with $B={a.B}$, "
              f"$G={a.G}$, $L={a.L}$, $M={a.M}$.  Lag orders are months.",
    )
    export_table(
        table_pvalues(df, lags=a.lags), "macro_table4_pvalues", tdir,
        caption="Bootstrap p-values of the DRGCT by lag order",
        label="tab:macropvals", float_format="%.3f",
    )
    save_figure(plot_pvalue_heatmap(df, alpha=a.alpha,
                                    index_order=[r[2] for r in MACRO_RELATIONS]),
                "macro_fig2_pvalue_heatmap", fdir)

    for cause, effect, short, _q in MACRO_RELATIONS:
        s = df[(df["index_label"] == short) & (df["direction"] == f"{cause} -> {effect}")]
        if s.empty:
            continue
        tag = short.lower().replace(" ", "_")
        save_figure(plot_lag_profile(s, alpha=a.alpha, label=f"{cause} $\\to$ {effect}"),
                    f"macro_fig3_lagprofile_{tag}", fdir)

    # ------------------------------------------------------------------ #
    # 3. Three-way comparison: DRGCT vs linear VAR vs smoothing benchmark
    # ------------------------------------------------------------------ #
    print("\n=== Comparison against a linear VAR F-test and the smoothing benchmark ===")
    comp = []
    for cause, effect, short, _q in MACRO_RELATIONS:
        u = macro[cause].to_numpy()
        v = macro[effect].to_numpy()
        for lag in a.lags:
            d = df[(df["index_label"] == short) & (df["lag"] == lag)
                   & (df["direction"] == f"{cause} -> {effect}")]["pvalue"]
            comp.append({
                "Relation": short,
                "Lag": lag,
                "DRGCT": float(d.iloc[0]) if len(d) else np.nan,
                "Linear VAR": linear_granger(u, v, lag),
                "NHKJ": nhkj_test(u, v, lag=lag).pvalue,
            })
        print(f"  {short:<24s} done")
    comp = pd.DataFrame(comp)
    export_table(
        comp.round(4), "macro_table5_comparison", tdir,
        caption="DRGCT against a linear VAR $F$-test and a smoothing-based "
                "nonparametric benchmark: bootstrap and asymptotic p-values",
        label="tab:macrocomp", float_format="%.3f",
        notes="Smoothing-based tests lose size control once the conditioning "
              "dimension exceeds two or three, so a non-rejection in the NHKJ "
              "column at lag 6 or beyond is weak evidence.  The linear column is "
              "correctly sized but blind to nonlinearity.",
    )

    # A dot plot is far easier to read than the table.
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(comp) / 2 + 1.4))
    labels, ys = [], []
    for i, (rel, g) in enumerate(comp.groupby("Relation", sort=False)):
        for _, r in g.iterrows():
            ys.append(len(ys))
            labels.append(f"{rel}  ({int(r['Lag'])}m)")
    ys = np.arange(len(comp))
    ax.axvspan(0, a.alpha, color=PALETTE["sand"], alpha=0.6, lw=0, zorder=0)
    ax.axvline(a.alpha, color=PALETTE["muted"], lw=0.8, ls=(0, (4, 3)), zorder=1)
    for col, colour, marker in (("DRGCT", PALETTE["blue"], "o"),
                                ("Linear VAR", PALETTE["green"], "s"),
                                ("NHKJ", PALETTE["terracotta"], "^")):
        ax.scatter(comp[col], ys, s=22, color=colour, marker=marker, label=col, zorder=3)
    ax.set_yticks(ys, labels, fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("p-value")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", ncol=3)
    ax.set_title("Three tests, same data: shaded band is rejection at 5%", loc="left")
    fig.tight_layout()
    save_figure(fig, "macro_fig4_comparison", fdir)

    # ------------------------------------------------------------------ #
    # 4. Great Inflation versus Great Moderation
    # ------------------------------------------------------------------ #
    if not a.skip_subsamples:
        print("\n=== Sub-samples: Great Inflation vs Great Moderation ===")
        monetary = [r for r in MACRO_RELATIONS if r[2] in
                    ("MP to output", "MP to prices", "Money to prices")]
        sub = macro_study(
            monetary,
            periods=[k for k in MACRO_PERIODS if k != "Full sample"],
            lags=a.sub_lags, alpha=a.alpha, drgc_kwargs=drgc_kwargs,
            seed=a.seed + 51, n_jobs=a.jobs,
            out_csv=tdir / "macro_subsample_raw.csv",
        )
        export_table(
            table_pvalues(sub, lags=a.sub_lags), "macro_table6_subsamples", tdir,
            caption="Monetary relations before and after the Great Moderation",
            label="tab:macrosub", float_format="%.3f",
            notes="The split is at January 1984, the conventional Great Moderation "
                  "date (McConnell and Perez-Quiros, 2000; Stock and Watson, 2002).  "
                  "Sample sizes are roughly 299 and 504 months.",
        )
        save_figure(
            plot_pvalue_heatmap(sub, alpha=a.alpha,
                                index_order=[r[2] for r in monetary]),
            "macro_fig5_subsamples", fdir,
        )

    # ------------------------------------------------------------------ #
    # 5. Headline relation: diagnostics and stability
    # ------------------------------------------------------------------ #
    fwd = df[df["forward"]]
    head = fwd.loc[fwd["pvalue"].idxmin()]
    h_cause, h_effect, h_lag = head["cause"], head["effect"], int(head["lag"])
    print(f"\n=== Headline: {h_cause} -> {h_effect} at lag {h_lag} "
          f"(p = {head['pvalue']:.4f}) ===")
    u = macro[h_cause].to_numpy()
    v = macro[h_effect].to_numpy()

    res = drgc_test(u, v, lag=h_lag, seed=a.seed, alpha=a.alpha,
                    x_name=h_cause, y_name=h_effect, return_networks=True, **drgc_kwargs)
    res.print()
    (tdir / "macro_headline_summary.txt").write_text(res.summary(), encoding="utf-8")
    save_figure(plot_bootstrap_distribution(res), "macro_fig6_bootstrap_null", fdir)
    save_figure(plot_empirical_process(res), "macro_fig7_empirical_process", fdir)
    save_figure(plot_mdn_fit(res), "macro_fig8_mdn_fit", fdir)

    if a.stability_draws > 0:
        print(f"  stability over {a.stability_draws} random-direction draws")
        stab = drgc_stability(u, v, lag=h_lag, n_draws=a.stability_draws, seed=a.seed,
                              x_name=h_cause, y_name=h_effect, alpha=a.alpha, **drgc_kwargs)
        pd.DataFrame({"draw": range(1, len(stab["pvalues"]) + 1),
                      "ks_stat": stab["ks_stats"],
                      "pvalue": stab["pvalues"]}).to_csv(
            tdir / "macro_headline_stability.csv", index=False)
        export_table(
            pd.DataFrame([{
                "Relation": f"{h_cause} -> {h_effect}",
                "Lag": h_lag,
                "Median p": round(stab["median"], 4),
                "5th pct": round(stab["q05"], 4),
                "95th pct": round(stab["q95"], 4),
                "Share rejecting": round(stab["share_reject"], 3),
                "Merged p": round(stab["merged_pvalue"], 4),
            }]),
            "macro_table7_stability", tdir,
            caption="Sensitivity of the headline p-value to the random draw of "
                    "$(\\mu_\\ell,\\nu_\\ell)$",
            label="tab:macrostab",
            notes=f"{a.stability_draws} independent draws of the $L={a.L}$ direction "
                  "pairs and of the network initialisation, data held fixed.  The "
                  "merged p-value is twice the median, valid under arbitrary "
                  "dependence.",
        )
        save_figure(plot_stability(stab, label=f"{h_cause} $\\to$ {h_effect}"),
                    "macro_fig9_stability", fdir)

    # ------------------------------------------------------------------ #
    print(f"\n[drgct] macro application written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
