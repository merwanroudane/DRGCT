#!/usr/bin/env python
"""Reproduce the Section 5 price-volume application end to end.

Steps
-----
1. Load the three bundled indices, transform to percentage changes
   (volume changes divided by 10, as in Section 5), and screen for
   stationarity.
2. Write a descriptive-statistics table.
3. Run 3 indices x 3 overlapping three-year sub-samples x 2 directions x
   10 lag orders = **180 DRGCTs**.
4. Write Table 5 (detection), Table 6 (lag-specific ticks) and a p-value
   companion table.
5. Draw the data overview, the p-value heat map, the lag profiles, and the
   single-test diagnostics (bootstrap null, empirical process, MDN fit,
   training curves) for one headline specification.
6. Optionally run a rolling-window study, which turns the three static
   sub-samples into a continuous picture of when causality switches on.

Usage
-----
    python scripts/run_application.py                     # full study
    python scripts/run_application.py --jobs 8
    python scripts/run_application.py --quick             # one index, 4 lags
    python scripts/run_application.py --rolling --roll-step 21
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None) -> int:
    import pandas as pd

    from drgct import drgc_stability, drgc_test
    from drgct.applications import price_volume_study, rolling_causality
    from drgct.datasets import (
        INDEX_LABELS,
        PAPER_PERIODS,
        load_index,
        subsample,
        to_percentage_changes,
    )
    from drgct.plots import (
        plot_bootstrap_distribution,
        plot_empirical_process,
        plot_lag_profile,
        plot_mdn_fit,
        plot_pvalue_heatmap,
        plot_rolling_pvalue,
        plot_series_overview,
        plot_stability,
        plot_training_curves,
        save_figure,
        use_journal_style,
    )
    from drgct.tables import (
        export_table,
        table_descriptives,
        table_detection,
        table_hyperparameters,
        table_lag_orders,
        table_pvalues,
    )
    from drgct.utils import check_stationarity

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indices", nargs="+", default=["spx500", "csi300", "nikkei225"])
    ap.add_argument("--periods", nargs="+", default=list(PAPER_PERIODS))
    ap.add_argument("--lags", nargs="+", type=int, default=list(range(1, 11)))
    ap.add_argument("-B", type=int, default=999)
    ap.add_argument("-G", type=int, default=10)
    ap.add_argument("-L", type=int, default=20)
    ap.add_argument("-M", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=20240926)
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stability-draws", type=int, default=30,
                    help="independent (mu, nu) draws for the headline stability check (0 disables)")
    ap.add_argument("--rolling", action="store_true", help="also run the rolling-window study")
    ap.add_argument("--roll-index", nargs="+", default=["csi300", "spx500"],
                    help="one or more dataset keys for the rolling-window study")
    ap.add_argument("--roll-window", type=int, default=750)
    ap.add_argument("--roll-step", type=int, default=21)
    ap.add_argument("--roll-lag", type=int, default=5)
    a = ap.parse_args(argv)

    if a.quick:
        a.indices, a.periods, a.lags, a.B = ["spx500"], ["2021-2024"], [1, 2, 3, 4], 299
        a.stability_draws = 4

    out = pathlib.Path(a.outdir)
    tdir, fdir = out / "tables", out / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    use_journal_style()
    drgc_kwargs = dict(B=a.B, G=a.G, L=a.L, M=a.M)

    # ---------------------------------------------------------------- #
    # 1-2. Data, stationarity screen, descriptives
    # ---------------------------------------------------------------- #
    print("\n=== Data ===")
    raw, trans, series_map = {}, {}, {}
    for key in a.indices:
        label = INDEX_LABELS.get(key, key)
        r = load_index(key)
        t = to_percentage_changes(r)
        raw[label], trans[label] = r, t
        series_map[f"{label}  $P_t$"] = t["P"].to_numpy()
        series_map[f"{label}  $V_t$"] = t["V"].to_numpy()
        print(f"  {label:<10s} {len(r):>5d} daily observations, "
              f"{r.index.min().date()} .. {r.index.max().date()}")
        for nm, v in (("P_t", t["P"]), ("V_t", t["V"])):
            chk = check_stationarity(v, f"{label} {nm}")
            print("    " + chk["message"])

    export_table(
        table_descriptives(series_map), "table0_descriptives", tdir,
        caption="Descriptive statistics of the transformed price and volume series",
        label="tab:descriptives",
        notes="$P_t$ is the daily percentage change in the closing level; $V_t$ is the daily "
              "percentage change in trading volume divided by 10, following Section 5 of "
              "Hui, Liu and Song (2025).  Ljung--Box statistics use 10 lags.  "
              "ADF and KPSS report p-values for the unit-root and stationarity nulls.",
        index=False, float_format="%.3f",
    )
    save_figure(plot_series_overview(raw, trans), "fig5_data_overview", fdir)

    # ---------------------------------------------------------------- #
    # 3. The 180-test grid
    # ---------------------------------------------------------------- #
    print("\n=== Granger causality grid ===")
    df = price_volume_study(
        indices=a.indices, periods=a.periods, lags=a.lags,
        alpha=a.alpha, drgc_kwargs=drgc_kwargs, seed=a.seed, n_jobs=a.jobs,
        out_csv=tdir / "price_volume_raw.csv",
    )

    # ---------------------------------------------------------------- #
    # 4. Tables 5 and 6
    # ---------------------------------------------------------------- #
    print("\n=== Tables ===")
    export_table(
        table_detection(df, alpha=a.alpha), "table5_detection", tdir,
        caption="Price--volume Granger causality detection",
        label="tab:detection",
        notes=f"A tick marks rejection of the null of non-causality at the "
              f"{100 * a.alpha:g}\\% level for at least one lag order between "
              f"{min(a.lags)} and {max(a.lags)}; a cross marks no rejection at any lag.",
    )
    export_table(
        table_lag_orders(df, alpha=a.alpha, lags=a.lags), "table6_lag_orders", tdir,
        caption="Price--volume Granger causality under specific lag orders",
        label="tab:lagorders",
        notes=f"Ticks mark rejection at the {100 * a.alpha:g}\\% level using the bootstrap "
              f"critical value with $B={a.B}$ replications.",
    )
    export_table(
        table_pvalues(df, lags=a.lags), "table6b_pvalues", tdir,
        caption="Bootstrap p-values of the DRGCT by lag order",
        label="tab:pvalues", float_format="%.3f",
    )

    # ---------------------------------------------------------------- #
    # 5. Figures
    # ---------------------------------------------------------------- #
    print("\n=== Figures ===")
    save_figure(plot_pvalue_heatmap(df, alpha=a.alpha), "fig6_pvalue_heatmap", fdir)

    for key in a.indices:
        label = INDEX_LABELS.get(key, key)
        sub = df[(df["index_label"] == label) & (df["period"] == a.periods[-1])]
        for direction, tag in (("P_t -> V_t", "p2v"), ("V_t -> P_t", "v2p")):
            s = sub[sub["direction"] == direction]
            if s.empty:
                continue
            save_figure(
                plot_lag_profile(s, alpha=a.alpha,
                                 label=f"{label}, {a.periods[-1]}, {direction}"),
                f"fig7_lagprofile_{key}_{tag}", fdir,
            )

    # Headline single test, with full diagnostics.
    head_key = a.indices[0]
    head_period = a.periods[-1]
    pv = subsample(to_percentage_changes(load_index(head_key)), head_period)
    head_lag = max(a.lags) if max(a.lags) <= 10 else 10
    print(f"  headline diagnostics: {INDEX_LABELS.get(head_key, head_key)}, "
          f"{head_period}, P_t -> V_t, lag {head_lag}")
    res = drgc_test(
        pv["P"].to_numpy(), pv["V"].to_numpy(), lag=head_lag,
        x_name="P_t", y_name="V_t", seed=a.seed, return_networks=True, **drgc_kwargs,
    )
    res.print()
    (tdir / "headline_summary.txt").write_text(res.summary(), encoding="utf-8")
    export_table(table_hyperparameters(res), "table7_hyperparameters", tdir,
                 caption="Hyper-parameters of the reported DRGCT", label="tab:hyper")
    save_figure(plot_bootstrap_distribution(res), "fig8_bootstrap_null", fdir)
    save_figure(plot_empirical_process(res), "fig9_empirical_process", fdir)
    save_figure(plot_mdn_fit(res), "fig10_mdn_fit", fdir)
    save_figure(plot_training_curves(res), "fig11_training_curves", fdir)

    # How much of the p-value is simulation noise from the random (mu, nu) draw?
    if a.stability_draws > 0:
        print(f"  stability check over {a.stability_draws} random-direction draws")
        stab = drgc_stability(
            pv["P"].to_numpy(), pv["V"].to_numpy(), lag=head_lag,
            n_draws=a.stability_draws, seed=a.seed, x_name="P_t", y_name="V_t",
            alpha=a.alpha, **drgc_kwargs,
        )
        pd.DataFrame({"draw": range(1, len(stab["pvalues"]) + 1),
                      "ks_stat": stab["ks_stats"],
                      "pvalue": stab["pvalues"]}).to_csv(
            tdir / "headline_stability.csv", index=False)
        export_table(
            pd.DataFrame([{
                "Median p": round(stab["median"], 4),
                "Mean p": round(stab["mean"], 4),
                "5th pct": round(stab["q05"], 4),
                "95th pct": round(stab["q95"], 4),
                "Share rejecting": round(stab["share_reject"], 3),
                "Merged p (2x median)": round(stab["merged_pvalue"], 4),
            }]),
            "table8_stability", tdir,
            caption="Sensitivity of the DRGCT p-value to the random draw of "
                    "$(\\mu_\\ell,\\nu_\\ell)$",
            label="tab:stability",
            notes=f"{a.stability_draws} independent draws of the $L={a.L}$ direction pairs and "
                  "of the network initialisation, holding the data fixed.  The merged p-value "
                  "is twice the median, which is a valid (conservative) p-value under arbitrary "
                  "dependence.",
        )
        save_figure(
            plot_stability(stab, label=f"{INDEX_LABELS.get(head_key, head_key)}, "
                                       f"{head_period}, $P_t \\to V_t$"),
            "fig13_stability", fdir,
        )

    # ---------------------------------------------------------------- #
    # 6. Rolling window
    # ---------------------------------------------------------------- #
    if a.rolling:
        print("\n=== Rolling window ===")
        for key in a.roll_index:
            full = to_percentage_changes(load_index(key))
            roll = rolling_causality(
                full["P"].to_numpy(), full["V"].to_numpy(),
                lag=a.roll_lag, window=a.roll_window, step=a.roll_step,
                dates=full.index, drgc_kwargs=drgc_kwargs, n_jobs=a.jobs, seed=a.seed,
            )
            roll.to_csv(tdir / f"rolling_{key}.csv", index=False)
            save_figure(
                plot_rolling_pvalue(
                    roll, alpha=a.alpha,
                    label=f"{INDEX_LABELS.get(key, key)}: rolling DRGCT, "
                          f"window {a.roll_window}, lag {a.roll_lag}",
                ),
                f"fig12_rolling_{key}", fdir,
            )

    print(f"\n[drgct] application outputs written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
