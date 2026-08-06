"""Command-line interface: ``drgct <command> [options]``.

Four commands, each a thin wrapper around the Python API:

``drgct test``
    Run one DRGCT (or a lag scan) on two columns of a CSV.
``drgct simulate``
    Run the Section 4 Monte Carlo and write Tables 3-4 plus the size/power
    figures.
``drgct app``
    Run the Section 5 price-volume study and write Tables 5-6 plus the
    heat-map figure.
``drgct info``
    Print version, bundled datasets, and the citation.

Every command takes ``--outdir`` and writes ``tables/`` and ``figures/``
sub-folders beneath it.
"""

from __future__ import annotations

import argparse
import pathlib
import sys


# --------------------------------------------------------------------------- #
def _add_drgc_options(ap):
    ap.add_argument("-G", type=int, default=10, help="MDN mixture components (default 10)")
    ap.add_argument("-L", type=int, default=20, help="number of (mu, nu) pairs (default 20)")
    ap.add_argument("-M", type=int, default=20, help="pseudo-samples per observation (default 20)")
    ap.add_argument("-B", type=int, default=1000, help="bootstrap replications (default 1000)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20250915)
    ap.add_argument("--epochs", type=int, default=None, help="override MLP/MDN epochs")


def _drgc_kwargs(a) -> dict:
    kw = dict(G=a.G, L=a.L, M=a.M, B=a.B)
    if getattr(a, "epochs", None):
        from .nets import MDNConfig, MLPConfig

        kw["mlp"] = MLPConfig(epochs=a.epochs)
        kw["mdn"] = MDNConfig(epochs=a.epochs)
    return kw


# --------------------------------------------------------------------------- #
def cmd_test(a) -> int:
    import pandas as pd

    from .core import drgc_lag_scan, drgc_test
    from .plots import plot_bootstrap_distribution, plot_empirical_process, plot_lag_profile, save_figure
    from .tables import export_table

    df = pd.read_csv(a.csv)
    for c in (a.x, a.y):
        if c not in df.columns:
            print(f"error: column {c!r} not in {a.csv}; found {list(df.columns)}", file=sys.stderr)
            return 2
    x = df[a.x].to_numpy(float)
    y = df[a.y].to_numpy(float)
    out = pathlib.Path(a.outdir)

    if a.lag_scan:
        lags = range(a.lag_min, a.lag_max + 1)
        print(f"[drgct] lag scan {a.x} -> {a.y}, lags {a.lag_min}..{a.lag_max}")
        scan, _ = drgc_lag_scan(x, y, lags=lags, seed=a.seed, x_name=a.x, y_name=a.y,
                                alpha=a.alpha, **_drgc_kwargs(a))
        export_table(scan.round(4), "lag_scan", out / "tables",
                     caption=f"DRGCT lag scan, {a.x} $\\to$ {a.y}", label="tab:lagscan")
        save_figure(plot_lag_profile(scan, alpha=a.alpha, label=f"{a.x} -> {a.y}"),
                    "lag_profile", out / "figures")
        return 0

    res = drgc_test(x, y, lag=a.lag, seed=a.seed, alpha=a.alpha,
                    x_name=a.x, y_name=a.y, **_drgc_kwargs(a))
    res.print()
    if a.save:
        export_table(res.to_frame().round(5), "drgct_result", out / "tables",
                     caption=f"DRGCT, {a.x} $\\to$ {a.y}, lag {a.lag}", label="tab:drgct")
        save_figure(plot_bootstrap_distribution(res), "bootstrap_null", out / "figures")
        save_figure(plot_empirical_process(res), "empirical_process", out / "figures")
    return 0


def cmd_simulate(a) -> int:
    from .plots import plot_power, plot_pvalue_ecdf, plot_size, save_figure
    from .simulate import monte_carlo, summarize
    from .tables import export_table, table_power, table_size

    out = pathlib.Path(a.outdir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    methods = tuple(a.methods)
    mc = monte_carlo(
        dgps=a.dgps,
        ns=a.ns,
        lags=range(a.lag_min, a.lag_max + 1),
        reps=a.reps,
        methods=methods,
        alpha=a.alpha,
        seed=a.seed,
        n_jobs=a.jobs,
        drgc_kwargs=_drgc_kwargs(a),
        out_csv=out / "tables" / "monte_carlo_raw.csv",
    )
    summ = summarize(mc, alpha=a.alpha)
    export_table(summ.round(4), "monte_carlo_summary", out / "tables",
                 caption="Monte-Carlo rejection frequencies", label="tab:mcsummary")

    size_dgps = [d for d in a.dgps if d.upper().startswith("S")]
    power_dgps = [d for d in a.dgps if d.upper().startswith("P")]
    if size_dgps:
        export_table(table_size(mc, dgps=size_dgps, methods=methods, alpha=a.alpha),
                     "table3_size", out / "tables",
                     caption="Empirical sizes under varying lags",
                     label="tab:size",
                     notes=f"Nominal level {100 * a.alpha:g}\\%; {a.reps} Monte-Carlo replications.")
        save_figure(plot_size(summ, nominal=a.alpha, dgps=size_dgps, methods=methods),
                    "fig_size", out / "figures")
        save_figure(plot_pvalue_ecdf(mc, dgps=size_dgps, methods=methods),
                    "fig_pvalue_ecdf", out / "figures")
    if power_dgps:
        export_table(table_power(mc, dgps=power_dgps, methods=methods, alpha=a.alpha),
                     "table4_power", out / "tables",
                     caption="Empirical powers under varying lags",
                     label="tab:power",
                     notes=f"Nominal level {100 * a.alpha:g}\\%; {a.reps} Monte-Carlo replications.")
        save_figure(plot_power(summ, dgps=power_dgps, methods=methods),
                    "fig_power", out / "figures")
    return 0


def cmd_app(a) -> int:
    from .applications import price_volume_study
    from .plots import plot_pvalue_heatmap, save_figure
    from .tables import export_table, table_detection, table_lag_orders, table_pvalues

    out = pathlib.Path(a.outdir)
    df = price_volume_study(
        indices=a.indices,
        periods=a.periods,
        lags=range(a.lag_min, a.lag_max + 1),
        alpha=a.alpha,
        drgc_kwargs=_drgc_kwargs(a),
        seed=a.seed,
        n_jobs=a.jobs,
        out_csv=out / "tables" / "price_volume_raw.csv",
    )
    export_table(table_detection(df, alpha=a.alpha), "table5_detection", out / "tables",
                 caption="Price--volume Granger causality detection", label="tab:detect",
                 notes=f"A tick marks rejection of non-causality at the "
                       f"{100 * a.alpha:g}\\% level at one or more lag orders.")
    export_table(table_lag_orders(df, alpha=a.alpha), "table6_lag_orders", out / "tables",
                 caption="Price--volume Granger causality under specific lag orders",
                 label="tab:lagorders")
    export_table(table_pvalues(df), "table6b_pvalues", out / "tables",
                 caption="Bootstrap p-values by lag order", label="tab:pvals")
    save_figure(plot_pvalue_heatmap(df, alpha=a.alpha), "fig_pvalue_heatmap", out / "figures")
    return 0


def cmd_info(a) -> int:
    from . import PAPER, __version__, cite
    from .datasets import available_datasets, data_dir

    print(f"drgct {__version__}")
    print(f"  paper : {PAPER['title']}")
    print(f"          {', '.join(PAPER['authors'])} ({PAPER['year']}), {PAPER['url']}")
    try:
        print(f"  data  : {data_dir()}")
        print(f"          {', '.join(available_datasets())}")
    except FileNotFoundError as exc:
        print(f"  data  : {exc}")
    print()
    print(cite("bibtex" if a.bibtex else "text"))
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="drgct",
        description="Deep-learning based doubly robust Granger causality test "
                    "(Hui, Liu & Song, 2025).",
    )
    ap.add_argument("--version", action="store_true", help="print the version and exit")
    sub = ap.add_subparsers(dest="command")

    # ---- test ---- #
    t = sub.add_parser("test", help="run the DRGCT on two columns of a CSV")
    t.add_argument("csv")
    t.add_argument("-x", required=True, help="column name of the candidate cause")
    t.add_argument("-y", required=True, help="column name of the effect")
    t.add_argument("--lag", type=int, default=1)
    t.add_argument("--lag-scan", action="store_true", help="scan lags instead of one test")
    t.add_argument("--lag-min", type=int, default=1)
    t.add_argument("--lag-max", type=int, default=10)
    t.add_argument("--outdir", default="results")
    t.add_argument("--save", action="store_true", help="also write tables and figures")
    _add_drgc_options(t)
    t.set_defaults(func=cmd_test)

    # ---- simulate ---- #
    s = sub.add_parser("simulate", help="Section 4 Monte Carlo (Tables 3-4)")
    s.add_argument("--dgps", nargs="+", default=["S1", "S2", "P1", "P2", "P3", "P4"])
    s.add_argument("--ns", nargs="+", type=int, default=[500, 1000, 2000])
    s.add_argument("--lag-min", type=int, default=1)
    s.add_argument("--lag-max", type=int, default=5)
    s.add_argument("--reps", type=int, default=1000)
    s.add_argument("--methods", nargs="+", default=["drgc", "nhkj"],
                   choices=["drgc", "drgc_naive", "nhkj"])
    s.add_argument("--jobs", type=int, default=1)
    s.add_argument("--outdir", default="results")
    _add_drgc_options(s)
    s.set_defaults(func=cmd_simulate)

    # ---- app ---- #
    p = sub.add_parser("app", help="Section 5 price-volume study (Tables 5-6)")
    p.add_argument("--indices", nargs="+", default=["spx500", "csi300", "nikkei225"])
    p.add_argument("--periods", nargs="+", default=["2019-2022", "2020-2023", "2021-2024"])
    p.add_argument("--lag-min", type=int, default=1)
    p.add_argument("--lag-max", type=int, default=10)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--outdir", default="results")
    _add_drgc_options(p)
    p.set_defaults(func=cmd_app)

    # ---- info ---- #
    i = sub.add_parser("info", help="version, bundled data, citation")
    i.add_argument("--bibtex", action="store_true")
    i.set_defaults(func=cmd_info)
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.version:
        from . import __version__

        print(__version__)
        return 0
    if not getattr(a, "command", None):
        ap.print_help()
        return 1
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
