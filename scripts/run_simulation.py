#!/usr/bin/env python
"""Reproduce the Section 4 Monte Carlo and write Tables 3-4 and the figures.

Two experiments are run:

**Experiment A -- size and power grid.**
    All six designs of Table 1, lag orders 1-5, comparing the doubly robust
    test (``drgc``), the *naive* deep plug-in without the doubly robust
    correction (``drgc_naive``) and the smoothing-based nonparametric
    benchmark (``nhkj``).

**Experiment B -- the type-I-error blow-up of the naive plug-in.**
    Design S1 only, at larger sample sizes, which is where Section 4 reports
    naive sizes of 0.151 at ``n = 1000`` and 0.321 at ``n = 2000``.  The
    doubly robust construction exists precisely to prevent this.

Outputs land in ``results/tables`` and ``results/figures``.

Usage
-----
    python scripts/run_simulation.py                       # defaults below
    python scripts/run_simulation.py --reps 1000 --ns 500 1000 2000 --jobs 10
    python scripts/run_simulation.py --quick                # 2-minute smoke test
    python scripts/run_simulation.py --skip-b               # experiment A only

The paper's own settings are ``--reps 1000 --ns 500 1000 2000 -B 1000``; that
is a multi-day single-core job, so pick ``--jobs`` to match your machine.
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
    from drgct.plots import (
        plot_power,
        plot_pvalue_ecdf,
        plot_size,
        save_figure,
        use_journal_style,
    )
    from drgct.simulate import monte_carlo, summarize
    from drgct.tables import (
        export_table,
        table_dgp_definitions,
        table_parameter_settings,
        table_power,
        table_size,
    )

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--ns", nargs="+", type=int, default=[500])
    ap.add_argument("--lags", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--dgps", nargs="+", default=["S1", "S2", "P1", "P2", "P3", "P4"])
    ap.add_argument("--methods", nargs="+", default=["drgc", "drgc_naive", "nhkj"])
    ap.add_argument("-B", type=int, default=499)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=20250915)
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    ap.add_argument("--quick", action="store_true", help="tiny grid, for testing the plumbing")
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--b-ns", nargs="+", type=int, default=[1000, 2000])
    ap.add_argument("--b-lags", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--b-reps", type=int, default=150)
    a = ap.parse_args(argv)

    if a.quick:
        a.reps, a.ns, a.lags = 20, [300], [1, 2]
        a.dgps, a.B, a.b_reps = ["S1", "P1"], 199, 20
        a.b_ns, a.b_lags = [400], [1]

    out = pathlib.Path(a.outdir)
    tdir, fdir = out / "tables", out / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    use_journal_style()

    # ---------------------------------------------------------------- #
    # Design documentation (Tables 1 and 2) -- no computation required
    # ---------------------------------------------------------------- #
    print("\n=== Design documentation ===")
    export_table(
        table_dgp_definitions(), "table1_dgps", tdir,
        caption="Data generating processes",
        label="tab:dgps",
        notes="$\\varepsilon_{1,t}$ and $\\varepsilon_{2,t}$ are i.i.d. $N(0,0.5)$ and mutually "
              "independent.  S1 and S2 satisfy the null of no Granger causality; "
              "P1--P4 satisfy the alternative.  Throughout, lag $=p=q$.",
        align="llll",
    )
    export_table(
        table_parameter_settings(a.lags), "table2_parameters", tdir,
        caption="Parameter settings by lag order",
        label="tab:params",
        notes="Coefficients are chosen so that the multi-lag sums in $Y_t$ do not diverge.",
    )

    summ_all = None

    # ---------------------------------------------------------------- #
    # Experiment A
    # ---------------------------------------------------------------- #
    if not a.skip_a:
        print("\n=== Experiment A: size and power grid ===")
        mc = monte_carlo(
            dgps=a.dgps, ns=a.ns, lags=a.lags, reps=a.reps,
            methods=tuple(a.methods), alpha=a.alpha, seed=a.seed, n_jobs=a.jobs,
            drgc_kwargs=dict(B=a.B),
            out_csv=tdir / "monte_carlo_raw.csv",
        )
        summ_all = summarize(mc, alpha=a.alpha)
        export_table(summ_all.round(4), "monte_carlo_summary", tdir,
                     caption="Monte-Carlo rejection frequencies with simulation standard errors",
                     label="tab:mcsummary")

        size_dgps = [d for d in a.dgps if d.upper().startswith("S")]
        power_dgps = [d for d in a.dgps if d.upper().startswith("P")]
        note = (f"Nominal level {100 * a.alpha:g}\\%; {a.reps} Monte-Carlo replications; "
                f"$B={a.B}$ bootstrap draws; $G=10$, $L=20$, $M=20$.  "
                "DRGC is the doubly robust test of Hui, Liu and Song (2025); "
                "DRGC-naive drops the conditional characteristic function correction; "
                "NHKJ is the smoothing-based nonparametric benchmark.")

        if size_dgps:
            export_table(table_size(mc, dgps=size_dgps, methods=tuple(a.methods), alpha=a.alpha),
                         "table3_size", tdir,
                         caption="Empirical sizes under varying lags", label="tab:size", notes=note)
            save_figure(plot_size(summ_all, nominal=a.alpha, dgps=size_dgps,
                                  methods=tuple(a.methods)), "fig1_size", fdir)
            save_figure(plot_pvalue_ecdf(mc, dgps=size_dgps, methods=tuple(a.methods)),
                        "fig2_pvalue_ecdf", fdir)
        if power_dgps:
            export_table(table_power(mc, dgps=power_dgps, methods=tuple(a.methods), alpha=a.alpha),
                         "table4_power", tdir,
                         caption="Empirical powers under varying lags", label="tab:power", notes=note)
            save_figure(plot_power(summ_all, dgps=power_dgps, methods=tuple(a.methods)),
                        "fig3_power", fdir)

    # ---------------------------------------------------------------- #
    # Experiment B
    # ---------------------------------------------------------------- #
    if not a.skip_b:
        print("\n=== Experiment B: why the doubly robust correction is needed ===")
        mcb = monte_carlo(
            dgps=["S1"], ns=a.b_ns, lags=a.b_lags, reps=a.b_reps,
            methods=("drgc", "drgc_naive"), alpha=a.alpha,
            seed=a.seed + 1, n_jobs=a.jobs, drgc_kwargs=dict(B=a.B),
            out_csv=tdir / "monte_carlo_naive_raw.csv",
        )
        summ_b = summarize(mcb, alpha=a.alpha)
        export_table(
            table_size(mcb, dgps=["S1"], methods=("drgc", "drgc_naive"), alpha=a.alpha),
            "table3b_naive_size", tdir,
            caption="Empirical size of the doubly robust test versus a naive deep plug-in "
                    "(DGP S1)",
            label="tab:naive",
            notes="DRGC-naive builds the empirical process from equation (5), i.e. without "
                  "subtracting $\\hat{\\varphi}(\\nu\\mid Y_{t-1})$.  Its size drifts upward "
                  "with $n$ because the bias then depends on the first power of the MLP "
                  "estimation error rather than on the product of two errors.",
        )
        save_figure(
            plot_size(summ_b, nominal=a.alpha, dgps=["S1"], methods=("drgc", "drgc_naive")),
            "fig4_naive_size", fdir,
        )

    print(f"\n[drgct] simulation outputs written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
