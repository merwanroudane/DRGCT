#!/usr/bin/env python
"""Example 2 -- Section 4 in miniature: size, power, and why double robustness matters.

Runs a small but genuine version of the paper's Monte Carlo:

* DGP S1 (null) and DGP P2 (alternative), lag orders 1, 3 and 5, n = 500;
* three estimators side by side -- the doubly robust test, the *naive* deep
  plug-in that omits the correction term, and the smoothing-based
  nonparametric benchmark;
* Tables 3 and 4 in the paper's layout, plus the size and power figures.

About 6 minutes on 8 cores with the defaults below. Raise ``REPS`` for
publication-grade numbers; ``scripts/run_simulation.py`` runs the full grid.

Run:
    python examples/02_simulation_size_power.py
"""

from __future__ import annotations

import pathlib
import warnings

warnings.filterwarnings("ignore")

from drgct.plots import plot_power, plot_pvalue_ecdf, plot_size, save_figure, use_journal_style
from drgct.simulate import monte_carlo, summarize
from drgct.tables import export_table, table_power, table_size

REPS = 60          # the paper uses 1000
NS = [500]
LAGS = [1, 3, 5]
DGPS = ["S1", "P2"]
METHODS = ("drgc", "drgc_naive", "nhkj")
B = 399            # the paper uses 1000
JOBS = -1
OUT = pathlib.Path("results_example2")


def main() -> None:
    use_journal_style()
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)

    mc = monte_carlo(
        dgps=DGPS, ns=NS, lags=LAGS, reps=REPS,
        methods=METHODS, alpha=0.05, seed=20250915, n_jobs=JOBS,
        drgc_kwargs=dict(B=B),
        out_csv=OUT / "tables" / "monte_carlo_raw.csv",
    )

    summ = summarize(mc)
    print("\n=== Rejection frequencies (with Monte-Carlo standard errors) ===")
    print(summ[["dgp", "n", "lag", "method", "reps", "rejection_rate", "mc_se"]]
          .to_string(index=False))

    note = (f"{REPS} Monte-Carlo replications, $B={B}$ bootstrap draws, nominal level 5\\%.  "
            "DRGC-naive omits the doubly robust correction term.")

    print("\n=== Table 3 layout: empirical size (DGP S1, the null) ===")
    t3 = table_size(mc, dgps=["S1"], methods=METHODS)
    print(t3.to_markdown(index=False))
    export_table(t3, "table3_size", OUT / "tables",
                 caption="Empirical sizes under varying lags", label="tab:size", notes=note)

    print("\n=== Table 4 layout: empirical power (DGP P2, the alternative) ===")
    t4 = table_power(mc, dgps=["P2"], methods=METHODS)
    print(t4.to_markdown(index=False))
    export_table(t4, "table4_power", OUT / "tables",
                 caption="Empirical powers under varying lags", label="tab:power", notes=note)

    print("\n=== Figures ===")
    save_figure(plot_size(summ, dgps=["S1"], methods=METHODS), "fig_size", OUT / "figures")
    save_figure(plot_power(summ, dgps=["P2"], methods=METHODS), "fig_power", OUT / "figures")
    save_figure(plot_pvalue_ecdf(mc, dgps=["S1"], methods=METHODS),
                "fig_pvalue_ecdf", OUT / "figures")

    print(f"""
=== What to look for ===

Size (DGP S1, no causality, nominal 5%)
  DRGC        should sit near 0.05 at every lag order.  This is the headline
              result: the doubly robust construction controls the type I error
              even though both nuisance estimators converge slowly.
  DRGC-naive  should be visibly *off* 0.05.  Section 4 of the paper reports it
              over-rejecting (0.151 at n = 1000, 0.321 at n = 2000, lag 5); in
              this implementation it under-rejects instead, because the
              in-sample least-squares residual is near-orthogonal to functions
              of Y_{t-1}.  Either way it is not a valid test.
  NHKJ        the paper reports severe undersizing from lag 2 onward (0.003 to
              0.043 against a 5% nominal level, Table 3).  Our implementation
              is undersized in S2 and at high lag in S1 but over-sized at
              lag 1 in S1 -- see results/README.md.

Power (DGP P2, causality present)
  DRGC        near 1.0 at lags 1-3, still useful at lags 4-5.
  NHKJ        the paper reports a collapse (0.052 at lag 5 with n = 500,
              Table 4).  Ours does not collapse -- but its size is not
              controlled, so its power is not comparable.  Read the size panel
              before the power panel.

p-value plot (fig_pvalue_ecdf)
  Under the null the ECDF of the p-values should follow the 45-degree line.
  Above it means over-rejection, below it means a conservative test.

Output written to {OUT.resolve()}
""")


if __name__ == "__main__":
    main()
