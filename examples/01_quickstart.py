#!/usr/bin/env python
"""Example 1 -- Quick start: two minutes, three tests.

Shows the three things you need on day one:

1. The test detects purely *nonlinear* causality that a linear VAR misses.
2. It correctly finds nothing in the reverse direction.
3. How to read every field of the result object.

Run:
    python examples/01_quickstart.py
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import statsmodels.api as sm

from drgct import drgc_both_directions, drgc_test


def simulate(n=600, seed=0):
    """X is an AR(1); Y depends on X only through sin(.) -- zero linear correlation."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    e1 = rng.normal(0, 0.7, n)
    e2 = rng.normal(0, 0.7, n)
    for t in range(1, n):
        x[t] = -0.5 * x[t - 1] + e1[t]
        y[t] = 0.5 * y[t - 1] + np.sin(-x[t - 1]) + e2[t]
    return x, y


def linear_granger(cause, effect, lag):
    """Textbook VAR F-test, for contrast."""
    n = len(effect)
    Yl = np.column_stack([effect[lag - j - 1: n - j - 1] for j in range(lag)])
    Xl = np.column_stack([cause[lag - j - 1: n - j - 1] for j in range(lag)])
    yy = effect[lag:]
    restricted = sm.OLS(yy, sm.add_constant(Yl)).fit()
    unrestricted = sm.OLS(yy, sm.add_constant(np.hstack([Yl, Xl]))).fit()
    return unrestricted.compare_f_test(restricted)[1]


def main() -> None:
    x, y = simulate()

    print("=" * 72)
    print("  1.  Linear VAR F-test -- the conventional benchmark")
    print("=" * 72)
    for lag in (1, 2, 3):
        print(f"  lag {lag}:  X -> Y  p = {linear_granger(x, y, lag):.4f}"
              f"      Y -> X  p = {linear_granger(y, x, lag):.4f}")
    print("\n  The linear test sees nothing: sin(.) is an odd function of a\n"
          "  symmetric shock, so the linear projection of Y on lagged X is ~0.\n")

    print("=" * 72)
    print("  2.  DRGCT, X -> Y   (the truth: X does cause Y)")
    print("=" * 72)
    res = drgc_test(x, y, lag=1, B=999, seed=1)
    res.print()

    print("\n" + "=" * 72)
    print("  3.  DRGCT, both directions at once")
    print("=" * 72)
    out = drgc_both_directions(x, y, lag=1, B=999, seed=1)
    for key, r in out.items():
        print(f"  {r.direction:<12s}  KS = {r.ks_stat:7.4f}   "
              f"p = {r.pvalue:.4f} {r.stars:<3}  ->  {r.decision}")

    print("\n" + "=" * 72)
    print("  4.  Everything the result object carries")
    print("=" * 72)
    print(f"  ks_stat            {res.ks_stat:.6f}")
    print(f"  pvalue             {res.pvalue:.6f}")
    print(f"  reject at 5%       {res.reject}")
    print(f"  critical values    {  {k: round(v, 4) for k, v in res.critical_values.items()} }")
    print(f"  n / n_eff          {res.n} / {res.n_eff}")
    print(f"  boot_stats         shape {res.boot_stats.shape}")
    print(f"  S_hat              shape {res.S_hat.shape}  (complex)")
    print(f"  influence z_(t,l)  shape {res.influence.shape}  (complex)")
    print(f"  residuals          shape {res.residuals.shape}")
    print(f"  mu / nu            {res.mu.shape} / {res.nu.shape}")
    print(f"  runtime            {res.elapsed:.2f} s")
    print("\n  As a one-row DataFrame:")
    print(res.to_frame().to_string(index=False))


if __name__ == "__main__":
    main()
