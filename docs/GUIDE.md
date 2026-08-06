# The DRGCT Applied Researcher's Guide

**How to run a deep-learning based doubly robust Granger causality analysis, end to end, and write it up.**

This guide is written for an applied economist, finance researcher or data
scientist who has two time series and a causal question, and who wants a
result that survives a referee. It assumes no deep-learning background. Every
code block is complete and runnable — copy them in order and you will finish
with a full set of journal-ready tables and figures.

Method: Hui, Y., Liu, C. and Song, X. (2025), *Deep learning based doubly
robust test for Granger causality*, [arXiv:2509.15798](https://arxiv.org/abs/2509.15798).
Software: [`drgct`](https://github.com/merwanroudane/DRGCT).

---

## Table of contents

1. [What you are going to build](#1-what-you-are-going-to-build)
2. [Installation and a 30-second check](#2-installation-and-a-30-second-check)
3. [The method in ten minutes](#3-the-method-in-ten-minutes)
4. [Design decisions to make *before* you write code](#4-design-decisions-to-make-before-you-write-code)
5. [Step-by-step: from raw data to a reported test](#5-step-by-step-from-raw-data-to-a-reported-test)
   - [5.1 Load and align the data](#51-load-and-align-the-data)
   - [5.2 Make the series stationary](#52-make-the-series-stationary)
   - [5.3 Descriptive statistics](#53-descriptive-statistics)
   - [5.4 Choose the hyper-parameters](#54-choose-the-hyper-parameters)
   - [5.5 Your first test](#55-your-first-test)
   - [5.6 Read the output line by line](#56-read-the-output-line-by-line)
   - [5.7 Scan the lag orders, both directions](#57-scan-the-lag-orders-both-directions)
   - [5.8 Check that the result is not an artefact of the random directions](#58-check-that-the-result-is-not-an-artefact-of-the-random-directions)
   - [5.9 Sub-samples and rolling windows](#59-sub-samples-and-rolling-windows)
   - [5.10 Benchmark against a smoothing-based test](#510-benchmark-against-a-smoothing-based-test)
6. [Building the tables](#6-building-the-tables)
7. [Building the figures](#7-building-the-figures)
8. [The complete application script](#8-the-complete-application-script)
9. [Verifying size for *your* design with a bespoke Monte Carlo](#9-verifying-size-for-your-design-with-a-bespoke-monte-carlo)
10. [Performance, scaling and reproducibility](#10-performance-scaling-and-reproducibility)
11. [Troubleshooting](#11-troubleshooting)
12. [Reporting checklist and a write-up template](#12-reporting-checklist-and-a-write-up-template)
13. [How the results shipped in this repository were produced](#13-how-the-results-shipped-in-this-repository-were-produced)

---

## 1. What you are going to build

By the end of this guide you will have produced, for your own pair of series:

| Deliverable | File(s) |
|---|---|
| Descriptive statistics with stationarity screens | `table0_descriptives.{tex,md,csv}` |
| Causality detection summary (tick/cross by direction and sub-sample) | `table5_detection.*` |
| Lag-by-lag decision table | `table6_lag_orders.*` |
| Lag-by-lag bootstrap p-values | `table6b_pvalues.*` |
| Hyper-parameter record for the referee | `table7_hyperparameters.*` |
| Robustness of the p-value to the random directions | `table8_stability.*` |
| Data overview figure | `fig5_data_overview.{pdf,png}` |
| p-value heat map over lags and sub-samples | `fig6_pvalue_heatmap.*` |
| Lag profile of the p-value | `fig7_lagprofile_*.*` |
| Bootstrap null distribution with the observed statistic | `fig8_bootstrap_null.*` |
| The empirical process across the `L` directions | `fig9_empirical_process.*` |
| Mixture-density-network fit diagnostic | `fig10_mdn_fit.*` |
| Training curves | `fig11_training_curves.*` |
| Rolling-window causality | `fig12_rolling_*.*` |
| Stability of the p-value across direction draws | `fig13_stability.*` |

All tables come out simultaneously as `booktabs` LaTeX, GitHub Markdown and
CSV; all figures as vector PDF plus a 400-dpi PNG.

---

## 2. Installation and a 30-second check

```bash
pip install "drgct[data] @ git+https://github.com/merwanroudane/DRGCT.git"
```

or, from a clone:

```bash
git clone https://github.com/merwanroudane/DRGCT.git
cd DRGCT
pip install -e ".[data,dev]"
```

Check that everything is wired up:

```bash
drgct info
```

Now the 30-second scientific check — a series where you *know* the answer:

```python
import numpy as np
from drgct import drgc_test

rng = np.random.default_rng(0)
n = 600

# X is an AR(1). Y depends on X only through a sine wave: no linear
# Granger causality at all, but strong nonlinear causality.
x = np.zeros(n)
e = rng.normal(0, 0.7, n)
for t in range(1, n):
    x[t] = -0.5 * x[t - 1] + e[t]

y = np.zeros(n)
u = rng.normal(0, 0.7, n)
for t in range(1, n):
    y[t] = 0.5 * y[t - 1] + np.sin(-x[t - 1]) + u[t]

drgc_test(x, y, lag=1, seed=1).print()
```

You should see a p-value at or near zero. Now flip the direction — nothing
should be found:

```python
drgc_test(y, x, lag=1, seed=1, x_name="Y", y_name="X").print()
```

If both of those behave, your installation is sound.

---

## 3. The method in ten minutes

You do not need the proofs to use the test correctly, but you do need four
ideas. (`docs/THEORY.md` maps every equation of the paper to the exact line of
code that implements it.)

### 3.1 The null is a conditional moment restriction

Let `W_{t-1} = (X_{t-1},…,X_{t-p}, Y_{t-1},…,Y_{t-q})'` be everything you
condition on, and let `m(Y_{t-1}) = E[Y_t | Y_{t-1},…,Y_{t-q}]` be the
best forecast of `Y_t` using **only its own past**. "X does not Granger-cause
Y in mean" says that adding X's past does not change that forecast:

```
H0 :  E[ Y_t − m(Y_{t−1}) | W_{t−1} ] = 0   almost surely.          (2)
```

### 3.2 A conditional moment restriction becomes an unconditional one

By Stinchcombe and White (1998), conditioning can be replaced by weighting
with a *generically comprehensively revealing* family. The paper uses complex
exponentials, `φ(W, w) = exp(i·w′W)`, so

```
H0 :  E[ (Y_t − m(Y_{t−1})) · exp(i·w′W_{t−1}) ] = 0   for all w ∈ W.  (3)
```

Crucially, no direction `w` is privileged: if the null fails, it fails for
*almost every* `w`, so a finite random sample of directions still detects it.

### 3.3 The doubly robust twist

`m` is unknown, so you plug in an estimate `m̂` from a neural network. The
naive move — put `m̂` straight into (3) — fails: the estimation error enters at
first order and the empirical process does not converge to a Gaussian limit,
so the test has no valid null distribution. The paper documents this as the
type I error blowing *up* with `n` (empirical sizes of 0.151 at `n = 1000` and
0.321 at `n = 2000` where the truth is "no causality").

You can run the naive version here with `doubly_robust=False`. In this
implementation its size breaks in the *opposite* direction — 0.000 to 0.020 at
`n = 500` — because the in-sample least-squares residual is near-orthogonal to
functions of `Y_{t−1}`, which shrinks the naive statistic while leaving its
bootstrap null unchanged. Either way the naive test is invalid; see
`results/README.md` for the numbers and the argument.

The fix is to centre the exponential in `X` by its own conditional
expectation, `φ(ν | Y_{t−1}) = E[exp(i·ν′X_{t−1}) | Y_{t−1}]`:

```
H0 :  E[ (Y_t − m(Y_{t−1})) · e^{i μ′Y_{t−1}} · ( e^{i ν′X_{t−1}} − φ(ν | Y_{t−1}) ) ] = 0.   (6)
```

Proposition 1 of the paper proves (3) and (6) are equivalent. But now the bias
from estimation depends on the **product** of the two errors, `‖m̂ − m‖ ·
‖φ̂ − φ‖`, instead of on `‖m̂ − m‖` alone. Two estimators each converging
slower than `n^{-1/2}` can still have a product that vanishes faster than
`n^{-1/2}` — which is exactly what makes deep networks admissible inside a
classical testing framework. That is the "doubly robust" idea, in the
Robins–Rotnitzky–Zhao (1994) / Chernozhukov et al. (2018) sense.

### 3.4 Two networks, one statistic, a free bootstrap

- **MLP** → `m̂(Y_{t−1})`, trained by least squares on `{Y_t, Y_{t−1}}`.
- **Mixture density network (MDN)** → `f̂(x | y)`, trained by maximum
  likelihood on `{X_{t−1}, Y_{t−1}}`; draw `M` pseudo-samples from it and
  average `exp(i·ν′X*)` to get `φ̂(ν | Y_{t−1})`.

Draw `L` random pairs `(μ_ℓ, ν_ℓ)` uniformly from a compact box, form

```
Ŝ_n(μ_ℓ, ν_ℓ) = (n−q)^{-1/2} Σ_t (Y_t − m̂) e^{i μ_ℓ′Y_{t−1}} ( e^{i ν_ℓ′X_{t−1}} − φ̂ ),   (9)
KS_n = max_ℓ max( |Re Ŝ_n| , |Im Ŝ_n| ).                                                 (10)
```

Critical values come from a **multiplier bootstrap**: multiply each summand by
an independent `ξ_t` with mean 0, variance 1 and bounded support (Rademacher
`±1` by default) and recompute. Nothing is re-estimated — the networks are
trained **once per test**. This is why `B = 1000` costs milliseconds while the
two networks cost seconds.

### 3.5 What you gain, and what you give up

| | Linear Granger (VAR *F*-test) | Kernel/smoothing nonparametric (NHKJ class) | **DRGCT** |
|---|---|---|---|
| Detects nonlinear causality | ✗ | ✓ | ✓ |
| Usable at lag 5–10 | ✓ | ✗ (curse of dimensionality) | ✓ |
| Bandwidth to choose | – | yes, and results are sensitive to it | no |
| Size control at large lag | ✓ | poor — badly undersized (paper's Table 3) | ✓ |
| Cost per test | milliseconds | seconds | **seconds to tens of seconds** |
| Deterministic given data | ✓ | ✓ | ✗ — depends on the random `(μ, ν)` draw and network init |

That last row is the honest cost, and §5.8 tells you what to do about it.

---

## 4. Design decisions to make *before* you write code

Write these down in your notes; a referee will ask about all six.

**(a) Which series is the cause?** The test is directional. `drgc_test(x, y)`
tests "`x` Granger-causes `y`". Always run both directions; a one-directional
result is far more interesting than a bidirectional one, and a bidirectional
one often signals an omitted common driver.

**(b) What transformation makes them stationary?** Assumption 1 requires
strict stationarity and exponential β-mixing. Levels of prices, GDP or money
stock are not stationary. Use log-differences, percentage changes or growth
rates, and *check* with `check_stationarity`.

**(c) What is the maximum plausible lag?** The whole selling point of the
DRGCT is that you no longer need to guess. Section 5 of the paper puts it
plainly: "researchers can set the lag order to a sufficiently large value,
thereby encompassing all potential causal lags." For daily financial data,
10 is generous. For monthly macro data with policy lags, 12–18. Just remember
that every extra lag costs you one observation and widens the network.

**(d) Same lag on both series?** The paper's framework uses `p = q = lag`.
This package also accepts `p ≠ q`. If you have a strong prior (say monetary
policy acts with a 12-month lag but inflation is persistent at 6), use it:
`drgc_test(x, y, p=12, q=6)`.

**(e) Sub-samples.** Structural breaks are the single most common reason a
causality result does not replicate. Split the sample the way the paper does
(overlapping windows) or use `rolling_causality` for a continuous picture.

**(f) Multiple testing.** If you run 10 lag orders × 2 directions × 3
sub-samples, you have run 60 tests at 5%. Three defensible options: (i) report
everything and let the *pattern* — not any single cell — carry the argument,
which is what the paper does; (ii) pre-register one headline lag; (iii) apply
a Holm or Benjamini–Hochberg correction within each direction. Say which one
you chose.

---

## 5. Step-by-step: from raw data to a reported test

We use the bundled S&P 500 / CSI 300 / Nikkei 225 data so that the code runs
for you verbatim. Substitute your own series at §5.1 and everything downstream
is unchanged.

### 5.1 Load and align the data

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from drgct.datasets import load_index, available_datasets

print(available_datasets())          # ['csi300', 'nikkei225', 'spx500']

spx = load_index("spx500")           # Date-indexed, columns Close and Volume
print(spx.head())
print(spx.shape)                     # (1257, 2)  --  the paper's T = 1257
```

**Using your own data.** Any two aligned NumPy arrays or pandas Series work.
The two rules that matter:

1. **Same length, same calendar.** Merge on the date index with an inner
   join *before* differencing, never after.
2. **No gaps you have silently filled.** Forward-filling a holiday creates
   an artificial zero return that the test will happily interpret as
   structure.

```python
# Example: your own CSV
raw = pd.read_csv("my_data.csv", index_col=0, parse_dates=True)
panel = raw[["gdp_growth", "credit_growth"]].dropna()      # inner join + drop NA
x = panel["credit_growth"].to_numpy()
y = panel["gdp_growth"].to_numpy()
```

### 5.2 Make the series stationary

For the price–volume application, Section 5 of the paper prescribes
percentage changes, with the volume change divided by 10 so the two series
have comparable scale:

```python
from drgct.datasets import to_percentage_changes
from drgct.utils import check_stationarity

pv = to_percentage_changes(spx)      # columns P and V
P = pv["P"].to_numpy()
V = pv["V"].to_numpy()

for name, s in (("P_t", P), ("V_t", V)):
    print(check_stationarity(s, name)["message"])
```

```
P_t: stationary (ADF p=0.000, KPSS p=0.100).
V_t: stationary (ADF p=0.000, KPSS p=0.100).
```

`check_stationarity` only returns `stationary=True` when ADF *rejects* the
unit root **and** KPSS *fails to reject* stationarity. If either fails,
difference again or rethink the transformation before going further — the
asymptotics in Section 3 of the paper simply do not apply to an integrated
series.

> **A note on scale.** `drgct` z-scores the inputs internally
> (`standardize=True`). This does not change the p-value — rescaling `Y`
> multiplies `KS_n` and every bootstrap replicate by the same constant, so
> the rank is unchanged — but it makes the networks train far more reliably.
> Set `standardize=False` only if you are deliberately experimenting.

### 5.3 Descriptive statistics

Referees expect the standard block: moments, normality, serial correlation in
the level and in the square, and unit-root tests.

```python
from drgct.tables import export_table, table_descriptives

desc = table_descriptives({"SPX 500  $P_t$": P, "SPX 500  $V_t$": V})
print(desc.to_markdown(index=False))

export_table(
    desc, "table0_descriptives", "results/tables",
    caption="Descriptive statistics of the transformed series",
    label="tab:descriptives",
    notes="Ljung--Box statistics use 10 lags. ADF and KPSS report p-values.",
)
```

### 5.4 Choose the hyper-parameters

Six numbers control the test. Here is what each one does, the paper's value,
and when to move it.

| Symbol | Argument | Paper's value | What it controls | Move it when |
|---|---|---|---|---|
| `G` | `G=10` | 10 | Mixture components of the MDN | Too small → MDN bias → **inflated size**. Too large → variance of `KS_n`. Cross-validate if `p ≥ 5`. |
| `L` | `L=20` | 20 | Number of random directions `(μ_ℓ, ν_ℓ)` | Raise it (50–100) when `p + q ≥ 10`; power rises, and simulation noise in the p-value falls. Cost is linear. |
| `M` | `M=20` | 20 | Pseudo-samples per observation from the MDN | The paper reports insensitivity; 20 is fine. Raise to 50 if `p ≥ 5`. |
| `B` | `B=1000` | 1000 | Bootstrap replications | Essentially free. Use ≥ 999 for a publishable p-value. |
| — | `w_lower`, `w_upper` | `[-1, 1]` (implied) | The compact set `W` from which directions are drawn | Widen to `[-2, 2]` to probe finer oscillations, narrow to `[-0.5, 0.5]` for smoother alternatives. Report what you used. |
| `H_n, L_n` | `mlp=`, `mdn=` | width `5·lag`, depth 1 | Network architecture | Depth 2 and width `10·lag` if `n > 2000`. Use `width="theory"` for the Lemma-1 rate. |

The defaults *are* the paper's values, so this is a no-op:

```python
from drgct.nets import MLPConfig, MDNConfig

settings = dict(
    G=10, L=20, M=20, B=1000,
    mlp=MLPConfig(width="paper", depth="paper", loss="l2", epochs=400, lr=5e-3),
    mdn=MDNConfig(width="paper", depth="paper", epochs=500, lr=5e-3, min_sigma=1e-2),
)
```

A deliberately richer configuration for a large sample:

```python
settings_rich = dict(
    G=15, L=60, M=50, B=1999,
    mlp=MLPConfig(width=64, depth=2, epochs=800, lr=3e-3, weight_decay=1e-5),
    mdn=MDNConfig(width=64, depth=2, epochs=900, lr=3e-3),
)
```

### 5.5 Your first test

```python
from drgct import drgc_test

res = drgc_test(
    P, V,                     # does the price change cause the volume change?
    lag=5,                    # p = q = 5
    G=10, L=20, M=20, B=999,
    alpha=0.05,
    seed=20240926,            # always set this
    x_name="P_t", y_name="V_t",
)
res.print()
```

### 5.6 Read the output line by line

```
========================================================================
  Doubly Robust Granger Causality Test (DRGCT)
  Hui, Liu & Song (2025), arXiv:2509.15798 -- Algorithm 1
========================================================================
  H0 : P_t does not Granger-cause V_t in mean
  Construction        : doubly robust
  Lag orders          : p = 5 (cause), q = 5 (effect)
  Sample              : n = 1256,  effective n - q = 1251
  MDN components G    : 10
  (mu, nu) pairs L    : 20   drawn U[-1.0, 1.0]
  Pseudo-samples M    : 20
  Bootstrap B         : 999   multipliers: rademacher
  Network (MLP/MDN)   : width 25/25, depth 1/1
------------------------------------------------------------------------
  KS_n statistic      : 1.108100
  Bootstrap p-value   : 0.4344
  Critical values     : 1%: 1.7350  5%: 1.5164  10%: 1.4013
  Decision at 5%      : DO NOT REJECT H0
------------------------------------------------------------------------
```

- **`KS_n`** is *not* comparable across specifications — it has no fixed
  scale. Only compare it to the bootstrap critical values printed beneath it.
- **`Bootstrap p-value`** is `p*_n = B^{-1} Σ_b 1{KS*_{n,b} ≥ KS_n}`. It can be
  exactly 0, meaning "smaller than `1/B`" — report it as `< 0.001`, not as 0.
- **`effective n − q`** is the sample the statistic actually uses. At lag 10
  on 750 observations you lose 10 — not a problem. At lag 10 on 60
  observations you have thrown away a sixth of your data.
- **`Critical values`** are quantiles of the bootstrap distribution and are
  what you would put in a table if you prefer critical values to p-values.

Everything is also available programmatically:

```python
res.pvalue, res.ks_stat, res.reject, res.stars
res.critical_values[0.05]
res.boot_stats          # (B,)  the bootstrap replicates
res.S_hat               # (L,)  complex Shat_n(mu_l, nu_l)
res.influence           # (n_eff, L) the summands z_{t,l}
res.residuals           # Y_t - mhat(Y_{t-1})
res.to_dict()           # flat record for a results table
res.to_frame()          # one-row DataFrame
```

### 5.7 Scan the lag orders, both directions

This is the core of the empirical exercise, and the thing the DRGCT does that
kernel methods cannot.

```python
from drgct import drgc_lag_scan

print("P_t -> V_t")
scan_pv, results_pv = drgc_lag_scan(
    P, V, lags=range(1, 11), B=999, seed=20240926,
    x_name="P_t", y_name="V_t",
)

print("V_t -> P_t")
scan_vp, results_vp = drgc_lag_scan(
    V, P, lags=range(1, 11), B=999, seed=20240927,
    x_name="V_t", y_name="P_t",
)
```

Each line is printed as it finishes, so a long scan tells you where it is:

```
P_t -> V_t
  lag  1 | KS =   0.5539 | p = 0.4665     |   3.2s
  lag  2 | KS =   0.8102 | p = 0.6096     |   3.9s
  ...
  lag 10 | KS =   1.3717 | p = 0.1231     |  17.4s
```

(Your numbers will differ: they depend on the seed, the sample window and the
`(μ, ν)` draw — see §5.8.)

`scan_pv` is a tidy DataFrame (`lag, ks_stat, pvalue, reject, cv_5, …`) ready
for `export_table`; `results_pv` is a `{lag: DRGCTResult}` dict if you need
the diagnostics for a particular lag.

**How to read a lag profile.** Three patterns recur:

- *Significant at every lag* — a robust, short-horizon effect. The lag-1
  result is doing the work; the rest inherit it.
- *Significant only from lag k onward* — the mechanism operates with a delay
  of at least `k` periods. This is the "long and variable lags" case, and it
  is exactly what a lag-1 test would have missed. Section 5 of the paper finds
  this for the S&P 500 in 2020–2023 (causality from lag 4) and the CSI 300 in
  2020–2023 (from lag 7).
- *Significant at scattered lags only* — be careful. With 10 tests at 5% you
  expect one false positive half the time. Lean on the stability check next.

### 5.8 Check that the result is not an artefact of the random directions

**Do not skip this.** Step 2(d) draws `L` directions at random. With `L = 20`
and `p + q = 20`, those directions cover the space thinly, so `KS_n` — and
hence the p-value — carries simulation noise on top of sampling noise. In the
bundled application, one draw put the S&P 500 lag-10 test at `p = 0.008` and
another at `p = 0.109`. Both are valid tests. Reporting only the first would
be misleading.

```python
from drgct import drgc_stability
from drgct.plots import plot_stability, save_figure

stab = drgc_stability(
    P, V, lag=10, n_draws=30, seed=20240926,
    G=10, L=20, M=20, B=999, x_name="P_t", y_name="V_t",
)

print(f"median p          : {stab['median']:.4f}")
print(f"5th–95th pct      : [{stab['q05']:.4f}, {stab['q95']:.4f}]")
print(f"share rejecting   : {stab['share_reject']:.1%}")
print(f"merged p-value    : {stab['merged_pvalue']:.4f}")

save_figure(plot_stability(stab, label="SPX 500, $P_t \\to V_t$"),
            "fig13_stability", "results/figures")
```

Three ways to act on what you see:

1. **Raise `L`.** Simulation noise falls as `L` grows, and power rises. Going
   from `L = 20` to `L = 100` costs about five times the (cheap) statistic
   evaluation and nothing at all in network training.
2. **Report the distribution.** A histogram of 30 p-values is more honest and
   more informative than one number.
3. **Use `merged_pvalue`.** It equals `min(1, 2 × median(p_r))`. By Rüger's
   inequality (see Vovk and Wang, 2020, on p-value merging), twice the median
   of arbitrarily dependent valid p-values is itself valid. It is
   conservative, so rejecting on it is a *stronger* claim than rejecting on
   any single draw. This is the number to put in a paper when a single-draw
   p-value sits near the boundary.

### 5.9 Sub-samples and rolling windows

Static sub-samples, exactly as in the paper:

```python
from drgct.datasets import PAPER_PERIODS, subsample

for name in PAPER_PERIODS:
    sub = subsample(pv, name)
    r = drgc_test(sub["P"].to_numpy(), sub["V"].to_numpy(),
                  lag=5, B=999, seed=1, x_name="P_t", y_name="V_t")
    print(f"{name}: n={len(sub):4d}  KS={r.ks_stat:6.3f}  p={r.pvalue:.4f} {r.stars}")
```

A continuous picture — the natural extension:

```python
from drgct.applications import rolling_causality
from drgct.plots import plot_rolling_pvalue

roll = rolling_causality(
    P, V, lag=5, window=750, step=21,     # ~1 month between windows
    dates=pv.index, drgc_kwargs=dict(B=999),
    x_name="P_t", y_name="V_t", n_jobs=-1, seed=7,
)
save_figure(plot_rolling_pvalue(roll, label="SPX 500: rolling DRGCT, window 750, lag 5"),
            "fig12_rolling_spx500", "results/figures")
```

Interpret the rolling plot as a *description of instability*, not as 25
independent tests: consecutive windows overlap by 97%, so their p-values are
strongly dependent. What matters is whether the series spends a sustained
stretch below the threshold, not whether it dips below once.

### 5.10 Benchmark against a smoothing-based test

Reviewers will ask "what does a conventional nonparametric test say?" The
package ships the smoothing-based benchmark used in Tables 3–4 of the paper.

```python
from drgct import nhkj_test

for lag in (1, 3, 5, 10):
    a = drgc_test(P, V, lag=lag, B=999, seed=1)
    b = nhkj_test(P, V, lag=lag)
    print(f"lag {lag:2d}   DRGC p = {a.pvalue:.4f}   NHKJ p = {b.pvalue:.4f}")
```

**How to read the comparison — carefully.** The paper's own Table 3 reports
NHKJ sizes between 0.003 and 0.043 at lag ≥ 2 against a 5% nominal level, and
Table 4 reports NHKJ power of 0.052 versus DRGC's 0.546 at lag 5 in DGP P2
with `n = 500`. On the paper's evidence, a *non-rejection* by NHKJ at lag ≥ 3
is weak evidence of anything.

Our own implementation behaves differently, and you should know that before
you quote it (`results/README.md` has the numbers): it is severely undersized
in the exponential-mean designs (0.000–0.005 in S2) and at high lag in S1, but
*over*-sized at lag 1 in S1 (0.170), and it keeps high power in P1–P3 rather
than collapsing. **Because its size is not controlled, its power numbers are
not comparable to the DRGCT's.** Use it as a reference point; when you want to
characterise NHKJ's properties in print, cite the paper's numbers, not ours.

> **Implementation note.** `nhkj_test` implements a Zheng (1996) /
> Fan–Li (1996) degenerate U-statistic test with the fourth-order Gaussian
> kernel and `h = c·n^{-0.15}` bandwidth schedule the DRGCT paper specifies
> for its benchmark. It is a member of the Nishiyama et al. (2011) class, not
> a line-by-line transcription of the original estimator, and its
> finite-sample behaviour is visibly sensitive to the bandwidth constant —
> which is itself the standard criticism of smoothing-based causality tests.
> Describe it as "a smoothing-based nonparametric benchmark" rather than as
> "the NHKJ test".

---

## 6. Building the tables

Every builder returns a plain DataFrame; `export_table` writes LaTeX,
Markdown and CSV at once.

```python
import pandas as pd
from drgct.tables import (
    export_table, table_detection, table_lag_orders,
    table_pvalues, table_hyperparameters, to_latex_booktabs,
)

# Put both directions into the tidy long format the builders expect.
long = pd.concat([
    scan_pv.assign(index_label="SPX 500", period="2019-2024", direction="P_t -> V_t"),
    scan_vp.assign(index_label="SPX 500", period="2019-2024", direction="V_t -> P_t"),
], ignore_index=True)

export_table(
    table_detection(long, alpha=0.05), "table5_detection", "results/tables",
    caption="Price--volume Granger causality detection",
    label="tab:detection",
    notes="A tick marks rejection at the 5\\% level for at least one lag order.",
)

export_table(
    table_lag_orders(long, alpha=0.05), "table6_lag_orders", "results/tables",
    caption="Granger causality under specific lag orders", label="tab:lagorders",
)

export_table(
    table_pvalues(long), "table6b_pvalues", "results/tables",
    caption="Bootstrap p-values by lag order", label="tab:pvalues",
    float_format="%.3f",
)

export_table(
    table_hyperparameters(res), "table7_hyperparameters", "results/tables",
    caption="Hyper-parameters of the reported DRGCT", label="tab:hyper",
)
```

The `.tex` files use `booktabs` and are ready to `\input{}`:

```latex
\usepackage{booktabs}
\usepackage{amssymb}   % for \checkmark in the detection tables
...
\input{results/tables/table6_lag_orders.tex}
```

Need a bespoke layout? `to_latex_booktabs` takes any DataFrame:

```python
tex = to_latex_booktabs(
    my_frame,
    caption="My table", label="tab:mine",
    notes="Standard errors in parentheses.",
    align="lcccc", float_format="%.3f",
)
open("results/tables/mine.tex", "w", encoding="utf-8").write(tex)
```

**The tidy long format.** Any DataFrame with columns
`index_label, period, direction, lag, pvalue` works with `table_detection`,
`table_lag_orders`, `table_pvalues` and `plot_pvalue_heatmap`. If your study
has no natural "index" or "period" dimension, fill them with a constant — the
builders will just produce a single row block. `lag_scan_frame` does this for
you.

---

## 7. Building the figures

```python
from drgct.plots import (
    use_journal_style, save_figure,
    plot_bootstrap_distribution, plot_empirical_process, plot_lag_profile,
    plot_pvalue_heatmap, plot_series_overview, plot_mdn_fit,
    plot_training_curves, plot_stability, plot_rolling_pvalue,
)

use_journal_style()      # serif, no top/right spines, 400-dpi PNG + vector PDF
FIG = "results/figures"

# 1. What the data look like
save_figure(plot_series_overview({"SPX 500": spx}, {"SPX 500": pv}),
            "fig5_data_overview", FIG)

# 2. The p-value across lags -- the headline result figure
save_figure(plot_lag_profile(scan_pv, alpha=0.05, label="SPX 500, $P_t \\to V_t$"),
            "fig7_lagprofile_p2v", FIG)

# 3. Lag x sub-sample heat map -- the graphical version of Table 6
save_figure(plot_pvalue_heatmap(long, alpha=0.05), "fig6_pvalue_heatmap", FIG)
```

For a *single* headline test, ask for the networks back and draw all four
diagnostics:

```python
res_diag = drgc_test(P, V, lag=5, B=999, seed=1,
                     x_name="P_t", y_name="V_t", return_networks=True)

save_figure(plot_bootstrap_distribution(res_diag), "fig8_bootstrap_null", FIG)
save_figure(plot_empirical_process(res_diag),      "fig9_empirical_process", FIG)
save_figure(plot_mdn_fit(res_diag),                "fig10_mdn_fit", FIG)
save_figure(plot_training_curves(res_diag),        "fig11_training_curves", FIG)
```

**What each diagnostic tells you.**

- `plot_bootstrap_distribution` — where `KS_n` falls in its resampled null.
  Put this next to any single reported test; it makes the p-value visually
  auditable.
- `plot_empirical_process` — `Re Ŝ_n` and `Im Ŝ_n` at each of the `L`
  directions, against the bootstrap envelope. Spikes outside the band show
  *which* direction in `W` drives the rejection. If exactly one of twenty
  directions escapes, treat the rejection with suspicion and raise `L`.
- `plot_mdn_fit` — the pooled MDN draws against the empirical marginal of
  `X_{t−1}`. If the draws are visibly too narrow or miss a mode, raise `G`.
  An MDN that fits badly inflates the type I error.
- `plot_training_curves` — MLP loss and MDN negative log-likelihood. A curve
  still falling steeply at the last epoch means you stopped too early; raise
  `epochs`.

All figures are Matplotlib objects, so you can post-process before saving:

```python
fig = plot_lag_profile(scan_pv)
ax = fig.axes[0]
ax.set_title("Panel A. S&P 500, prices to volumes", loc="left")
ax.axvspan(3.5, 6.5, color="0.9", zorder=0)          # highlight a lag range
save_figure(fig, "fig7_custom", FIG, formats=("pdf", "png", "svg"))
```

---

## 8. The complete application script

Everything above, in one file. Save it as `my_study.py` and run it.

```python
#!/usr/bin/env python
"""A complete DRGCT application study, from raw data to journal output."""

from __future__ import annotations

import pathlib
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from drgct import drgc_lag_scan, drgc_stability, drgc_test, nhkj_test
from drgct.datasets import PAPER_PERIODS, load_index, subsample, to_percentage_changes
from drgct.plots import (
    plot_bootstrap_distribution, plot_empirical_process, plot_lag_profile,
    plot_mdn_fit, plot_pvalue_heatmap, plot_series_overview, plot_stability,
    plot_training_curves, save_figure, use_journal_style,
)
from drgct.tables import (
    export_table, table_descriptives, table_detection,
    table_hyperparameters, table_lag_orders, table_pvalues,
)
from drgct.utils import check_stationarity

# ----------------------------------------------------------------------- #
# 0. Configuration -- change these six lines and nothing else
# ----------------------------------------------------------------------- #
OUT      = pathlib.Path("results")
TDIR     = OUT / "tables"
FDIR     = OUT / "figures"
LAGS     = range(1, 11)
SETTINGS = dict(G=10, L=20, M=20, B=999)
SEED     = 20240926
ALPHA    = 0.05

TDIR.mkdir(parents=True, exist_ok=True)
FDIR.mkdir(parents=True, exist_ok=True)
use_journal_style()

# ----------------------------------------------------------------------- #
# 1. Data
# ----------------------------------------------------------------------- #
raw = load_index("csi300")                 # <- swap in your own DataFrame
pv  = to_percentage_changes(raw)           # <- or your own transformation
X_NAME, Y_NAME = "P_t", "V_t"
x = pv["P"].to_numpy()
y = pv["V"].to_numpy()

print(f"Sample: {len(pv)} observations, "
      f"{pv.index.min().date()} to {pv.index.max().date()}")
for nm, s in ((X_NAME, x), (Y_NAME, y)):
    print("  " + check_stationarity(s, nm)["message"])

export_table(
    table_descriptives({f"CSI 300  ${X_NAME}$": x, f"CSI 300  ${Y_NAME}$": y}),
    "table0_descriptives", TDIR,
    caption="Descriptive statistics", label="tab:descriptives",
    notes="Ljung--Box statistics use 10 lags; ADF and KPSS report p-values.",
)
save_figure(plot_series_overview({"CSI 300": raw}, {"CSI 300": pv}),
            "fig5_data_overview", FDIR)

# ----------------------------------------------------------------------- #
# 2. Lag scans, both directions, in every sub-sample
# ----------------------------------------------------------------------- #
rows = []
for period in PAPER_PERIODS:
    sub = subsample(pv, period)
    a, b = sub["P"].to_numpy(), sub["V"].to_numpy()
    for direction, (cause, effect) in {
        f"{X_NAME} -> {Y_NAME}": (a, b),
        f"{Y_NAME} -> {X_NAME}": (b, a),
    }.items():
        print(f"\n{period}  {direction}   (n = {len(sub)})")
        scan, _ = drgc_lag_scan(
            cause, effect, lags=LAGS, seed=SEED, alpha=ALPHA, **SETTINGS
        )
        rows.append(scan.assign(index_label="CSI 300", period=period,
                                direction=direction))
long = pd.concat(rows, ignore_index=True)
long.to_csv(TDIR / "raw_results.csv", index=False)

# ----------------------------------------------------------------------- #
# 3. Tables
# ----------------------------------------------------------------------- #
export_table(table_detection(long, alpha=ALPHA), "table5_detection", TDIR,
             caption="Granger causality detection", label="tab:detection",
             notes=f"A tick marks rejection at the {100 * ALPHA:g}\\% level "
                   "for at least one lag order.")
export_table(table_lag_orders(long, alpha=ALPHA, lags=LAGS),
             "table6_lag_orders", TDIR,
             caption="Granger causality under specific lag orders",
             label="tab:lagorders")
export_table(table_pvalues(long, lags=LAGS), "table6b_pvalues", TDIR,
             caption="Bootstrap p-values by lag order", label="tab:pvalues",
             float_format="%.3f")

# ----------------------------------------------------------------------- #
# 4. Figures
# ----------------------------------------------------------------------- #
save_figure(plot_pvalue_heatmap(long, alpha=ALPHA), "fig6_pvalue_heatmap", FDIR)
for direction, tag in ((f"{X_NAME} -> {Y_NAME}", "x2y"),
                       (f"{Y_NAME} -> {X_NAME}", "y2x")):
    s = long[(long["direction"] == direction) & (long["period"] == "2021-2024")]
    save_figure(plot_lag_profile(s, alpha=ALPHA, label=f"CSI 300, 2021-2024, {direction}"),
                f"fig7_lagprofile_{tag}", FDIR)

# ----------------------------------------------------------------------- #
# 5. Headline specification: full diagnostics + stability + benchmark
# ----------------------------------------------------------------------- #
HEAD_LAG = 5
sub = subsample(pv, "2021-2024")
a, b = sub["P"].to_numpy(), sub["V"].to_numpy()

res = drgc_test(a, b, lag=HEAD_LAG, seed=SEED, alpha=ALPHA,
                x_name=X_NAME, y_name=Y_NAME, return_networks=True, **SETTINGS)
res.print()
(TDIR / "headline_summary.txt").write_text(res.summary(), encoding="utf-8")

export_table(table_hyperparameters(res), "table7_hyperparameters", TDIR,
             caption="Hyper-parameters of the reported test", label="tab:hyper")
save_figure(plot_bootstrap_distribution(res), "fig8_bootstrap_null", FDIR)
save_figure(plot_empirical_process(res),      "fig9_empirical_process", FDIR)
save_figure(plot_mdn_fit(res),                "fig10_mdn_fit", FDIR)
save_figure(plot_training_curves(res),        "fig11_training_curves", FDIR)

stab = drgc_stability(a, b, lag=HEAD_LAG, n_draws=30, seed=SEED,
                      alpha=ALPHA, x_name=X_NAME, y_name=Y_NAME, **SETTINGS)
print(f"\nStability over {stab['n_draws']} direction draws: "
      f"median p = {stab['median']:.4f}, merged p = {stab['merged_pvalue']:.4f}, "
      f"{stab['share_reject']:.0%} reject")
save_figure(plot_stability(stab, label="CSI 300, 2021-2024"), "fig13_stability", FDIR)

print("\nSmoothing-based benchmark:")
for lag in (1, 3, 5, 10):
    d = drgc_test(a, b, lag=lag, seed=SEED, **SETTINGS)
    k = nhkj_test(a, b, lag=lag)
    print(f"  lag {lag:2d}   DRGC p = {d.pvalue:.4f}   NHKJ p = {k.pvalue:.4f}")

print(f"\nAll output written to {OUT.resolve()}")
```

Run it:

```bash
python my_study.py
```

Or skip the script entirely and use the CLI:

```bash
drgct test my_data.csv -x credit_growth -y gdp_growth --lag-scan --lag-max 12 --save
```

---

## 9. Verifying size for *your* design with a bespoke Monte Carlo

The paper's Table 3 shows the test is correctly sized on *its* designs. If
your data look different — much shorter, heavier tailed, more persistent — it
costs an hour to check that the test is still correctly sized on a design that
resembles yours.

The honest way: simulate under the null with your data's own dynamics.

```python
import numpy as np
import statsmodels.api as sm
from drgct import drgc_test

# 1. Fit a simple own-past model to each series, so the simulated null
#    inherits your persistence and your innovation scale.
def ar_fit(s, lag):
    n = len(s)
    Z = np.column_stack([s[lag - j - 1:n - j - 1] for j in range(lag)])
    m = sm.OLS(s[lag:], sm.add_constant(Z)).fit()
    return m.params, np.std(m.resid)

LAG = 5
bx, sx = ar_fit(x, LAG)
by, sy = ar_fit(y, LAG)

# 2. Simulate independent AR processes -- the null holds by construction.
def sim_null(n, rng):
    burn = 500
    T = n + burn
    a = np.zeros(T); b = np.zeros(T)
    for t in range(LAG, T):
        a[t] = bx[0] + bx[1:] @ a[t - LAG:t][::-1] + rng.normal(0, sx)
        b[t] = by[0] + by[1:] @ b[t - LAG:t][::-1] + rng.normal(0, sy)
    return a[burn:], b[burn:]

# 3. Empirical size.
REPS = 200
rng = np.random.default_rng(0)
rejections = 0
for r in range(REPS):
    a, b = sim_null(len(x), rng)
    rejections += drgc_test(a, b, lag=LAG, B=499, seed=r).pvalue < 0.05
print(f"empirical size = {rejections / REPS:.3f}  "
      f"(MC s.e. = {np.sqrt(0.05 * 0.95 / REPS):.3f})")
```

If the size comes out well above 0.05, the usual culprits are (i) `G` too
small, so the MDN is biased, and (ii) too few epochs, so `m̂` is badly
under-fitted. Raise `G` to 15 and `epochs` to 800 and re-run.

To reproduce the paper's own designs instead:

```python
from drgct.simulate import monte_carlo, summarize
from drgct.tables import table_size, table_power

mc = monte_carlo(
    dgps=["S1", "S2", "P1", "P2", "P3", "P4"],
    ns=[500], lags=[1, 2, 3, 4, 5], reps=200,
    methods=("drgc", "drgc_naive", "nhkj"),
    drgc_kwargs=dict(B=499), n_jobs=-1, seed=20250915,
)
print(summarize(mc).to_string(index=False))
print(table_size(mc).to_markdown(index=False))
print(table_power(mc).to_markdown(index=False))
```

or, from the shell:

```bash
python scripts/run_simulation.py --reps 200 --ns 500 --jobs 10
```

---

## 10. Performance, scaling and reproducibility

### Where the time goes

Roughly 95% of a single test is spent training the two networks; the KS
statistic and the whole bootstrap take milliseconds. On a 2024-era laptop CPU:

| `n` | lag | Time per test |
|---|---|---|
| 500 | 1 | ~1.5 s |
| 500 | 5 | ~2 s |
| 750 | 10 | ~8–12 s |
| 2000 | 5 | ~6 s |

Consequences:

- **`B` is free.** Never economise on bootstrap replications. Use 999 or 1999.
- **`L` is nearly free.** It costs one extra column in a matrix product.
  Raising `L` from 20 to 100 is the cheapest power improvement available.
- **The lag order is expensive**, because it widens both networks
  (`H_n = 5·lag`) and enlarges the MDN output head (`G × p` means and scales).
- **Replications are the binding constraint.** A Monte Carlo is
  `reps × time_per_test`; parallelise with `n_jobs`.

### Parallelism

Every driver (`monte_carlo`, `price_volume_study`, `rolling_causality`,
`lag_scan_frame`) takes `n_jobs`; `-1` means `cpu_count() - 1`. Workers set
`torch.set_num_threads(1)`, because these networks are far too small for
intra-op threading to help — process-level parallelism wins outright.

On Windows and macOS the process pool re-imports your script, so **always**
guard the entry point:

```python
if __name__ == "__main__":
    main()
```

### Reproducibility

Pass `seed=` to every call. It seeds Python's `random`, NumPy and PyTorch, so
the network initialisation, the `(μ, ν)` draw, the MDN pseudo-samples and the
bootstrap multipliers are all fixed. Two runs with the same seed give
bit-identical results on the same machine and PyTorch version.

Across machines or PyTorch versions, floating-point non-determinism means the
p-value can move in the third decimal. Never build an argument on a p-value of
0.0499 — that is what §5.8 is for.

Record the environment with your results:

```python
import sys, torch, numpy, drgct
print(f"drgct {drgct.__version__} | python {sys.version.split()[0]} "
      f"| torch {torch.__version__} | numpy {numpy.__version__}")
```

---

## 11. Troubleshooting

**`ValueError: x contains NaN or inf`**
Clean before testing. `to_percentage_changes` already drops non-finite rows;
for your own transformation use
`np.isfinite(x) & np.isfinite(y)` as a joint mask so the two series stay
aligned.

**`ValueError: Effective sample size is 25 … at least 30 usable observations are required`**
`n − max(p, q)` is too small. Reduce the lag order or get more data. As a rule
of thumb you want at least 30 observations per lag before the networks have
anything to learn.

**The p-value is exactly 0.000**
It is `< 1/B`. Report it as `< 0.001` with `B = 999`, and raise `B` to 9999 if
you need a finer figure.

**The p-value swings between runs**
Expected — see §5.8. Raise `L`, run `drgc_stability`, report the merged
p-value.

**Empirical size is far above 5% in my own Monte Carlo**
In order of likelihood: `G` too small (MDN bias); too few epochs (`m̂`
under-fitted); `n` too small for lag `q` (fewer than ~30 observations per
lag); the simulated series is not actually stationary. Check
`plot_mdn_fit` and `plot_training_curves` first.

**The test never rejects, even on data I am sure are causal**
Check the direction of the arguments (`drgc_test(cause, effect)`, in that
order). Then raise `L` to 100, and check `plot_empirical_process` — if every
direction sits inside the envelope, the dependence really is weak at that lag.
Also confirm your transformation has not destroyed the signal: differencing
twice will annihilate most economic relationships.

**MDN training diverges (loss becomes `nan`)**
Raise `min_sigma` to `5e-2`; a component has collapsed onto a single point.
Gradient clipping at norm 5 is already on.

**It is too slow**
Reduce `epochs` to 200/250 for exploratory work and restore the defaults for
the final run; use `n_jobs=-1`; test a coarser lag grid
(`lags=[1, 2, 3, 5, 8, 10]`) before filling it in.

**`RuntimeError` about pickling on Windows**
You called a parallel driver without an `if __name__ == "__main__":` guard.

---

## 12. Reporting checklist and a write-up template

### Checklist

- [ ] Transformation used, and the ADF/KPSS evidence that it delivered
      stationarity.
- [ ] Sample period, `n`, and the effective `n − q` at your maximum lag.
- [ ] Lag orders tested, both directions, and whether `p = q`.
- [ ] `G`, `L`, `M`, `B`, the support of `(μ, ν)`, the multiplier
      distribution, and the network width/depth. `table_hyperparameters`
      produces this in one line of code.
- [ ] The random seed.
- [ ] The stability check: median p, the 5th–95th percentile range across
      direction draws, and the merged p-value.
- [ ] How you handled multiple testing across lags and sub-samples.
- [ ] A benchmark: at minimum a linear VAR *F*-test, ideally also
      `nhkj_test`, with a sentence noting that smoothing tests are badly
      undersized at lag ≥ 3.
- [ ] Software and version, and a link to your replication code.

### Method paragraph you can adapt

> We test for Granger causality in mean using the deep-learning based doubly
> robust test (DRGCT) of Hui, Liu and Song (2025). The null is
> `E[Y_t | W_{t−1}] = E[Y_t | Y_{t−1},…,Y_{t−q}]`, where
> `W_{t−1}` collects `p` lags of `X` and `q` lags of `Y`. The conditional mean
> is estimated with a ReLU multilayer perceptron and the conditional
> characteristic function of `X_{t−1}` given `Y_{t−1}` with a mixture density
> network with `G = 10` components; the resulting empirical process is doubly
> robust, so its bias depends on the product of the two estimation errors and
> the test attains a parametric rate despite the semiparametric first stage.
> We evaluate the process at `L = 20` random directions drawn uniformly from
> `[−1, 1]^{p+q}`, take the Kolmogorov–Smirnov functional of its real and
> imaginary parts, and obtain critical values from a multiplier bootstrap with
> `B = 999` Rademacher multipliers. Because the test does not suffer the curse
> of dimensionality that constrains kernel-smoothing causality tests, we scan
> lag orders 1 through 10 rather than pre-selecting one. All computations use
> the `drgct` Python package (Roudane, 2026), version 1.0.0, with seed
> 20240926; replication code is available at [URL].

### Results paragraph you can adapt

> Table 6 reports the outcome at each lag order. Causality from `P_t` to `V_t`
> is rejected at the 5% level from lag 4 onward in the 2020–2023 sub-sample and
> at every lag in 2021–2024, while the reverse direction is never significant.
> The pattern is robust to the random draw of directions: across 30
> independent draws the median p-value at lag 5 is 0.012, the 5th–95th
> percentile range is [0.004, 0.041], and the merged p-value of Vovk and Wang
> (2020) is 0.024. A linear VAR *F*-test at the same lag order fails to reject
> (p = 0.37), so the relationship is genuinely nonlinear.

### Bibliography entries

```bibtex
@article{HuiLiuSong2025DRGCT,
  title   = {Deep learning based doubly robust test for Granger causality},
  author  = {Hui, Yongchang and Liu, Chijin and Song, Xiaojun},
  journal = {arXiv preprint arXiv:2509.15798},
  year    = {2025},
  url     = {https://arxiv.org/abs/2509.15798}
}

@software{Roudane2026drgct,
  title   = {drgct: Deep-learning based doubly robust Granger causality
             testing in Python},
  author  = {Roudane, Merwan},
  version = {1.0.0},
  year    = {2026},
  url     = {https://github.com/merwanroudane/DRGCT}
}
```

`drgct.cite("bibtex")` prints both.

---

## 13. How the results shipped in this repository were produced

Everything in `results/` was generated by the two scripts in `scripts/`, on a
12-core laptop CPU, with the commands recorded in
[`results/README.md`](../results/README.md). Nothing in `results/` is
hand-edited.

Two honest caveats, stated up front:

1. **Simulation scale.** The paper uses 1000 Monte-Carlo replications and
   `B = 1000`. The shipped tables use fewer replications so that the run
   finishes in hours rather than days; the exact counts are recorded in
   `results/README.md` and in the note beneath every table. Monte-Carlo
   standard errors are reported alongside every rejection frequency, so you
   can see precisely how much precision was traded away. Pass
   `--reps 1000 --ns 500 1000 2000 -B 1000` to run the paper's grid.

2. **Data.** The paper does not name its data vendor. The bundled CSVs come
   from Yahoo! Finance over the paper's exact window (27 September 2019 to
   26 September 2024); the S&P 500 file has exactly the paper's `T = 1257`
   observations. For the CSI 300, Yahoo truncates the index series
   (`000300.SS`) to the last three years, so the exchange-traded tracker
   `510300.SS` is used instead — see `src/drgct/data/SOURCES.md`. Volume
   series in particular differ across vendors, so the application results
   agree with the paper's *qualitatively for two of the three indices* and
   differ for the third. `results/README.md` documents exactly where they
   agree and where they do not. Swap in your own vendor's CSV with the same
   two columns and every script runs unchanged.

---

## Where to go next

- [`docs/SYNTAX.md`](SYNTAX.md) — complete API reference: every function,
  every argument, every return field.
- [`docs/THEORY.md`](THEORY.md) — equation-by-equation map from the paper to
  the code, with the assumptions spelled out.
- [`docs/FAQ.md`](FAQ.md) — short answers to the questions that come up most.
- [`examples/`](../examples) — four runnable scripts, from a two-minute
  quick start to the full replication.
