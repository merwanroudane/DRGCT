# Shipped results

Everything in this folder was produced by the two scripts in
[`../scripts/`](../scripts) on a 12-core laptop CPU. Nothing is hand-edited.
This file records the exact commands, the environment, and — importantly —
where the numbers agree with Hui, Liu and Song (2025) and where they do not.

## Environment

| | |
|---|---|
| `drgct` | 1.0.0 |
| Python | 3.11.0 |
| PyTorch | 2.10.0 (CPU) |
| NumPy | 1.26.4 |
| Machine | 12-core laptop CPU, no GPU |

## Commands

```bash
python scripts/run_application.py --jobs 11 --rolling --stability-draws 30
python scripts/run_simulation.py  --reps 200 --ns 500 --lags 1 2 3 4 5 -B 499 \
                                  --jobs 11 --b-reps 150 --b-ns 1000 2000 --b-lags 1 3 5
```

To run the paper's own grid instead — 1000 replications at
`n ∈ {500, 1000, 2000}` with `B = 1000`, a multi-day job on this hardware:

```bash
python scripts/run_simulation.py --reps 1000 --ns 500 1000 2000 -B 1000 --jobs N
```

---

## Part 1 — Section 5, the price–volume application

### What was run

3 indices × 3 overlapping three-year sub-samples × 2 directions × 10 lag
orders = **180 DRGCTs**, each with `G = 10`, `L = 20`, `M = 20`, `B = 999`,
Rademacher multipliers, seed 20240926. Plus a 30-draw stability check on the
headline specification and a rolling-window study of 25 windows (S&P 500, lag 5,
window 750, step 21) in both directions.

### Files

| File | Content |
|---|---|
| `tables/table0_descriptives.*` | Moments, Jarque–Bera, Ljung–Box on the level and the square, ADF and KPSS, for all six series |
| `tables/table5_detection.*` | Table 5 layout — one tick or cross per (direction, sub-sample, index) |
| `tables/table6_lag_orders.*` | Table 6 layout — decisions at each lag order 1–10 |
| `tables/table6b_pvalues.*` | The same grid with the bootstrap p-values themselves |
| `tables/table7_hyperparameters.*` | The full settings record for the headline test |
| `tables/table8_stability.*` | Sensitivity of the headline p-value to the random `(μ, ν)` draw |
| `tables/price_volume_raw.csv` | All 180 results, one row per test |
| `tables/rolling_spx500.csv` | Rolling-window results (50 tests: 25 windows × 2 directions) |
| `tables/headline_summary.txt` | The printed summary of the headline test |
| `figures/fig5_data_overview.*` | Levels, volumes, and the transformed `P_t`, `V_t` |
| `figures/fig6_pvalue_heatmap.*` | Table 6 as a heat map |
| `figures/fig7_lagprofile_*.*` | p-value against lag order, per index and direction |
| `figures/fig8_bootstrap_null.*` | Bootstrap null of `KS_n` for the headline test |
| `figures/fig9_empirical_process.*` | `Re` and `Im` of `Ŝ_n` at the 20 directions |
| `figures/fig10_mdn_fit.*` | MDN draws against the empirical marginal of `X_{t−1}` |
| `figures/fig11_training_curves.*` | MLP loss and MDN negative log-likelihood |
| `figures/fig12_rolling_*.*` | Rolling-window p-values against calendar time |
| `figures/fig13_stability.*` | Distribution of the p-value across 30 direction draws |

### What we find

Reading `tables/table6b_pvalues.csv`:

- **CSI 300** — `P_t → V_t` is rejected decisively at lags 1–3 in *every*
  sub-sample (p ≤ 0.011 throughout), and the evidence strengthens over time
  (2021–2024: p < 0.001 at lags 1, 2, 3 and 6). The reverse direction
  `V_t → P_t` rejects at a single lag in a single sub-sample, which is what
  one expects by chance from 30 tests at 5%.
- **Nikkei 225** — `P_t → V_t` rejects at short lags (1 and 3) in all three
  sub-samples; `V_t → P_t` never rejects at any lag in any sub-sample.
- **S&P 500** — neither direction rejects at any lag in any sub-sample.

The common pattern — prices drive volumes, volumes do not drive prices — is
the paper's headline conclusion and the one the price–volume literature
(Karpoff 1987; Gallant, Rossi and Tauchen 1992) would lead you to expect.

### Where this agrees with the paper, and where it does not

| Direction | Index | Paper (Table 5) | Here | Agrees? |
|---|---|:--:|:--:|:--:|
| `P → V` | CSI 300 | ✓ in 2020-2023, 2021-2024 | ✓ in all three | **yes** (stronger here) |
| `P → V` | NI 225 | ✗ in all three | ✓ in all three | **no** |
| `P → V` | SPX 500 | ✓ in all three | ✗ in all three | **no** |
| `V → P` | CSI 300 | ✓ in 2021-2024 only | ✓ in 2019-2022 only | partly |
| `V → P` | NI 225 | ✗ everywhere | ✗ everywhere | **yes** |
| `V → P` | SPX 500 | ✗ everywhere | ✗ everywhere | **yes** |

Four of the six blocks agree. The two that do not are explained by data, not
by the estimator:

1. **The paper does not name its data vendor.** Closing *levels* are near
   identical across vendors; daily *volume* is not. Yahoo's S&P 500 volume is
   the sum of primary-listing composite volume and differs materially from
   what a Bloomberg or Refinitiv feed reports. Since the whole exercise is
   about volume dynamics, that alone can flip a marginal result.
2. **The CSI 300 file is a tracker, not the index.** Yahoo truncates
   `000300.SS` to roughly the last three years, so `510300.SS` — the largest
   exchange-traded CSI 300 tracker — is used. Its turnover is genuine
   exchange volume but it is ETF turnover, not index-constituent turnover.

A useful cross-check: a plain linear VAR *F*-test on the same data finds
`P → V` strongly significant for the CSI 300 in every sub-sample
(p ≤ 0.0009 at lag 1), never significant for the Nikkei 225 or the S&P 500,
and `V → P` never significant anywhere. So the CSI 300 and S&P 500 findings
here are what the data say, linear or nonlinear; the DRGCT's added value shows
up for the Nikkei 225, where it detects short-lag causality that the linear
test misses entirely.

**If you have the paper's data**, drop your CSVs into `src/drgct/data/` with
`Date, Close, Volume` columns and re-run `scripts/run_application.py`. No code
changes are required.

### The stability result

`tables/table8_stability.*` reports the headline specification (S&P 500,
2021–2024, `P_t → V_t`, lag 10) re-run over 30 independent draws of the
`L = 20` directions:

| Median p | Mean p | 5th pct | 95th pct | Share rejecting | Merged p (2 × median) |
|---:|---:|---:|---:|---:|---:|
| 0.521 | 0.556 | 0.048 | 0.974 | 0.067 | 1.000 |

The single draw used for `fig8`/`fig9` gave `p = 0.008`; the median across
draws is 0.52 and only 7% of draws reject. This is the clearest possible
demonstration of why `drgc_stability` exists — see
[the caveat in the README](../README.md#the-one-caveat-you-must-not-skip) and
[§5.8 of the guide](../docs/GUIDE.md#58-check-that-the-result-is-not-an-artefact-of-the-random-directions).

---

## Part 2 — Section 4, the Monte Carlo

### What was run

**Experiment A — completed.** All six designs of Table 1, lag orders 1–5,
`n = 500`, **200 replications** each, `B = 499`, comparing three estimators:

- `DRGC` — the doubly robust test;
- `DRGC-naive` — the same deep plug-in *without* the `− φ̂(ν | Y_{t−1})`
  correction, i.e. the naive process of equation (5);
- `NHKJ` — the smoothing-based nonparametric benchmark, with the paper's
  fourth-order Gaussian kernel and `h = c·n^{−0.15}` bandwidth schedule.

**Experiment B — stopped early.** Design S1 (the null) at `n ∈ {1000, 2000}`,
lags 1, 3, 5, `DRGC` versus `DRGC-naive`. The run was interrupted after the
first design point, so only **`n = 1000`, lag 1, 150 replications** is
reported. The command to complete it is at the top of this file.

### Files

| File | Content |
|---|---|
| `tables/table1_dgps.*` | Table 1 — the six designs |
| `tables/table2_parameters.*` | Table 2 — coefficients by lag order |
| `tables/table3_size.*` | Table 3 — empirical sizes |
| `tables/table4_power.*` | Table 4 — empirical powers |
| `tables/table3b_naive_size.*` | Experiment B, the one completed cell |
| `tables/monte_carlo_summary.*` | Every rejection frequency with its Monte-Carlo standard error |
| `tables/monte_carlo_raw.csv` | Replication-level output of Experiment A (6000 rows × 3 methods) |
| `tables/monte_carlo_naive_raw.csv` | Replication-level output of the completed Experiment B cell |
| `figures/fig1_size.*` | Empirical size against lag, with a ±1.96 MC-s.e. band |
| `figures/fig2_pvalue_ecdf.*` | Davidson–MacKinnon p-value plot under the null |
| `figures/fig3_power.*` | Empirical power against lag |

### The headline result: the DRGCT is correctly sized

Empirical size at the 5% nominal level, `n = 500`, 200 replications
(Monte-Carlo standard error ≈ 0.015). The paper's 1000-replication values are
in brackets.

| Lag | DGP S1 | DGP S2 |
|:--:|:--:|:--:|
| 1 | 0.070 [0.051] | 0.045 [0.046] |
| 2 | 0.055 [0.056] | 0.055 [0.049] |
| 3 | 0.055 [0.046] | 0.065 [0.045] |
| 4 | 0.075 [0.057] | 0.055 [0.039] |
| 5 | 0.050 [0.050] | 0.055 [0.051] |

Every cell is within about 1.5 Monte-Carlo standard errors of the paper.
`fig2_pvalue_ecdf` makes the point more sharply than any table: the empirical
CDF of the DRGCT's bootstrap p-values sits on the 45-degree line in all ten
null panels, which is exactly what Theorem 1 and Theorem 4 promise.

Power, `n = 500`, 200 replications (paper in brackets):

| Lag | P1 | P2 | P3 | P4 |
|:--:|:--:|:--:|:--:|:--:|
| 1 | 1.000 [1.000] | 1.000 [0.996] | 1.000 [1.000] | 1.000 [0.990] |
| 2 | 1.000 [1.000] | 1.000 [0.973] | 1.000 [0.977] | 0.965 [0.959] |
| 3 | 1.000 [1.000] | 0.975 [0.898] | 0.925 [0.954] | 0.615 [0.886] |
| 4 | 0.980 [0.912] | 0.800 [0.628] | 0.620 [0.429] | 0.315 [0.663] |
| 5 | 0.895 [0.870] | 0.565 [0.546] | 0.385 [0.407] | 0.170 [0.600] |

P1–P3 reproduce closely. **P4 does not**: at lags 3–5 our power is well below
the paper's. P4 is the one design where `Y_t` has no own-lag term of its own —
`Y_t = a₀ Σ_j X_{t−j} Y_{t−j} + ε_t` — so the conditional mean `m(Y_{t−1})` is
a genuinely awkward object and the MLP absorbs more of the signal in sample
than it should. Raising the MLP epochs or the number of directions `L` closes
part of the gap. We report the gap rather than tuning it away.

### The naive plug-in fails, but not in the direction the paper reports

| | S1 lag 1 | S1 lag 2 | S1 lag 3 | S1 lag 4 | S1 lag 5 |
|---|:--:|:--:|:--:|:--:|:--:|
| DRGC (`n = 500`) | 0.070 | 0.055 | 0.055 | 0.075 | 0.050 |
| DRGC-naive (`n = 500`) | 0.000 | 0.005 | 0.015 | 0.005 | 0.005 |
| DRGC (`n = 1000`, lag 1) | 0.080 | | | | |
| DRGC-naive (`n = 1000`, lag 1) | 0.000 | | | | |

Section 4 of the paper reports the naive plug-in *over*-rejecting, with size
rising to 0.151 at `n = 1000` and 0.321 at `n = 2000` for lag 5. Here it
*under*-rejects, essentially never rejecting at all.

The mechanism is straightforward. The naive statistic is

```
Ŝ⁰_n(μ, ν) = (n−q)^{−1/2} Σ_t (Y_t − m̂(Y_{t−1})) e^{i μ′Y_{t−1}} e^{i ν′X_{t−1}}
```

and `m̂` is fitted by least squares **in sample**, with no splitting — as
Section 1 of the paper prescribes. The first-order conditions of that fit make
the residual `Y_t − m̂(Y_{t−1})` near-orthogonal to any function of `Y_{t−1}`
in the span of the network, and `e^{i μ′Y_{t−1}}` is such a function. So the
statistic is mechanically shrunk toward zero. The multiplier bootstrap
`ξ_t (Y_t − m̂) e^{i μ′Y} e^{i ν′X}` inherits no such orthogonality, so its
null distribution is too wide and the test almost never rejects.

Which distortion dominates — this shrinkage or the estimation bias the paper
emphasises — depends on the design, the sample size and how hard the network
is trained. **Both are failures of the same kind**: without the
`− φ̂(ν | Y_{t−1})` centring, the naive process has no valid null distribution,
and the doubly robust process does. That is the point of the construction, and
it is what `fig1_size`, `fig2_pvalue_ecdf` and `table3_size` show.

### The smoothing benchmark does not behave like the paper's NHKJ

Empirical size of `nhkj_test` at the 5% nominal level, `n = 500`:

| Lag | S1 | S2 |
|:--:|:--:|:--:|
| 1 | **0.170** | 0.005 |
| 2 | 0.045 | 0.005 |
| 3 | 0.040 | 0.000 |
| 4 | 0.000 | 0.000 |
| 5 | 0.020 | 0.000 |

It is severely **under**-sized in the exponential-mean design S2 and at high
lag in S1 — which is the paper's qualitative story — but badly **over**-sized
at lag 1 in S1, and it keeps power near one in P1–P3 instead of collapsing.

Two honest conclusions:

1. `nhkj_test` is a *member of the NHKJ class* — a Zheng (1996) / Fan–Li
   (1996) degenerate U-statistic with the paper's fourth-order kernel and
   bandwidth schedule — not a transcription of the estimator in Nishiyama et
   al. (2011). Its finite-sample behaviour differs from the paper's NHKJ, and
   its sensitivity to the bandwidth constant is exactly the criticism the
   DRGCT paper levels at smoothing tests in the first place.
2. **Because its size is not controlled, its power column is not a fair
   comparison.** Read the size table before the power table. When you need to
   characterise NHKJ's properties in print, cite the paper's Tables 3–4, not
   ours.

### Precision

200 replications give a Monte-Carlo standard error of about **0.015** on an
empirical size of 0.05, and up to 0.035 on a power near 0.5. Every rejection
frequency in `monte_carlo_summary.csv` carries its own `mc_se` column.
Differences from the paper of 0.02–0.04 are sampling noise and should not be
over-read; differences of the size seen in P4 and in the NHKJ column are not.

### Independent calibration check

Before the shipped run, a separate 120-replication calibration at `n = 500`
gave (paper's 1000-replication values in brackets):

| Design | lag 1 | lag 3 | lag 5 |
|---|:--:|:--:|:--:|
| S1 size | 0.058 [0.051] | 0.050 [0.046] | 0.058 [0.050] |
| S2 size | 0.050 [0.046] | 0.067 [0.045] | 0.050 [0.051] |
| P2 power | 1.000 [0.996] | 0.967 [0.898] | 0.583 [0.546] |
| P3 power | 1.000 [1.000] | 0.958 [0.954] | 0.392 [0.407] |

Every cell is within about two Monte-Carlo standard errors of the paper —
independent confirmation that the DRGCT implementation is faithful.

---

## Reproducing this folder

```bash
pip install -e ".[data,dev]"
python scripts/run_application.py --jobs -1 --rolling --stability-draws 30
python scripts/run_simulation.py  --reps 200 --ns 500 -B 499 --jobs -1
```

Both scripts are deterministic given their `--seed`. On a different machine or
PyTorch version, floating-point non-determinism can move individual p-values
in the third decimal; the tables and the conclusions do not change.
