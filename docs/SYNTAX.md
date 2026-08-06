# `drgct` — Complete Syntax Reference

Every public function, every argument, every return field. Version 1.0.0
(`pip install drgct` — [PyPI](https://pypi.org/project/drgct/)).

For the *how-to*, read [`GUIDE.md`](GUIDE.md). For the *why*, read
[`THEORY.md`](THEORY.md).

---

## Contents

- [Import surface](#import-surface)
- [1. The test — `drgct.core`](#1-the-test--drgctcore)
  - [`drgc_test`](#drgc_test)
  - [`DRGCTResult`](#drgctresult)
  - [`drgc_lag_scan`](#drgc_lag_scan)
  - [`drgc_both_directions`](#drgc_both_directions)
  - [`drgc_stability`](#drgc_stability)
- [2. Networks — `drgct.nets`](#2-networks--drgctnets)
  - [`MLPConfig`](#mlpconfig) · [`MDNConfig`](#mdnconfig)
  - [`MLP`](#mlp) · [`MixtureDensityNetwork`](#mixturedensitynetwork)
  - [`fit_conditional_mean`](#fit_conditional_mean) · [`fit_conditional_density`](#fit_conditional_density)
  - [`paper_width`, `theory_width`, `theory_depth`](#architecture-helpers)
- [3. Benchmark — `drgct.nhkj`](#3-benchmark--drgctnhkj)
- [4. Designs — `drgct.dgp`](#4-designs--drgctdgp)
- [5. Monte Carlo — `drgct.simulate`](#5-monte-carlo--drgctsimulate)
- [6. Data — `drgct.datasets`](#6-data--drgctdatasets)
- [7. Applications — `drgct.applications`](#7-applications--drgctapplications)
- [8. Tables — `drgct.tables`](#8-tables--drgcttables)
- [9. Figures — `drgct.plots`](#9-figures--drgctplots)
- [10. Utilities — `drgct.utils`](#10-utilities--drgctutils)
- [11. Command line](#11-command-line)
- [12. Argument cheat sheet](#12-argument-cheat-sheet)

---

## Import surface

```python
from drgct import (
    # the test
    drgc_test, drgc_lag_scan, drgc_both_directions, drgc_stability, DRGCTResult,
    # benchmark
    nhkj_test, NHKJResult,
    # simulation
    simulate_dgp, monte_carlo, summarize, size_power_tables,
    DGP_NAMES, SIZE_DGPS, POWER_DGPS,
    # data
    load_index, load_all, available_datasets, to_percentage_changes,
    subsample, describe, INDEX_KEYS, INDEX_LABELS, PAPER_PERIODS,
    # applications
    price_volume_study, rolling_causality, lag_scan_frame, DIRECTIONS,
    # configuration and utilities
    MLPConfig, MDNConfig, set_seed, check_stationarity, build_lag_design,
    # submodules
    tables, plots,
)
import drgct
drgct.__version__          # '1.0.0'
drgct.PAPER                # metadata of the source article
drgct.cite()               # citation text; cite('bibtex') for BibTeX
```

---

## 1. The test — `drgct.core`

### `drgc_test`

```python
drgc_test(
    x, y, lag=None, *,
    p=None, q=None,
    G=10, L=20, M=20, B=1000,
    alpha=0.05,
    w_lower=-1.0, w_upper=1.0,
    multiplier="rademacher",
    doubly_robust=True,
    standardize=True,
    mlp=None, mdn=None,
    seed=None,
    alphas=(0.10, 0.05, 0.01),
    x_name="X", y_name="Y",
    return_networks=False,
) -> DRGCTResult
```

Tests `H0 : X does not Granger-cause Y in mean`. Implements Steps 1–5 of
Algorithm 1 of Hui, Liu and Song (2025).

| Argument | Type | Default | Description |
|---|---|---|---|
| `x` | array_like | — | Candidate **cause** `X_t`. 1-D, finite, ≥ 10 observations. Accepts list, ndarray, `pandas.Series`. |
| `y` | array_like | — | **Effect** `Y_t`, same length as `x`, aligned in calendar time. |
| `lag` | int | `None` | Common lag order, sets `p = q = lag`. Either `lag` or both `p` and `q` are required. |
| `p` | int | `None` | Lags of `X` entering `W_{t−1}`. |
| `q` | int | `None` | Lags of `Y` entering `W_{t−1}`. The paper maintains `p ≤ q`; `p > q` is allowed and starts the effective sample at `max(p,q)+1`. |
| `G` | int | `10` | Mixture components of the MDN. Too small → bias → size distortion; too large → variance. |
| `L` | int | `20` | Number of random pairs `(μ_ℓ, ν_ℓ)`. Higher → more power, more cost (linear), less simulation noise. |
| `M` | int | `20` | Pseudo-samples drawn per observation from the fitted conditional density. |
| `B` | int | `1000` | Multiplier-bootstrap replications. Essentially free — no re-estimation. |
| `alpha` | float | `0.05` | Headline level driving `.reject`. |
| `w_lower`, `w_upper` | float | `-1.0`, `1.0` | Bounds of the uniform box `W = W₁ × W₂` from which directions are drawn. |
| `multiplier` | str | `"rademacher"` | `ξ_t` distribution: `"rademacher"` (±1), `"mammen"`, `"normal"`. The first two have bounded support, as Theorem 4 requires. |
| `doubly_robust` | bool | `True` | `False` builds the naive process (5), i.e. drops `φ̂(ν \| Y_{t−1})`. Provided to reproduce the size failure of Section 4. **Never use for inference.** |
| `standardize` | bool | `True` | z-score inputs. Leaves the p-value invariant; greatly stabilises training. |
| `mlp` | `MLPConfig` or dict | `None` | Conditional-mean network hyper-parameters. A dict is splatted into `MLPConfig`. |
| `mdn` | `MDNConfig` or dict | `None` | Density network hyper-parameters. `G` overrides `mdn.n_components`. |
| `seed` | int | `None` | Master seed: network init, `(μ,ν)` draw, MDN samples, bootstrap multipliers. |
| `alphas` | sequence | `(0.10, 0.05, 0.01)` | Levels at which bootstrap critical values are reported. |
| `x_name`, `y_name` | str | `"X"`, `"Y"` | Labels used in `.summary()` and `.direction`. |
| `return_networks` | bool | `False` | Attach the fitted models and standardised inputs to `result.settings["networks"]`. Required by `plot_mdn_fit`. |

**Raises** `ValueError` if neither `lag` nor `(p, q)` is given; if `x` and `y`
have different lengths; if either contains NaN/inf; if `n − max(p,q) < 30`.

**Examples**

```python
# minimal
res = drgc_test(x, y, lag=3)

# fully specified, reproducible, publication settings
res = drgc_test(
    x, y, p=5, q=8,
    G=15, L=100, M=50, B=1999,
    w_lower=-2.0, w_upper=2.0,
    multiplier="mammen",
    mlp=MLPConfig(width=64, depth=2, epochs=800, lr=3e-3),
    mdn=MDNConfig(width=64, depth=2, epochs=900, min_sigma=2e-2),
    seed=42, x_name="oil", y_name="cpi",
    return_networks=True,
)

# reproduce the paper's naive-plug-in failure
naive = drgc_test(x, y, lag=5, doubly_robust=False, seed=42)
```

---

### `DRGCTResult`

Dataclass returned by `drgc_test`.

**Scalar fields**

| Field | Type | Meaning |
|---|---|---|
| `ks_stat` | float | `KS_n` of equation (10). Scale is specification-dependent — compare only to `critical_values`. |
| `pvalue` | float | `p*_n = B⁻¹ Σ_b 1{KS*_{n,b} ≥ KS_n}`. A value of 0 means `< 1/B`. |
| `reject` | bool | `pvalue < alpha`. |
| `alpha` | float | Headline level. |
| `p`, `q`, `lag` | int | Lag orders used; `lag = max(p, q)`. |
| `n`, `n_eff` | int | Original length and `n − max(p,q)`. |
| `direction` | str | e.g. `"P_t -> V_t"`. |
| `doubly_robust` | bool | Which process was built. |
| `elapsed` | float | Wall-clock seconds. |

**Array fields**

| Field | Shape | Meaning |
|---|---|---|
| `critical_values` | dict | `{alpha: quantile(KS*, 1−alpha)}`. |
| `boot_stats` | `(B,)` | Bootstrap replicates `KS*_{n,b}`. |
| `S_hat` | `(L,)` complex | `Ŝ_n(μ_ℓ, ν_ℓ)`. |
| `mu` | `(L, q)` | Direction draws for the `Y` block. |
| `nu` | `(L, p)` | Direction draws for the `X` block. |
| `residuals` | `(n_eff,)` | `Y_t − m̂(Y_{t−1})` on the standardised scale. |
| `m_hat` | `(n_eff,)` | MLP fitted conditional means. |
| `phi_hat` | `(n_eff, L)` complex | `φ̂(ν_ℓ \| Y_{t−1})`. All-zero when `doubly_robust=False`. |
| `influence` | `(n_eff, L)` complex | The summands `z_{t,ℓ}`. The bootstrap is `(ξ @ influence)/√n_eff`. |
| `settings` | dict | Full hyper-parameter record, plus `mlp_history`, `mdn_history`, and `networks` when requested. |

**Properties and methods**

| Member | Returns | Notes |
|---|---|---|
| `.stars` | str | `***` < 0.01, `**` < 0.05, `*` < 0.10, else `""`. |
| `.decision` | str | `"reject H0"` / `"do not reject H0"`. |
| `.summary()` | str | Formatted block, ready to paste into an appendix. |
| `.print()` | None | Prints `.summary()`. |
| `.to_dict()` | dict | Flat record for a results table. |
| `.to_frame()` | DataFrame | One-row version of `.to_dict()`. |

---

### `drgc_lag_scan`

```python
drgc_lag_scan(x, y, lags=range(1, 11), *, progress=True, seed=None, **kwargs)
    -> (pandas.DataFrame, dict[int, DRGCTResult])
```

Runs `drgc_test` over a grid of lag orders — the structure of Table 6 of the
paper. Lag `k` uses seed `seed + k`, so runs are independent yet reproducible.
`**kwargs` are forwarded to `drgc_test`.

The DataFrame has one row per lag with columns `direction, lag, p, q, n,
n_eff, ks_stat, pvalue, reject, stars, doubly_robust, cv_10, cv_5, cv_1,
elapsed`.

```python
scan, results = drgc_lag_scan(x, y, lags=range(1, 13), B=999, seed=1)
scan.query("reject")["lag"].tolist()      # which lags reject
results[7].print()                        # full output for lag 7
```

---

### `drgc_both_directions`

```python
drgc_both_directions(x, y, lag, *, x_name="X", y_name="Y", seed=None, **kwargs)
    -> {"x_to_y": DRGCTResult, "y_to_x": DRGCTResult}
```

```python
out = drgc_both_directions(P, V, lag=5, x_name="P_t", y_name="V_t", seed=1)
print(out["x_to_y"].pvalue, out["y_to_x"].pvalue)
```

---

### `drgc_stability`

```python
drgc_stability(x, y, lag, *, n_draws=25, seed=0, progress=False, **kwargs) -> dict
```

Re-runs the test over `n_draws` independent random-direction draws and network
initialisations, holding the data fixed. **Run this before reporting any
borderline result.**

| Key | Type | Meaning |
|---|---|---|
| `pvalues` | `(n_draws,)` | One p-value per draw. |
| `ks_stats` | `(n_draws,)` | One `KS_n` per draw. |
| `median`, `mean` | float | Central tendency of the p-values. |
| `q05`, `q95` | float | 5th and 95th percentiles. |
| `share_reject` | float | Fraction below `alpha`. |
| `merged_pvalue` | float | `min(1, 2 × median)` — valid under arbitrary dependence (Rüger; Vovk and Wang, 2020). Conservative. |
| `alpha`, `lag`, `n_draws`, `direction` | — | Echo of the settings. |
| `results` | list | The `DRGCTResult` objects. |

```python
stab = drgc_stability(x, y, lag=10, n_draws=30, L=20, B=999, seed=1)
print(stab["median"], stab["q05"], stab["q95"], stab["merged_pvalue"])
```

---

## 2. Networks — `drgct.nets`

### `MLPConfig`

```python
MLPConfig(
    width="paper", depth="paper", beta0=2, loss="l2",
    epochs=400, lr=5e-3, batch_size=512, weight_decay=0.0,
    patience=60, min_delta=1e-6, dropout=0.0,
    device=None, verbose=False,
)
```

| Field | Default | Description |
|---|---|---|
| `width` | `"paper"` | `"paper"` → `H_n = 5·lag`; `"theory"` → `n^{q/(2(β₀+q))} log²n` (Lemma 1); or an int. |
| `depth` | `"paper"` | `"paper"` → 1; `"theory"` → `round(log n)`; or an int. |
| `beta0` | `2` | Sobolev smoothness of `m(·)` (Assumption 4), used only by `"theory"`. |
| `loss` | `"l2"` | `"l2"` or `"smooth_l1"`; the paper permits either. |
| `epochs` | `400` | Maximum Adam epochs. |
| `lr` | `5e-3` | Learning rate. |
| `batch_size` | `512` | Effectively full batch for `n ≤ 512`. |
| `weight_decay` | `0.0` | L2 penalty. |
| `patience` | `60` | Early-stopping patience on the running training loss. |
| `min_delta` | `1e-6` | Minimum improvement counted as progress. |
| `dropout` | `0.0` | Dropout after each hidden layer. |
| `device` | `None` | `"cpu"`, `"cuda"`, or `None` for automatic. |
| `verbose` | `False` | Print the loss every 50 epochs. |

### `MDNConfig`

```python
MDNConfig(
    n_components=10, width="paper", depth="paper", beta0=2,
    min_sigma=1e-2, epochs=500, lr=5e-3, batch_size=512,
    weight_decay=0.0, patience=70, min_delta=1e-6,
    device=None, verbose=False,
)
```

Same fields, plus:

| Field | Default | Description |
|---|---|---|
| `n_components` | `10` | `G`. Overridden by `drgc_test(G=...)`. |
| `min_sigma` | `1e-2` | Floor on component standard deviations — the finite-sample counterpart of `σ_g(y) ≥ C⁻¹G^{−ω₂}` in Assumption 6(i). Raise to `5e-2` if training diverges. |

### `MLP`

```python
MLP(d_in, width, depth, dropout=0.0)          # torch.nn.Module
model(x) -> Tensor of shape (N,)
```

### `MixtureDensityNetwork`

```python
MixtureDensityNetwork(d_y, d_x, n_components, width, depth, min_sigma=1e-2)

model(y)               -> (log_alpha (N,G), mu (N,G,dx), sigma (N,G,dx))
model.log_prob(y, x)   -> (N,)     log f̂(x | y)
model.sample(y, M)     -> (N,M,dx) draws from f̂(· | y)
```

Diagonal-covariance Gaussian mixture; for `d_x = 1` this is exactly the
univariate mixture of Assumption 6(i).

### `fit_conditional_mean`

```python
fit_conditional_mean(ylag, y, config=None, *, lag=None) -> MeanFit
```

`MeanFit` has `.fitted` (the in-sample `m̂(Y_{t−1})`), `.model`, `.history`,
`.width`, `.depth`. Inputs are expected pre-standardised.

### `fit_conditional_density`

```python
fit_conditional_density(ylag, xlag, config=None, *, n_samples=20, lag=None, seed=None)
    -> DensityFit
```

`DensityFit` has `.samples` of shape `(N, M, p)`, `.model`, `.history`,
`.width`, `.depth`, `.n_components`.

### Architecture helpers

```python
paper_width(lag, multiplier=5) -> int      # 5 * lag
theory_width(n, q, beta0=2)    -> int      # n^{q/(2(beta0+q))} log^2 n
theory_depth(n)                -> int      # round(log n)
```

---

## 3. Benchmark — `drgct.nhkj`

```python
nhkj_test(
    x, y, lag=None, *,
    p=None, q=None,
    bandwidth_const=None, bandwidth=None, bandwidth_exponent=-0.15,
    kernel_order=4, alpha=0.05, standardize=True, chunk=512,
    x_name="X", y_name="Y",
) -> NHKJResult
```

Smoothing-based nonparametric conditional-moment test of the Nishiyama,
Hitomi, Kawasaki and Jeong (2011) class, in the Zheng (1996) / Fan–Li (1996)
degenerate U-statistic form, configured as the DRGCT paper specifies for its
benchmark. See the module docstring for the exact statement of what is and is
not a transcription of the original article.

| Argument | Default | Description |
|---|---|---|
| `bandwidth_const` | `None` | `c` in `h = c·n^{−0.15}`. Default schedule: `2.5` at lag 1, `3.0` at lags 2–3, `3.5` at lags 4–5, `4.0` beyond. Section 4 shifts these up by 0.5 for the exponential-mean designs S2/P3. |
| `bandwidth` | `None` | Set `h` directly, overriding `bandwidth_const`. |
| `bandwidth_exponent` | `-0.15` | Exponent in `h = c·n^{exponent}`. |
| `kernel_order` | `4` | `4` → `K₄(u) = 0.5(3 − u²)φ(u)`; `2` → standard Gaussian. |
| `standardize` | `True` | z-score each coordinate of `W_{t−1}` so one scalar bandwidth is sensible. |
| `chunk` | `512` | Row-block size for the `O(n²d)` pairwise kernel computation. Lower it if memory is tight. |

`NHKJResult` fields: `stat` (asymptotically `N(0,1)`), `pvalue` (one-sided),
`gamma`, `sigma`, `bandwidth`, `bandwidth_const`, `kernel_order`, `p`, `q`,
`lag`, `n`, `n_eff`, `dim_w`, `alpha`, `reject`, `direction`, `elapsed`,
`.stars`, `.summary()`, `.print()`, `.to_dict()`.

Also exported: `gaussian_kernel(u)`, `gaussian_kernel4(u)`.

---

## 4. Designs — `drgct.dgp`

```python
simulate_dgp(dgp, n, lag, *, rng=None, burn=500, sigma2=0.5,
             innovation_scale="variance", params=None) -> SimulatedSeries
```

| Argument | Default | Description |
|---|---|---|
| `dgp` | — | `"S1"`, `"S2"` (null) or `"P1"`–`"P4"` (alternative). |
| `n` | — | Observations kept. |
| `lag` | — | `p = q = lag`. |
| `rng` | `None` | `Generator`, int seed, or `None`. |
| `burn` | `500` | Initial observations discarded (`n₀` in the paper). |
| `sigma2` | `0.5` | Innovation dispersion. |
| `innovation_scale` | `"variance"` | The paper writes `N(0, 0.5)`; read as a variance by default, `"sd"` to read it as a standard deviation. |
| `params` | `None` | Override the Table 2 coefficients: keys `a`, `b`, `c` (lists of length `lag`) and `a_exp`, `c_scalar`, `a0`. |

`SimulatedSeries` has `.x`, `.y`, `.dgp`, `.lag`, `.n`, `.causal`, `.params`,
`.as_tuple()`.

Other exports: `DGP_NAMES`, `SIZE_DGPS`, `POWER_DGPS`, `PARAMETERS`
(the Table 2 dictionary), `DGP_EQUATIONS` (LaTeX), `dgp_parameters(lag)`,
`dgp_table()` (Table 1), `parameter_table(lags)` (Table 2).

---

## 5. Monte Carlo — `drgct.simulate`

```python
monte_carlo(
    dgps=("S1","S2","P1","P2","P3","P4"),
    ns=(500, 1000, 2000),
    lags=(1, 2, 3, 4, 5),
    reps=1000, *,
    methods=("drgc", "nhkj"),
    alpha=0.05, seed=20250915, n_jobs=1,
    drgc_kwargs=None, nhkj_kwargs=None, burn=500,
    progress=True, out_csv=None, flush_every=200,
) -> pandas.DataFrame
```

Returns a **long** frame: `dgp, n, lag, rep, causal, method, stat, pvalue,
reject, elapsed`.

- `methods` may include `"drgc_naive"` — the plug-in without the doubly robust
  correction.
- `n_jobs=-1` uses `cpu_count() − 1`. Guard the entry point with
  `if __name__ == "__main__":`.
- `out_csv` streams partial results every `flush_every` completed jobs so a
  long run survives an interruption.

```python
summarize(df, *, alpha=None) -> DataFrame
# dgp, n, lag, method, reps, rejection_rate, mean_pvalue, causal, mc_se

size_power_tables(df, *, alpha=None, methods=("drgc","nhkj")) -> {"size":…, "power":…}
# (lag, n) row MultiIndex x (DGP, method) column MultiIndex

run_replication(dgp, n, lag, rep, *, seed=0, methods=("drgc","nhkj"), alpha=0.05,
                drgc_kwargs=None, nhkj_kwargs=None, burn=500, torch_threads=1)
    -> list[dict]

paper_bandwidth_const(dgp, lag) -> float     # the Section 4 NHKJ schedule
PAPER_BANDWIDTH_CONST                        # the underlying dict
```

---

## 6. Data — `drgct.datasets`

```python
available_datasets()  -> ['csi300', 'nikkei225', 'spx500']
data_dir()            -> pathlib.Path

load_index(name, *, start=None, end=None, path=None) -> DataFrame  # Close, Volume
load_all(keys=INDEX_KEYS, **kwargs)                  -> {key: DataFrame}

to_percentage_changes(df, *, price_col="Close", volume_col="Volume",
                      volume_divisor=10.0, in_percent=True) -> DataFrame  # P, V

subsample(df, period)   -> DataFrame     # period is a PAPER_PERIODS key or (start, end)
describe(series_map, *, add_tests=True) -> DataFrame
```

Constants: `INDEX_KEYS`, `INDEX_LABELS`, `PAPER_PERIODS`
(`{"2019-2022": ("2019-09-27","2022-09-26"), …}`), `PAPER_START`, `PAPER_END`.

`describe` returns: Obs., Mean, Median, Std. dev., Min., Max., Skewness,
Kurtosis, Jarque–Bera + p, Ljung–Box(10) + p, Ljung–Box²(10) + p, ADF p,
KPSS p.

Rebuild or extend the bundled CSVs:

```bash
python data/fetch_data.py
python data/fetch_data.py --start 2005-01-01 --end 2024-12-31
python data/fetch_data.py --tickers ^FTSE=ftse100 ^GDAXI=dax
```

Use your own file without touching the package:

```python
df = load_index("anything", path="/path/to/my.csv")   # needs Date, Close, Volume
```

---

## 7. Applications — `drgct.applications`

```python
price_volume_study(
    indices=INDEX_KEYS,
    periods=tuple(PAPER_PERIODS),
    lags=range(1, 11), *,
    directions=("P_t -> V_t", "V_t -> P_t"),
    alpha=0.05, drgc_kwargs=None, volume_divisor=10.0,
    seed=20240926, n_jobs=1, progress=True, out_csv=None,
) -> DataFrame
```

Long frame: `index_key, index_label, period, start, end, direction, n_obs,
lag, ks_stat, pvalue, reject, stars, cv_5, n_eff, elapsed`. Feeds
`table_detection`, `table_lag_orders`, `table_pvalues` and
`plot_pvalue_heatmap` directly.

`periods` may be an explicit mapping: `{"post-COVID": ("2020-04-01", "2024-09-26")}`.

```python
rolling_causality(
    x, y, *, lag=5, window=750, step=21, dates=None, alpha=0.05,
    both_directions=True, x_name="P_t", y_name="V_t",
    drgc_kwargs=None, seed=7, n_jobs=1, progress=True,
) -> DataFrame
# direction, start_idx, end_idx, start_date, end_date, lag, ks_stat, pvalue, reject

lag_scan_frame(
    x, y, lags=range(1, 11), *, x_name="X", y_name="Y",
    both_directions=True, alpha=0.05, drgc_kwargs=None,
    seed=11, n_jobs=1, progress=True,
) -> DataFrame
# same tidy layout, with placeholder index_label / period

run_one_test(x, y, lag, *, meta, alpha=0.05, drgc_kwargs=None,
             seed=None, torch_threads=1) -> dict
```

---

## 8. Tables — `drgct.tables`

### Export

```python
export_table(df, stem, outdir="results/tables", *, caption="", label="",
             notes="", index=False, float_format="%.3f", quiet=False,
             **latex_kwargs) -> {"csv":…, "md":…, "tex":…}

to_latex_booktabs(df, *, caption="", label="", notes="", align=None,
                  index=False, float_format="%.3f", position="htbp",
                  small=True, escape=True) -> str
```

`export_table` writes `.tex` (booktabs), `.md` and `.csv` in one call.
`align` defaults to `l` for object columns and `c` for numeric ones.
`escape=True` translates `&`, `%`, `_`, `#` and the tick/cross glyphs into
`\checkmark` / `$\times$`.

### Builders

| Function | Paper table | Signature |
|---|---|---|
| `table_dgp_definitions()` | Table 1 | — |
| `table_parameter_settings(lags=(1,2,3,4,5))` | Table 2 | — |
| `table_size(mc_df, *, dgps=("S1","S2"), methods=("drgc","nhkj"), alpha=0.05, digits=3)` | Table 3 | takes `monte_carlo` output |
| `table_power(mc_df, *, dgps=("P1","P2","P3","P4"), methods=…, alpha=0.05, digits=3)` | Table 4 | " |
| `table_detection(app_df, *, alpha=0.05, rule="any")` | Table 5 | `rule` ∈ `{"any","majority","all"}` |
| `table_lag_orders(app_df, *, alpha=0.05, lags=None)` | Table 6 | ticks/crosses |
| `table_pvalues(app_df, *, digits=3, lags=None)` | — | Table 6 with numbers |
| `table_descriptives(series_map, *, digits=3)` | — | moments + diagnostics |
| `table_hyperparameters(result_or_settings)` | — | referee-proof settings record |

Also exported: `TICK`, `CROSS`, `fmt_rate(v, digits=3)`.

**The tidy long format** required by `table_detection`, `table_lag_orders`,
`table_pvalues` and `plot_pvalue_heatmap`:

| Column | Type | Example |
|---|---|---|
| `index_label` | str | `"SPX 500"` |
| `period` | str | `"2021-2024"` |
| `direction` | str | `"P_t -> V_t"` |
| `lag` | int | `1 … 10` |
| `pvalue` | float | `0.0132` |

---

## 9. Figures — `drgct.plots`

```python
use_journal_style(*, base_size=9.5, serif=True) -> None
save_figure(fig, stem, outdir="results/figures", *, formats=("pdf","png"),
            close=True, quiet=False) -> dict
PALETTE        # colour-blind-safe named colours
METHOD_STYLE   # consistent colour/marker/linestyle per estimator
```

| Function | Input | Shows |
|---|---|---|
| `plot_size(summary_df, *, nominal=0.05, dgps=("S1","S2"), methods=("drgc","nhkj"), ylim=None, figsize=None, band=True)` | `summarize()` | Size vs lag, with the nominal level and a ±1.96 MC-s.e. band |
| `plot_power(summary_df, *, dgps=("P1",…,"P4"), methods=…, figsize=None, ncol=2)` | `summarize()` | Power vs lag, one panel per DGP, one line style per `n` |
| `plot_size_power_grid(summary_df, **kwargs)` | `summarize()` | `(size_fig, power_fig)` |
| `plot_pvalue_ecdf(mc_df, *, dgps=("S1","S2"), methods=…, n=None, lags=None, figsize=None)` | `monte_carlo()` | Davidson–MacKinnon p-value plot; uniformity under `H0` |
| `plot_bootstrap_distribution(result, *, bins=45, alphas=(0.10,0.05,0.01), figsize=(5.4,3.2))` | `DRGCTResult` | Bootstrap null with `KS_n` and critical values |
| `plot_empirical_process(result, *, envelope=0.95, figsize=(6.6,3.0))` | `DRGCTResult` | `Re`/`Im` of `Ŝ_n` at each direction vs the bootstrap envelope |
| `plot_training_curves(result, *, figsize=(5.6,2.6))` | `DRGCTResult` | MLP loss and MDN NLL |
| `plot_mdn_fit(result, *, coordinate=0, bins=40, figsize=(5.4,3.0))` | `DRGCTResult` with `return_networks=True` | MDN draws vs the empirical marginal of `X_{t−1}` |
| `plot_lag_profile(scan_df, *, alpha=0.05, label="", figsize=(5.6,3.0), show_stat=True)` | `drgc_lag_scan()` | p-value (and `KS_n`) against lag |
| `plot_pvalue_heatmap(app_df, *, alpha=0.05, directions=None, index_order=…, figsize=None, annotate=True)` | tidy long | Lag × period p-value map, one panel per direction |
| `plot_series_overview(raw_map, transformed_map=None, *, figsize=None)` | `{label: DataFrame}` | Levels, volumes, and the transformed series |
| `plot_stability(stability, *, label="", bins=20, figsize=(5.8,3.0))` | `drgc_stability()` | p-value distribution across direction draws |
| `plot_rolling_pvalue(roll_df, *, alpha=0.05, label="", figsize=(6.4,3.0))` | `rolling_causality()` | p-value against window end date |

All return a `matplotlib.figure.Figure`, so you can post-process before
saving. `save_figure` accepts any Matplotlib format (`"svg"`, `"eps"`, …).

---

## 10. Utilities — `drgct.utils`

```python
set_seed(seed) -> numpy.random.Generator
    # seeds random, numpy, torch; returns the Generator to use

as_series(v, name="series") -> ndarray
    # coerce to finite 1-D float64; raises on NaN/inf/short input

build_lag_design(x, y, p, q, *, allow_p_gt_q=True) -> LagDesign
    # .y (n_eff,)  .ylag (n_eff,q)  .xlag (n_eff,p)  .w (n_eff,p+q)
    # .p .q .n .n_eff .dim_w .t_index

zscore(a, *, ddof=0, eps=1e-12) -> (z, mean, scale)

rademacher(rng, size)                  -> ndarray
mammen(rng, size)                      -> ndarray
draw_multipliers(rng, size, kind="rademacher") -> ndarray

check_stationarity(series, name="series", *, alpha=0.05) -> dict
    # adf_stat, adf_pvalue, kpss_stat, kpss_pvalue, stationary, message
```

`check_stationarity` reports `stationary=True` only when ADF rejects the unit
root **and** KPSS fails to reject stationarity.

---

## 11. Command line

```
drgct <command> [options]
```

### `drgct test`

```bash
drgct test DATA.csv -x cause_col -y effect_col [options]
```

| Option | Default | Description |
|---|---|---|
| `-x`, `-y` | required | Column names. |
| `--lag` | `1` | Single lag order. |
| `--lag-scan` | off | Scan instead of a single test. |
| `--lag-min`, `--lag-max` | `1`, `10` | Scan range. |
| `-G -L -M -B` | `10 20 20 1000` | Test hyper-parameters. |
| `--alpha` | `0.05` | |
| `--seed` | `20250915` | |
| `--epochs` | — | Override both networks' epochs. |
| `--outdir` | `results` | |
| `--save` | off | Also write tables and figures. |

```bash
drgct test macro.csv -x m2_growth -y cpi_inflation --lag-scan --lag-max 18 --save
```

### `drgct simulate`

```bash
drgct simulate [--dgps S1 S2 P1 P2 P3 P4] [--ns 500 1000 2000]
               [--lag-min 1] [--lag-max 5] [--reps 1000]
               [--methods drgc drgc_naive nhkj] [--jobs -1] [--outdir results]
```

### `drgct app`

```bash
drgct app [--indices spx500 csi300 nikkei225]
          [--periods 2019-2022 2020-2023 2021-2024]
          [--lag-min 1] [--lag-max 10] [--jobs -1] [--outdir results]
```

### `drgct info`

```bash
drgct info            # version, data directory, citation
drgct info --bibtex
```

### The full-study scripts

```bash
python scripts/run_simulation.py  --reps 1000 --ns 500 1000 2000 -B 1000 --jobs 10
python scripts/run_application.py --jobs 10 --rolling --stability-draws 30
python scripts/run_simulation.py  --quick        # 2-minute plumbing check
python scripts/run_application.py --quick
```

---

## 12. Argument cheat sheet

| Want to… | Do this |
|---|---|
| Test one direction at one lag | `drgc_test(x, y, lag=k, seed=1)` |
| Test both directions | `drgc_both_directions(x, y, lag=k, seed=1)` |
| Scan lags 1–12 | `drgc_lag_scan(x, y, lags=range(1, 13), seed=1)` |
| Different lags for cause and effect | `drgc_test(x, y, p=12, q=6)` |
| More power | raise `L` (cheap), then `B`, then `epochs` |
| Fix a size distortion | raise `G`, raise `epochs`, check `plot_mdn_fit` |
| Faster exploratory runs | `mlp=MLPConfig(epochs=150)`, `mdn=MDNConfig(epochs=200)`, `B=299` |
| Check the result is not a lucky draw | `drgc_stability(x, y, lag=k, n_draws=30)` |
| Reproduce the naive-plug-in failure | `drgc_test(..., doubly_robust=False)` |
| Use the theoretical architecture | `mlp=MLPConfig(width="theory", depth="theory")` |
| Bounded, skewed multipliers | `multiplier="mammen"` |
| Probe finer oscillations in `W` | `w_lower=-3, w_upper=3` |
| Compare against a smoothing test | `nhkj_test(x, y, lag=k)` |
| Continuous-time view of causality | `rolling_causality(x, y, window=750, step=21)` |
| LaTeX for a manuscript | `export_table(df, "name", caption=..., label=...)` |
| Vector + raster figure in one call | `save_figure(fig, "name")` |
