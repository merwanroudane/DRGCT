# Price and volume in three markets

<p class="lede">The application from Section 5 of the paper, reproduced in full:
does the daily percentage change in an index level Granger-cause the percentage
change in trading volume, and vice versa? Three markets, three overlapping
three-year sub-samples, ten lag orders, both directions — 180 tests — plus a
rolling-window extension the paper points to but does not carry out.</p>

Produced by
[`scripts/run_application.py`](https://github.com/merwanroudane/DRGCT/blob/main/scripts/run_application.py):

```bash
python scripts/run_application.py --jobs -1 --rolling --stability-draws 30
```

<div class="stat-row">
  <div class="stat"><div class="v">180</div><div class="l">doubly robust tests in the main grid</div></div>
  <div class="stat"><div class="v">1257</div><div class="l">daily observations for the S&amp;P 500 — exactly the paper's <i>T</i></div></div>
  <div class="stat"><div class="v">1–10</div><div class="l">lag orders, in trading days</div></div>
  <div class="stat"><div class="v">7%</div><div class="l">of direction draws that rejected the headline test — see §5</div></div>
</div>

## 1. Why prices and volume

Trading volume rises when prices move sharply and falls when they do not; heavy
volume is often followed by large price moves. The relationship is one of the
oldest documented regularities in empirical finance (Karpoff, 1987; Gallant,
Rossi and Tauchen, 1992; Chen, Hong and Stein, 2001), and it is a natural test
bed for a nonlinear causality test: the mechanism is asymmetric, involves
thresholds, and operates over several days rather than one.

Following Section 5 of the paper, both series are transformed to percentage
changes, and the volume change is divided by ten so the two have a comparable
scale:

<div class="eq">
<code>P(t) = 100 · (Close(t)/Close(t−1) − 1)</code><br>
<code>V(t) = 100 · (Volume(t)/Volume(t−1) − 1) / 10</code>
</div>

## 2. Data

Daily closing levels and volumes over the paper's exact window,
**27 September 2019 – 26 September 2024**.

| Key | Index | Ticker | Observations |
|---|---|---|---|
| `spx500` | S&P 500 | `^GSPC` | 1257 — matches the paper's `T = 1257` |
| `csi300` | CSI 300 (exchange-traded tracker) | `510300.SS` | 1211 |
| `nikkei225` | Nikkei 225 | `^N225` | 1220 |

<div class="warn"><strong>These are not the paper's data</strong>
<p>The paper does not name its vendor. Closing <i>levels</i> are near identical
across vendors; daily <i>volume</i> is not — and volume dynamics are the whole
exercise here. For the CSI 300, Yahoo truncates the index series
(<code>000300.SS</code>) to roughly the last three years, so the largest
exchange-traded tracker is used instead: genuine exchange turnover, but ETF
turnover rather than index-constituent turnover. Expect the results below to
agree with the paper qualitatively in some blocks and not others; §6 sets out
exactly which.</p></div>

{{figure: fig5_data_overview.png | **Figure 1.** Closing levels, volumes, and the transformed series `P(t)` and `V(t)` for the three indices.}}

{{table: results/tables/table0_descriptives.csv | **Table 1.** Descriptive statistics for the six transformed series. All are strongly leptokurtic and serially correlated in the square — the usual daily-financial-data signature.}}

## 3. Results

Hyper-parameters as in the paper: `G = 10`, `L = 20`, `M = 20`, `B = 999`,
Rademacher multipliers, 5% level.

{{figure: fig6_pvalue_heatmap.png | **Figure 2.** Bootstrap p-values for every index, sub-sample and lag order, one panel per direction. Outlined cells reject at 5%.}}

{{table: results/tables/table5_detection.csv | **Table 2.** Detection summary, in the layout of the paper's Table 5. A tick marks rejection at 5% for at least one lag order between 1 and 10.}}

{{table: results/tables/table6_lag_orders.csv | **Table 3.** Decisions at each lag order, the layout of the paper's Table 6.}}

{{table: results/tables/table6b_pvalues.csv | **Table 4.** The same grid with the bootstrap p-values themselves.}}

### What we find

**CSI 300 — prices drive volumes, decisively.** Rejection at lags 1–3 in *every*
sub-sample, with `p ≤ 0.011` throughout, strengthening over time (in 2021–2024,
`p < 0.001` at lags 1, 2, 3 and 6). The reverse direction rejects at one lag in
one sub-sample out of thirty tests — chance.

**Nikkei 225 — prices drive volumes at short horizons.** Rejection at lags 1
and 3 in all three sub-samples. Volumes never predict prices at any lag in any
sub-sample.

**S&P 500 — nothing, in either direction, at any lag.**

The common pattern across the two markets where anything is detectable —
prices lead volumes, volumes do not lead prices — is the paper's headline
conclusion and matches what the price–volume literature would predict.

{{figure: fig7_lagprofile_csi300_p2v.png | **Figure 3.** CSI 300, 2021–2024, prices to volumes: the p-value against the lag order. The effect is concentrated at one to three days, with a revival at six.}}

## 4. A rolling window

The paper's three overlapping sub-samples are a coarse way of asking whether
causality is stable. Sliding a fixed window through calendar time answers the
sharper question: *when* does it switch on?

{{figure: fig12_rolling_spx500.png | **Figure 4.** Rolling DRGCT for the S&P 500: 750-day windows advanced one month at a time, both directions, lag 5. Consecutive windows overlap by 97%, so read this as a description of instability, not as 25 independent tests. Nothing sustained appears.}}

```python
from drgct.applications import rolling_causality
roll = rolling_causality(P, V, lag=5, window=750, step=21,
                         dates=pv.index, n_jobs=-1, seed=7)
```

## 5. The headline test does not survive its robustness check

The single most significant cell in the S&P 500 block was `P → V` at lag 10 in
2021–2024, where one seed produced `p = 0.008`. Re-running it across thirty
independent draws of the `L = 20` evaluation directions tells a different
story.

{{table: results/tables/table8_stability.csv | **Table 5.** Thirty independent draws of the direction pairs and of the network initialisation, data held fixed.}}

The median p-value is **0.52** and only **7%** of draws reject. The `p = 0.008`
was a lucky seed. Reporting it alone would have been misleading.

{{figure: fig13_stability.png | **Figure 5.** The p-value distribution across thirty draws — close to uniform on [0,1], which is what you see when the null is true.}}

{{figure: fig9_empirical_process.png | **Figure 6.** Why: the real and imaginary parts of the empirical process at each of the twenty directions. Exactly one direction escapes the bootstrap envelope. Contrast this with the <a href="macro.html#6-is-the-headline-a-lucky-draw">macro headline</a>, where many directions escape and all thirty draws reject.}}

<div class="good"><strong>The lesson</strong>
<p>This is the clearest demonstration on the site of why
<code>drgc_stability</code> exists. Two honest analysts using different seeds
would have reached opposite conclusions from the same data. Raise <code>L</code>,
report the distribution, and quote the merged p-value.</p></div>

## 6. Where this agrees with the paper, and where it does not

| Direction | Index | Paper (Table 5) | Here | Agrees? |
|---|---|:--:|:--:|:--:|
| `P → V` | CSI 300 | ✓ in 2020-23, 2021-24 | ✓ in all three | yes, stronger here |
| `P → V` | NI 225 | ✗ everywhere | ✓ in all three | **no** |
| `P → V` | SPX 500 | ✓ in all three | ✗ everywhere | **no** |
| `V → P` | CSI 300 | ✓ in 2021-24 only | ✓ in 2019-22 only | partly |
| `V → P` | NI 225 | ✗ everywhere | ✗ everywhere | yes |
| `V → P` | SPX 500 | ✗ everywhere | ✗ everywhere | yes |

Four of six blocks agree. The two that do not are a data story, not an
estimator story. A plain linear VAR *F*-test on these same series finds
`P → V` strongly significant for the CSI 300 in every sub-sample (`p ≤ 0.0009`
at lag 1) and never significant for the Nikkei 225 or the S&P 500 — so the
CSI 300 and S&P 500 findings here are what *this* data says, linear or
nonlinear. Where the DRGCT adds value is the Nikkei 225, where it detects
short-lag causality that the linear test misses entirely.

**If you have the paper's data**, drop your CSVs into `src/drgct/data/` with
`Date, Close, Volume` columns and re-run the script. No code changes needed.

## 7. Diagnostics for a single reported test

{{figure: fig8_bootstrap_null.png | **Figure 7.** The bootstrap null distribution with the observed statistic and the 10/5/1% critical values. Put this beside any single test you report.}}

{{figure: fig10_mdn_fit.png | **Figure 8.** Mixture density network fit: pooled draws from the estimated conditional density against the empirical marginal of the lagged cause.}}

{{figure: fig11_training_curves.png | **Figure 9.** MLP training loss and MDN negative log-likelihood. Both are still declining at the epoch cap here, which means the networks are mildly under-trained at lag 10 — raise `epochs`. The package now emits a warning when this happens.}}

## References

- Chen, J., Hong, H. and Stein, J. C. (2001). Forecasting crashes: trading volume, past returns, and conditional skewness in stock prices. *Journal of Financial Economics* 61, 345–381.
- Copeland, T. E. (1976). A model of asset trading under the assumption of sequential information arrival. *The Journal of Finance* 31, 1149–1168.
- Gallant, A. R., Rossi, P. E. and Tauchen, G. (1992). Stock prices and volume. *The Review of Financial Studies* 5, 199–242.
- Hui, Y., Liu, C. and Song, X. (2025). *Deep learning based doubly robust test for Granger causality.* [arXiv:2509.15798](https://arxiv.org/abs/2509.15798).
- Karpoff, J. M. (1987). The relation between price changes and trading volume: a survey. *Journal of Financial and Quantitative Analysis* 22, 109–126.
