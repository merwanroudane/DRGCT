# US macroeconomics, 1959–2025

<p class="lede">Sixty-seven years of monthly FRED data, six relations that the
macro literature argues about, both directions, lag orders from one month to
eighteen. This is the application the DRGCT was built for: economic mechanisms
operate with long and variable lags, and long lags are exactly where
kernel-smoothing causality tests stop working.</p>

Everything on this page was produced by
[`scripts/run_macro_application.py`](https://github.com/merwanroudane/DRGCT/blob/main/scripts/run_macro_application.py).
Nothing is hand-edited. Reproduce it with:

```bash
pip install "drgct[data]"
python scripts/run_macro_application.py --jobs -1
```

<div class="stat-row">
  <div class="stat"><div class="v">801</div><div class="l">monthly observations, Feb 1959 – Dec 2025</div></div>
  <div class="stat"><div class="v">108</div><div class="l">doubly robust tests (72 pooled + 36 sub-sample)</div></div>
  <div class="stat"><div class="v">6</div><div class="l">relations, each tested in both directions</div></div>
  <div class="stat"><div class="v">1–18</div><div class="l">lag orders in months</div></div>
</div>

## 1. The questions

Each relation is a proposition somebody has defended in print.

| # | Relation | The claim being tested |
|---|---|---|
| 1 | Fed funds rate → Industrial production | Monetary policy moves real activity — and Friedman (1961) insisted the lags are *long and variable*, which is precisely what a lag-1 test cannot see. |
| 2 | Fed funds rate → CPI inflation | The price leg of monetary transmission, conventionally thought slower than the output leg. |
| 3 | M2 money growth → CPI inflation | The quantity theory, stated as a forecasting claim. |
| 4 | WTI oil price → CPI inflation | Pass-through of energy supply shocks into the headline basket. |
| 5 | WTI oil price → Industrial production | Hamilton's (1983) oil-shock channel. Famously **asymmetric** — price increases hurt more than decreases help — so a linear test is the wrong instrument by construction. |
| 6 | Industrial production → Unemployment rate | Okun's law read dynamically rather than as a contemporaneous regression. |

Each is also run in reverse, because a one-directional finding is a much
stronger statement than a feedback loop, and because reverse causality is the
standard objection to any of these claims.

## 2. Data and transformations

Eight monthly series from the Federal Reserve Bank of St. Louis. Indices,
prices, money and employment enter as `100 × Δlog` (monthly growth in per
cent); the two series already measured in per cent — the federal funds rate and
the unemployment rate — enter as first differences.

{{table: results/macro/tables/macro_table1_series.csv | **Table 1.** Series, transformations and stationarity screens. ADF reports the p-value of the unit-root null, KPSS the p-value of the stationarity null. A series is marked stationary only when ADF rejects *and* KPSS does not.}}

<div class="warn"><strong>An honest problem with the pooled sample</strong>
<p>Four series — CPI inflation, PCE inflation, industrial production growth and
payroll growth — <b>fail the KPSS screen on the full sample</b> even though ADF
rejects the unit root. Assumption 1 of the paper requires strict stationarity,
so the pooled results below are on shakier ground than the sub-sample results.</p>
<p>The failure is not mysterious. It is the Great Moderation: the mean and
variance of US inflation and output growth shift around 1984, and KPSS with a
constant reads a level shift as non-stationarity. Split the sample and the
screen behaves — CPI inflation has KPSS <i>p</i> = 0.083 within 1984–2025
against 0.010 pooled. That is one more reason to take
<a href="#4-before-and-after-the-great-moderation">Section 4</a> more seriously
than Section 3.</p></div>

{{figure: macro_fig1_series.png | **Figure 1.** The six series used in the tests, in levels (left) and after transformation (right). The transformed series are what the test actually sees. The volatility break around 1984 is visible in industrial production and inflation.}}

{{table: results/macro/tables/macro_table2_descriptives.csv | **Table 2.** Descriptive statistics. Ljung–Box statistics use 10 lags; all six series are strongly serially correlated in the level and in the square, which is why the conditional mean `m(Y(t−1))` is a non-trivial object.}}

## 3. Results: the pooled sample

Hyper-parameters: `G = 10` mixture components, `M = 20` pseudo-samples,
`B = 999` bootstrap replications, Rademacher multipliers. `L` is raised from
the paper's 20 to **60** random directions, because at lag 18 the conditioning
set has `p + q = 36` dimensions and twenty directions cover that space far too
thinly — see [the guide's §5.4](guide.html#54-choose-the-hyper-parameters).

{{figure: macro_fig2_pvalue_heatmap.png | **Figure 2.** Bootstrap p-values for all twelve directed relations at six lag orders. Outlined cells reject the null of non-causality at 5%. Warm cells are rejections; cool cells are not.}}

{{table: results/macro/tables/macro_table4_pvalues.csv | **Table 3.** Bootstrap p-values by lag order, pooled sample.}}

### What the pooled sample says

**Oil prices Granger-cause inflation, decisively and at every short horizon.**
`p < 0.001` at one, three and six months, and still 0.036 at eighteen. This is
the strongest result on the page, and it survives the robustness check in
[Section 6](#6-is-the-headline-a-lucky-draw) — all thirty independent direction
draws reject. The reverse direction is significant at a single lag out of six,
which is what you would expect by chance.

**Industrial production Granger-causes unemployment.** `p < 0.001`, 0.005 and
0.037 at one, three and six months: Okun's law holds as a dynamic forecasting
statement, not merely as a contemporaneous correlation. Note that unemployment
also predicts industrial production at lag 1 — genuine two-way feedback at the
monthly frequency, as two cyclical variables should show.

**Money growth predicts inflation at short and medium horizons.** `p` = 0.022
at one month and 0.013 at six. Not overwhelming, but the quantity theory is not
dead in the data.

**Monetary policy is the weak case here** — and that is the interesting part.
The federal funds rate predicts industrial production at one month
(`p` = 0.002) and then nothing out to eighteen. It never predicts inflation at
conventional levels in the pooled sample. Before concluding that monetary
policy does not work, read Section 4: pooling 1959–2025 pools two entirely
different monetary regimes.

{{figure: macro_fig3_lagprofile_oil_to_prices.png | **Figure 3a.** Oil prices to CPI inflation. The p-value (blue, left axis) sits in the rejection band out to six months; `KS_n` (green, right axis) tracks it.}}

{{figure: macro_fig3_lagprofile_output_to_unemployment.png | **Figure 3b.** Industrial production to unemployment — Okun's law, decaying with horizon exactly as one would hope.}}

{{figure: macro_fig3_lagprofile_mp_to_output.png | **Figure 3c.** Federal funds rate to industrial production, pooled sample. Significant at one month, then flat. Compare with the sub-sample split below.}}

## 4. Before and after the Great Moderation

The pooled sample runs across a structural break that almost everyone
recognises: US output growth and inflation become dramatically less volatile
after about 1984 (McConnell and Perez-Quiros, 2000; Stock and Watson, 2002),
and monetary policy operates under a different reaction function. Testing the
three monetary relations separately in the two eras is not a robustness check —
it is the right specification.

{{table: results/macro/tables/macro_table6_subsamples.csv | **Table 4.** The three monetary relations, Great Inflation (1959–1983, *n* ≈ 299) versus Great Moderation (1984–2025, *n* ≈ 502).}}

{{figure: macro_fig5_subsamples.png | **Figure 4.** The same numbers as a heat map. The story is in the top block: monetary policy to output.}}

**Monetary transmission to output was detectable in the Great Inflation era and
is weaker afterwards.** In 1959–1983 the federal funds rate predicts industrial
production at three months (`p` = 0.018), with borderline evidence at six
(0.078) and twelve (0.056). In 1984–2025 only the six-month horizon rejects
(0.033). Pooling the two eras is what produced the flat profile in Figure 3c.

**The price leg appears only in the Great Moderation** (`p` = 0.048 at three
months), and even then it is marginal.

**Money to prices weakens.** Borderline at six months in the Great Inflation
(`p` = 0.077), nothing in the Great Moderation. Consistent with the standard
account of money-demand instability and the decline of monetary aggregates as
policy indicators after the early 1980s.

**Reverse causality shows up where you would expect it.** Inflation predicts
the federal funds rate at three months in the Great Inflation era (`p` = 0.025)
and industrial production predicts the federal funds rate at three months in
the Great Moderation (`p` = 0.002) — that is the policy *reaction function*
being picked up, not policy transmission. Granger causality is symmetric in the
statistical sense; only economics tells you which arrow is a mechanism and
which is a reaction.

## 5. Three tests, the same data

Every relation was also run through a textbook linear VAR *F*-test and through
the smoothing-based nonparametric benchmark that the paper compares against.

{{figure: macro_fig4_comparison.png | **Figure 5.** p-values from the three tests for each relation and lag order. Shaded band is rejection at 5%. Blue circles: DRGCT. Green squares: linear VAR *F*-test. Terracotta triangles: smoothing benchmark.}}

{{table: results/macro/tables/macro_table5_comparison.csv | **Table 5.** The same comparison in numbers.}}

<div class="note"><strong>How to read this comparison</strong>
<p><b>The DRGCT is the most conservative of the three, and that deserves an
explanation rather than a boast.</b> US macro series are strongly and
<i>linearly</i> predictable, and with <i>n</i> ≈ 800 the linear <i>F</i>-test has
enormous power against exactly the alternative it was built for. It rejects
almost everywhere. The smoothing benchmark also rejects almost everywhere, but
in our runs it is <b>not size-controlled</b> at these conditioning dimensions
(see the <a href="simulation.html">simulation page</a>), so its column is not
evidence of anything.</p>
<p>The right reading of a DRGCT non-rejection at, say, twelve months is
therefore: <i>no evidence of predictive content beyond what the effect's own
twelve-month history already carries</i> — not "no relationship". The DRGCT
conditions on a twelve-dimensional own-past nonparametrically, which is a much
harder benchmark to beat than the linear test's twelve coefficients.</p>
<p>Where the DRGCT rejects and the linear test does not — money to prices at
one month is the clearest case — you have evidence of genuinely nonlinear
predictive content.</p></div>

## 6. Is the headline a lucky draw?

The strongest result is oil prices to CPI inflation at one month. Before
reporting it, it should survive the random draw of evaluation directions
described in [the guide's §5.8](guide.html#58-check-that-the-result-is-not-an-artefact-of-the-random-directions).

{{table: results/macro/tables/macro_table7_stability.csv | **Table 6.** The headline re-run over thirty independent draws of the `L = 60` directions and of the network initialisation, data held fixed.}}

It does, emphatically: **all thirty draws reject**, the median p-value is below
0.001, and the merged (conservative) p-value is 0.000. Contrast this with the
price–volume application, where the corresponding check found only 7% of draws
rejecting — that is the difference between a real effect and a lucky seed.

{{figure: macro_fig9_stability.png | **Figure 6.** Distribution of the p-value across thirty direction draws. Everything is in the rejection band.}}

{{figure: macro_fig6_bootstrap_null.png | **Figure 7.** The bootstrap null distribution of `KS_n` for the headline test, with the observed statistic in terracotta. This is the figure that makes a p-value auditable.}}

{{figure: macro_fig7_empirical_process.png | **Figure 8.** Real and imaginary parts of the empirical process at each of the sixty random directions, against the pointwise bootstrap envelope. Many directions escape the band — the departure from the null is broad, not an artefact of one lucky direction. Compare this with the price–volume case, where exactly one direction of twenty escaped.}}

{{figure: macro_fig8_mdn_fit.png | **Figure 9.** Mixture density network diagnostic: pooled draws from the fitted conditional density against the empirical marginal of the lagged cause. A visible mismatch here would mean `G` is too small and the type I error is inflated.}}

## 7. Caveats

<div class="warn"><strong>Five things to hold in mind</strong>
<p><b>Stationarity.</b> Four of the eight transformed series fail the KPSS screen
on the pooled sample. The sub-sample results in Section 4 rest on firmer ground
than the pooled results in Section 3.</p>
<p><b>Multiple testing.</b> Section 3 is 72 tests at the 5% level; roughly four
false rejections are expected by chance. The argument here rests on
<i>patterns</i> — consistent significance across adjacent lag orders and across
sub-samples — not on any single cell. Apply Holm or Benjamini–Hochberg within a
direction if you need a formal correction.</p>
<p><b>Bivariate only.</b> Every test conditions on two series. Oil prices and
inflation both respond to global demand; monetary policy and output both respond
to fiscal shocks. Version 1.0 of the package cannot condition on a third series,
which the paper itself flags as future work.</p>
<p><b>Granger, not structural.</b> These are forecasting statements. They say
nothing about what would happen under an intervention.</p>
<p><b>Real-time versus revised.</b> FRED serves revised data. A forecaster in
1975 did not see these numbers. Nothing here is a claim about real-time
predictability.</p></div>

## 8. Reproducing and extending

```bash
# the full study, about ten minutes on twelve cores
python scripts/run_macro_application.py --jobs -1

# a coarser, faster pass
python scripts/run_macro_application.py --lags 1 3 6 12 --skip-subsamples --jobs -1

# refresh the FRED data or add series
python data/fetch_macro.py --add GDPC1 TB3MS
```

Your own relation, in five lines:

```python
from drgct import drgc_lag_scan
from drgct.datasets import load_macro

m = load_macro()
scan, _ = drgc_lag_scan(m["WTI oil price"].to_numpy(),
                        m["CPI inflation"].to_numpy(),
                        lags=[1, 3, 6, 12], L=60, B=999, seed=1)
```

The whole grid, including your own series, is one call:

```python
from drgct.applications import macro_study

df = macro_study(
    relations=[("M2 money growth", "Nonfarm payrolls", "Money to jobs", "")],
    periods=["Full sample"], lags=[1, 6, 12], drgc_kwargs=dict(L=60, B=999),
    n_jobs=-1,
)
```

## Data source

All series from **FRED**, Federal Reserve Bank of St. Louis, downloaded by
[`data/fetch_macro.py`](https://github.com/merwanroudane/DRGCT/blob/main/data/fetch_macro.py)
and bundled with the package so that everything here runs offline. Provenance
is recorded in
[`src/drgct/data/SOURCES_MACRO.md`](https://github.com/merwanroudane/DRGCT/blob/main/src/drgct/data/SOURCES_MACRO.md).
FRED series are in the public domain unless an individual series page says
otherwise.

## References

- Friedman, M. (1961). The lag in effect of monetary policy. *Journal of Political Economy* 69, 447–466.
- Hamilton, J. D. (1983). Oil and the macroeconomy since World War II. *Journal of Political Economy* 91, 228–248.
- Hui, Y., Liu, C. and Song, X. (2025). *Deep learning based doubly robust test for Granger causality.* [arXiv:2509.15798](https://arxiv.org/abs/2509.15798).
- McConnell, M. M. and Perez-Quiros, G. (2000). Output fluctuations in the United States: what has changed since the early 1980's? *American Economic Review* 90, 1464–1476.
- Okun, A. M. (1962). Potential GNP: its measurement and significance. *Proceedings of the Business and Economic Statistics Section*, American Statistical Association.
- Stock, J. H. and Watson, M. W. (2002). Has the business cycle changed and why? *NBER Macroeconomics Annual* 17, 159–218.
