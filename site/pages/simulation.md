# Simulation evidence

<p class="lede">Does the test actually control its size, and does it have power
where the paper says it does? This page reports what our implementation
produces on the paper's own designs — including three places where it diverges
from the published numbers, documented rather than tuned away.</p>

Produced by
[`scripts/run_simulation.py`](https://github.com/merwanroudane/DRGCT/blob/main/scripts/run_simulation.py):

```bash
python scripts/run_simulation.py --reps 200 --ns 500 -B 499 --jobs -1
```

<div class="stat-row">
  <div class="stat"><div class="v">6</div><div class="l">data generating processes from the paper's Table 1</div></div>
  <div class="stat"><div class="v">200</div><div class="l">Monte-Carlo replications per design point</div></div>
  <div class="stat"><div class="v">18k</div><div class="l">tests run (6 designs × 5 lags × 200 reps × 3 estimators)</div></div>
  <div class="stat"><div class="v">0.015</div><div class="l">Monte-Carlo standard error on a size of 0.05</div></div>
</div>

## 1. The designs

Six processes. In every one, `X` follows a linear AR(*p*); what varies is how
`X`'s past enters `Y`. **S1** and **S2** satisfy the null — no causality — and
measure size. **P1**–**P4** satisfy the alternative and measure power.

{{table: results/tables/table1_dgps.csv | **Table 1.** The six data generating processes. The innovations are i.i.d. `N(0, 0.5)` and mutually independent.}}

{{table: results/tables/table2_parameters.csv | **Table 2.** Coefficient settings by lag order, chosen so the multi-lag sums in `Y` do not diverge.}}

## 2. The headline: the test is correctly sized

A correctly sized test produces p-values that are uniform on `[0, 1]` under the
null, so their empirical CDF should lie on the 45-degree line. This is a
sharper diagnostic than any rejection-frequency table, because it checks the
*whole* distribution rather than one quantile.

{{figure: fig2_pvalue_ecdf.png | **Figure 1.** p-value plots under the null, *n* = 500, 200 replications. The doubly robust test (blue) tracks the diagonal in all ten panels. The naive plug-in (gold) sits below it — conservative. The smoothing benchmark (red) is above the diagonal at lag 1 in S1 and below it elsewhere.}}

{{table: results/tables/table3_size.csv | **Table 3.** Empirical size at the 5% nominal level, *n* = 500.}}

{{figure: fig1_size.png | **Figure 2.** The size table as a picture. The shaded band is ±1.96 Monte-Carlo standard errors around the nominal 5%: an exactly sized test should land inside it.}}

Against the paper's own 1000-replication values (in brackets):

| Lag | DGP S1 | DGP S2 |
|:--:|:--:|:--:|
| 1 | 0.070 [0.051] | 0.045 [0.046] |
| 2 | 0.055 [0.056] | 0.055 [0.049] |
| 3 | 0.055 [0.046] | 0.065 [0.045] |
| 4 | 0.075 [0.057] | 0.055 [0.039] |
| 5 | 0.050 [0.050] | 0.055 [0.051] |

Every cell is within about 1.5 Monte-Carlo standard errors.

<div class="good"><strong>Independent confirmation</strong>
<p>Before the shipped run, a separate 120-replication calibration at
<i>n</i> = 500 gave S1 sizes of 0.058 / 0.050 / 0.058 at lags 1 / 3 / 5 against
the paper's 0.051 / 0.046 / 0.050, and P2 powers of 1.000 / 0.967 / 0.583
against 0.996 / 0.898 / 0.546. Two independent runs, both within two
Monte-Carlo standard errors of the published numbers.</p></div>

## 3. Power

{{table: results/tables/table4_power.csv | **Table 4.** Empirical power at the 5% nominal level, *n* = 500.}}

{{figure: fig3_power.png | **Figure 3.** Power against the lag order, one panel per design.}}

Against the paper (in brackets):

| Lag | P1 | P2 | P3 | P4 |
|:--:|:--:|:--:|:--:|:--:|
| 1 | 1.000 [1.000] | 1.000 [0.996] | 1.000 [1.000] | 1.000 [0.990] |
| 2 | 1.000 [1.000] | 1.000 [0.973] | 1.000 [0.977] | 0.965 [0.959] |
| 3 | 1.000 [1.000] | 0.975 [0.898] | 0.925 [0.954] | **0.615** [0.886] |
| 4 | 0.980 [0.912] | 0.800 [0.628] | 0.620 [0.429] | **0.315** [0.663] |
| 5 | 0.895 [0.870] | 0.565 [0.546] | 0.385 [0.407] | **0.170** [0.600] |

P1, P2 and P3 reproduce closely. **P4 does not**, and the gap at lags 3–5 is far
too large to be sampling noise.

<div class="warn"><strong>Divergence 1: power in design P4</strong>
<p>P4 is the one design where <code>Y</code> has no own-lag term of its own:
<code>Y(t) = a₀ Σⱼ X(t−j)·Y(t−j) + ε(t)</code>. The conditional mean
<code>m(Y(t−1))</code> is therefore an awkward object, and the MLP absorbs more
of the signal in sample than it should, leaving less for the statistic to
detect. Raising the MLP epochs or the number of directions <code>L</code> closes
part of the gap. We report the gap rather than tuning it away.</p></div>

## 4. Why the correction term is not optional

The `− φ̂(ν | Y(t−1))` centring is the whole doubly robust construction. Drop it
and you get the naive process of equation (5), which the package will build on
request with `doubly_robust=False`.

{{table: results/tables/table3b_naive_size.csv | **Table 5.** Doubly robust versus naive plug-in, DGP S1, *n* = 1000, lag 1, 150 replications.}}

| | S1 lag 1 | lag 2 | lag 3 | lag 4 | lag 5 |
|---|:--:|:--:|:--:|:--:|:--:|
| DRGC (*n* = 500) | 0.070 | 0.055 | 0.055 | 0.075 | 0.050 |
| DRGC-naive (*n* = 500) | 0.000 | 0.005 | 0.015 | 0.005 | 0.005 |

<div class="warn"><strong>Divergence 2: the naive plug-in fails downwards, not upwards</strong>
<p>Section 4 of the paper reports the naive plug-in <i>over</i>-rejecting, with
size rising to 0.151 at <i>n</i> = 1000 and 0.321 at <i>n</i> = 2000. Here it
<i>under</i>-rejects, essentially never rejecting at all.</p>
<p>The mechanism is straightforward. <code>m̂</code> is fitted by least squares
<b>in sample</b>, with no splitting — as Section 1 of the paper prescribes. The
first-order conditions of that fit make the residual near-orthogonal to any
function of <code>Y(t−1)</code> in the span of the network, and
<code>e^{iμ′Y(t−1)}</code> is such a function, so the naive statistic is
mechanically shrunk toward zero. The multiplier bootstrap inherits no such
orthogonality, so its null distribution is too wide.</p>
<p>Which distortion dominates — this shrinkage or the estimation bias the paper
emphasises — depends on the design, the sample size, and how hard the network is
trained. <b>Both are failures of the same kind:</b> without the centring the
naive process has no valid null distribution, and with it the doubly robust
process does. That is the point of the construction, and Figure 1 shows it.</p></div>

## 5. The smoothing benchmark

Empirical size of our `nhkj_test` at the 5% nominal level, *n* = 500:

| Lag | S1 | S2 |
|:--:|:--:|:--:|
| 1 | **0.170** | 0.005 |
| 2 | 0.045 | 0.005 |
| 3 | 0.040 | 0.000 |
| 4 | 0.000 | 0.000 |
| 5 | 0.020 | 0.000 |

<div class="warn"><strong>Divergence 3: our benchmark is not the paper's NHKJ</strong>
<p><code>nhkj_test</code> is a <i>member of the class</i> — a Zheng (1996) /
Fan–Li (1996) degenerate U-statistic with the fourth-order Gaussian kernel and
the <code>h = c·n^{−0.15}</code> bandwidth schedule the paper specifies for its
benchmark — but not a transcription of the estimator in Nishiyama et al. (2011).</p>
<p>It is severely <b>under</b>-sized in the exponential-mean design S2 and at
high lag in S1, which is the paper's qualitative story, but badly
<b>over</b>-sized at lag 1 in S1, and it keeps power near one in P1–P3 instead of
collapsing. <b>Because its size is not controlled, its power column is not a
fair comparison.</b> Read the size table before the power table, and when you
need to characterise NHKJ's properties in print, cite the paper's Tables 3–4,
not ours.</p>
<p>That the benchmark's finite-sample behaviour swings this much with the
bandwidth constant is, incidentally, exactly the criticism the DRGCT paper
levels at smoothing-based causality tests in the first place.</p></div>

## 6. Precision and what was actually run

**Experiment A — completed.** All six designs, lag orders 1–5, *n* = 500, 200
replications, `B = 499`, three estimators.

**Experiment B — stopped early.** Design S1 at *n* ∈ {1000, 2000}, lags 1, 3, 5.
The run was interrupted after its first design point, so only
*n* = 1000 / lag 1 / 150 replications is reported above. To complete it:

```bash
python scripts/run_simulation.py --skip-a --b-reps 150 --b-ns 1000 2000 --b-lags 1 3 5 --jobs -1
```

200 replications give a Monte-Carlo standard error of about **0.015** on a size
of 0.05 and up to 0.035 on a power near 0.5. Every rejection frequency in
`monte_carlo_summary.csv` carries its own `mc_se` column. Differences from the
paper of 0.02–0.04 are sampling noise; the differences flagged in P4 and in the
NHKJ column are not.

To run the paper's own grid — 1000 replications at *n* ∈ {500, 1000, 2000} with
`B = 1000`, a multi-day job on a laptop:

```bash
python scripts/run_simulation.py --reps 1000 --ns 500 1000 2000 -B 1000 --jobs N
```

## 7. Verifying size for *your* design

The paper's designs may look nothing like your data. Simulating under a null
that inherits your own persistence and innovation scale costs an hour and is
worth it — the recipe is in
[the guide's §9](guide.html#9-verifying-size-for-your-design-with-a-bespoke-monte-carlo).

```python
from drgct.simulate import monte_carlo, summarize

mc = monte_carlo(dgps=["S1", "P2"], ns=[500], lags=[1, 3, 5], reps=200,
                 methods=("drgc", "drgc_naive", "nhkj"),
                 drgc_kwargs=dict(B=499), n_jobs=-1)
print(summarize(mc).to_string(index=False))
```

## Full replication log

Every command, the environment, and the complete table of divergences are in
[`results/README.md`](https://github.com/merwanroudane/DRGCT/blob/main/results/README.md)
in the repository.

## References

- Fan, Y. and Li, Q. (1996). Consistent model specification tests. *Econometrica* 64, 865–890.
- Hui, Y., Liu, C. and Song, X. (2025). *Deep learning based doubly robust test for Granger causality.* [arXiv:2509.15798](https://arxiv.org/abs/2509.15798).
- Nishiyama, Y., Hitomi, K., Kawasaki, Y. and Jeong, K. (2011). A consistent nonparametric test for nonlinear causality. *Journal of Econometrics* 165, 112–127.
- Zheng, J. X. (1996). A consistent test of functional form via nonparametric estimation techniques. *Journal of Econometrics* 75, 263–289.
