# FAQ

Short answers. Long answers are in [`GUIDE.md`](GUIDE.md) and
[`THEORY.md`](THEORY.md).

---

### What exactly is the null hypothesis?

`X` does not Granger-cause `Y` **in mean**:
`E[Y_t | X_{t−1},…,X_{t−p}, Y_{t−1},…,Y_{t−q}] = E[Y_t | Y_{t−1},…,Y_{t−q}]`.
The test says nothing about causality in variance, in quantiles or in
distribution, and — as always with Granger causality — nothing about
structural causality in the sense of an intervention.

### Why "doubly robust"? Nothing is being estimated causally.

The term is borrowed from the missing-data and treatment-effect literature
(Robins, Rotnitzky and Zhao, 1994) and refers to the *algebraic* structure of
the moment condition, not to a causal-inference design. Two nuisance functions
enter — the conditional mean `m` and the conditional characteristic function
`φ` — and the bias of the test statistic is proportional to the **product** of
their estimation errors. Each may converge slower than `n^{−1/2}`; the product
still vanishes fast enough. That is what makes deep networks admissible in a
classical testing framework.

### Do I need a GPU?

No. The networks are tiny (one hidden layer of width `5·lag` by default) and
CPU is faster than GPU for them, because kernel-launch overhead dominates.
Pass `mlp=MLPConfig(device="cuda")` if you insist.

### How long does one test take?

Roughly 1.5 s at `n = 500, lag = 1`, and 8–12 s at `n = 750, lag = 10` on a
laptop CPU. Almost all of it is network training. The bootstrap is
milliseconds regardless of `B`, because it re-uses the fitted quantities.

### Why is `B` free but `L` is not?

`B` only redraws the multipliers and multiplies a stored `(n_eff, L)` matrix.
`L` adds a column to that matrix *and* to the MDN pseudo-sample projection —
still cheap, just not free. Neither touches the networks. Use `B = 999` or
`1999` always; raise `L` to 50–100 whenever `p + q ≥ 10`.

### My p-value changes when I change the seed. Is the test broken?

No, and this is the single most important practical caveat. Step 2(d) of
Algorithm 1 draws `L` directions `(μ_ℓ, ν_ℓ)` at random. Each draw yields a
*valid* test, but with `L = 20` in a 20-dimensional conditioning space, the
draws differ enough that the p-value carries simulation noise on top of
sampling noise. Run `drgc_stability(x, y, lag, n_draws=30)`, report the
distribution, and quote the `merged_pvalue` — `min(1, 2 × median)`, valid
under arbitrary dependence by Rüger's inequality. Raising `L` shrinks the
noise directly.

In the bundled application this mattered: one draw put the S&P 500 lag-10
test at `p = 0.008` while the median across 30 draws was 0.52 and only 6.7%
of draws rejected. Reporting the first number alone would have been wrong.

### How do I choose `p` and `q`?

Set them generously — that is the entire point of the method. Kernel-based
causality tests break down past two or three lags; the DRGCT does not.
Scan the range with `drgc_lag_scan` and report the profile rather than
pre-selecting one lag. For daily financial data 10 is ample; for monthly macro
data with policy lags, 12–18.

### `p ≠ q`?

Allowed. `drgc_test(x, y, p=12, q=6)`. The paper maintains `p ≤ q`; when
`p > q` the effective sample starts at `max(p, q) + 1` instead.

### What should `G` be?

10 unless you have a reason. Lemma 3 makes the trade-off explicit: too small
and the mixture cannot approximate `f_{X|Y}` (bias → **inflated type I
error**); too large and the estimate is noisy (variance in `KS_n`).
Cross-validate over `{5, 10, 15, 20}` when `p ≥ 5`, and always look at
`plot_mdn_fit`.

### Does standardising the data change the result?

Not the p-value. Rescaling `Y` multiplies `KS_n` and every bootstrap replicate
`KS*_n` by the same constant, so the rank of the observed statistic in its
bootstrap distribution — and hence the p-value — is unchanged. It does change
how well the networks train, which is why `standardize=True` is the default.

### Should I use `doubly_robust=False`?

Only to demonstrate why you should not. It builds the naive process (5),
which has no valid null distribution. The paper documents this as
over-rejection (empirical sizes of 0.151 at `n = 1000` and 0.321 at
`n = 2000` where the true size is 5%). In this implementation the failure runs
the other way — 0.000 to 0.020 at `n = 500`, 0.000 at `n = 1000` — because the
in-sample least-squares residual is near-orthogonal to functions of
`Y_{t−1}`, shrinking the naive statistic while leaving its bootstrap null
unchanged. Either way the naive test is not correctly sized. See
`results/tables/table3_size.*`, `table3b_naive_size.*` and
[`results/README.md`](../results/README.md).

### Can I test three or more series?

Not in version 1.0. The paper is explicit that the bivariate case is what is
proved, and flags multivariate extension (`X_t` and `Z_t` jointly causing
`Y_t`) and panel data as future work. A pragmatic interim approach is to run
pairwise tests and be candid that you have not conditioned on the third
series.

### Can I test causality in variance, or in quantiles?

Not with this test. It targets the conditional **mean**. For causality in risk
see Hong, Liu and Wang (2009); for quantile causality, the quantile-regression
literature.

### Does it handle non-stationary data?

No. Assumption 1 requires strict stationarity and exponential β-mixing.
Transform first, and verify with `check_stationarity` (which requires ADF to
reject *and* KPSS not to reject). Cointegrated levels need a VECM-style
framing that is outside this package's scope.

### What about missing values and irregular sampling?

Merge on the date index with an inner join **before** differencing, and drop
non-finite rows. Do not forward-fill: an imputed holiday becomes an artificial
zero return that the test reads as structure. Irregular sampling breaks the
fixed-lag structure entirely.

### How small can `n` be?

The code refuses fewer than 30 usable observations. Realistically you want at
least 30 observations per lag, so `n ≈ 300` for lag 10. Below `n ≈ 200` the
networks have too little to learn from and power collapses.

### The p-value printed as 0.0000. Do I write "p = 0"?

No. It means `p < 1/B`. With `B = 999`, write `p < 0.001`. Raise `B` to 9999
if you need a finer number.

### Why is the smoothing benchmark rejecting so rarely?

Because smoothing tests lose size control once the conditioning dimension
exceeds two or three — Table 3 of the paper reports NHKJ sizes between 0.003
and 0.043 at lag ≥ 2 against a 5% nominal level. That is the curse of
dimensionality, and it is the problem the DRGCT exists to solve. Read a NHKJ
non-rejection at lag ≥ 3 as weak evidence, and say so in the paper.

Note that our own `nhkj_test` is undersized in some cells (0.000–0.005 in the
exponential-mean design S2) and *over*-sized in others (0.170 at lag 1 in S1),
and unlike the paper's NHKJ it does not lose power as the lag order grows.
Since its size is not controlled, do not read its power numbers as a fair
comparison — see [`results/README.md`](../results/README.md).

### Is `nhkj_test` the actual test from Nishiyama et al. (2011)?

No — it is a member of that class, not a line-by-line transcription. It
implements a Zheng (1996) / Fan–Li (1996) degenerate U-statistic conditional
moment test with the fourth-order Gaussian kernel and `h = c·n^{−0.15}`
bandwidth schedule the DRGCT paper specifies for its benchmark. Its
finite-sample behaviour in our runs differs from the paper's NHKJ (see the
previous answer), which is itself informative: smoothing tests of this class
are very sensitive to the bandwidth. Describe it in a paper as "a
smoothing-based nonparametric benchmark", and cite the paper's own numbers
when characterising NHKJ.

### Why do the bundled application results differ from Table 6 of the paper?

Two reasons, both documented in [`results/README.md`](../results/README.md):
the paper does not name its data vendor (volume series in particular differ
substantially across vendors), and the CSI 300 file here uses the
exchange-traded tracker `510300.SS` because Yahoo truncates the index series.
The Nikkei 225 and CSI 300 findings agree with the paper qualitatively; the
S&P 500 does not. Swap in your vendor's CSV — same two columns — and every
script runs unchanged.

### Are the shipped simulation results the paper's full grid?

No. The paper uses 1000 replications at `n ∈ {500, 1000, 2000}` with
`B = 1000`; the shipped tables use fewer, so that the run finishes in hours
rather than days. Every rejection frequency is reported with its Monte-Carlo
standard error, and `results/README.md` records the exact commands. Run
`python scripts/run_simulation.py --reps 1000 --ns 500 1000 2000 -B 1000 --jobs N`
for the paper's grid.

### Is the package deterministic?

Given a seed and a fixed machine and PyTorch version, yes — bit-identical.
Across machines or PyTorch versions, floating-point non-determinism can move
the p-value in the third decimal. Never build an argument on `p = 0.0499`.

### How do I cite this?

Cite the paper and, if it helped, the software:

```python
import drgct
print(drgct.cite("bibtex"))
```

### Where do I get it, and where do I report a bug?

Install from PyPI: <https://pypi.org/project/drgct/> (`pip install drgct`).
Report bugs at <https://github.com/merwanroudane/DRGCT/issues>. Please include
the output of

```python
import sys, torch, numpy, drgct
print(drgct.__version__, sys.version, torch.__version__, numpy.__version__)
```

along with a minimal reproducible example.
