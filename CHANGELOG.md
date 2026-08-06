# Changelog

All notable changes to `drgct`. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-08-06

First public release: a complete implementation of

> Hui, Y., Liu, C. and Song, X. (2025). *Deep learning based doubly robust
> test for Granger causality.* [arXiv:2509.15798v2](https://arxiv.org/abs/2509.15798)

### The test

- `drgc_test` — Algorithm 1 in full: an MLP for the conditional mean
  `m(Y_{t−1})`, a mixture density network for the conditional characteristic
  function `φ(ν | Y_{t−1})`, the doubly robust empirical process (8)–(9), the
  Kolmogorov–Smirnov statistic (10), and the multiplier bootstrap (12), which
  reuses the fitted quantities so the networks are trained once per test.
- `drgc_lag_scan`, `drgc_both_directions` — the lag-order and two-directional
  workflows the paper's Section 5 uses.
- `drgc_stability` — **addition beyond the paper.** Re-runs the test across
  independent draws of the `L` random directions and reports the p-value
  distribution plus a Rüger/Vovk–Wang merged p-value. Necessary in practice:
  with `L = 20` in a 20-dimensional conditioning space, individual draws can
  land on either side of 0.05.
- `doubly_robust=False` builds the naive process (5) so the size failure it
  produces can be reproduced directly.

### Around it

- `nhkj_test` — smoothing-based nonparametric benchmark (Zheng 1996 / Fan–Li
  1996 degenerate U-statistic, fourth-order Gaussian kernel,
  `h = c·n^{−0.15}`).
- `drgct.dgp`, `drgct.simulate` — the six designs of Table 1 with the Table 2
  coefficients, and a parallel Monte Carlo reproducing Tables 3–4.
- `drgct.datasets` — bundled daily price and volume series for the S&P 500,
  CSI 300 and Nikkei 225 over the paper's window (27 Sep 2019 – 26 Sep 2024),
  plus the Section 5 percentage-change transformation and stationarity screen.
- `drgct.applications` — the 180-test price–volume grid of Section 5, plus
  rolling-window causality.
- `drgct.tables` — booktabs LaTeX, GitHub Markdown and CSV in one call.
- `drgct.plots` — thirteen journal-quality figures, vector PDF plus 400 dpi PNG.
- `drgct` command line: `test`, `simulate`, `app`, `info`.

### Documentation

`docs/GUIDE.md` (applied researcher's guide, raw data to write-up),
`docs/SYNTAX.md` (complete API reference), `docs/THEORY.md`
(equation-by-equation map from paper to code), `docs/FAQ.md`, and four
runnable examples.

### Known divergences from the paper

Documented rather than tuned away, in
[`results/README.md`](https://github.com/merwanroudane/DRGCT/blob/main/results/README.md):

- the naive plug-in under-rejects here rather than over-rejecting;
- `nhkj_test` does not reproduce the paper's NHKJ finite-sample behaviour, so
  its power numbers are not a fair comparison;
- DGP P4 power is below the paper's at lags 3–5;
- the bundled market data are not the paper's (unnamed) vendor's.

[1.0.0]: https://github.com/merwanroudane/DRGCT/releases/tag/v1.0.0
