<div class="hero" markdown="1">

# drgct

<p class="sub">A complete, documented Python implementation of the <b>deep-learning based doubly robust test for Granger causality</b> — with real economic applications, journal-quality output, and every divergence from the source paper written down rather than hidden.</p>

<div class="badges">
<a href="https://pypi.org/project/drgct/"><img alt="PyPI" src="https://img.shields.io/pypi/v/drgct?color=1F4E79&label=PyPI&logo=pypi&logoColor=white"></a>
<a href="https://pypi.org/project/drgct/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/drgct?logo=python&logoColor=white"></a>
<a href="https://github.com/merwanroudane/DRGCT"><img alt="GitHub" src="https://img.shields.io/badge/source-GitHub-1E2A32?logo=github"></a>
<a href="https://arxiv.org/abs/2509.15798"><img alt="arXiv" src="https://img.shields.io/badge/paper-arXiv%3A2509.15798-B4462F"></a>
<a href="https://github.com/merwanroudane/DRGCT/blob/main/LICENSE"><img alt="MIT" src="https://img.shields.io/badge/licence-MIT-3E7B5E"></a>
</div>

</div>

```bash
pip install drgct
```

The method is due to **Hui, Y., Liu, C. and Song, X. (2025)**, *Deep learning
based doubly robust test for Granger causality*,
[arXiv:2509.15798v2](https://arxiv.org/abs/2509.15798). This site documents an
independent open-source implementation, built and maintained by
**Dr Merwan Roudane**.

<div class="stat-row">
  <div class="stat"><div class="v">10+</div><div class="l">lag orders you can actually test, where kernel methods stop at two or three</div></div>
  <div class="stat"><div class="v">0.05</div><div class="l">empirical size at the 5% nominal level, at every lag we checked</div></div>
  <div class="stat"><div class="v">~2 s</div><div class="l">per test at <i>n</i> = 500 on a laptop CPU — no GPU needed</div></div>
  <div class="stat"><div class="v">260+</div><div class="l">tests run on real economic data for the two applications on this site</div></div>
</div>

## Where to start

<div class="grid two">
  <a class="card" href="guide.html">
    <h3>Applied guide →</h3>
    <p>From a raw CSV to a finished results section: design decisions, complete runnable code at every step, a reporting checklist and a write-up template.</p>
  </a>
  <a class="card" href="macro.html">
    <h3>US macro application →</h3>
    <p>Sixty years of monthly FRED data. Oil shocks, monetary transmission, the quantity theory and Okun's law, tested at lags of one to eighteen months.</p>
  </a>
  <a class="card" href="theory.html">
    <h3>Theory →</h3>
    <p>Every equation of the paper mapped to the exact line of code, the assumptions spelled out, and each open choice documented.</p>
  </a>
  <a class="card" href="api.html">
    <h3>API reference →</h3>
    <p>Every public function, every argument, every return field, with worked examples.</p>
  </a>
</div>

## The problem this solves

Granger causality testing has a standing dilemma.

The linear VAR *F*-test is cheap, correctly sized and completely blind to
nonlinearity. Kernel-smoothing nonparametric tests see nonlinearity, but past
two or three lags they collapse under the curse of dimensionality: they lose
size control, lose power, and become hostage to a bandwidth choice.

Yet economic mechanisms routinely operate with *long and variable lags* —
monetary policy on output, energy prices on the consumer basket, credit
conditions on investment. Testing them at lag 1 answers a question nobody
asked.

|                                   | Linear VAR *F* | Kernel / smoothing | **DRGCT** |
|-----------------------------------|:--:|:--:|:--:|
| Detects nonlinear causality       | ✗ | ✓ | ✓ |
| Usable at lag 5–18                | ✓ | ✗ | ✓ |
| Bandwidth to tune                 | – | yes, and it matters | none |
| Size control at large lag         | ✓ | poor | ✓ |
| Local power rate                  | `n^{-1/2}` | slower | `n^{-1/2}` |
| Cost per test                     | ms | s | s to tens of s |

## How it works, in one screen

Let `W(t-1)` collect `p` lags of `X` and `q` lags of `Y`, and let
`m(Y(t-1)) = E[Y(t) | Y(t-1), …, Y(t-q)]` be the best forecast of `Y` from its
own past alone. The null is that adding `X`'s past changes nothing:

<div class="eq"><span class="tag">(2)</span>
<code>H₀ :  E[ Y(t) − m(Y(t−1)) | W(t−1) ] = 0   almost surely</code>
</div>

Weighting by the generically comprehensively revealing family
`exp(i·w′W)` of Stinchcombe and White (1998) turns that conditional
restriction into an unconditional one, and Proposition 1 of the paper shows it
is equivalent to the **doubly robust** form:

<div class="eq"><span class="tag">(6)</span>
<code>H₀ :  E[ (Y(t) − m(Y(t−1))) · e^{iμ′Y(t−1)} · ( e^{iν′X(t−1)} − φ(ν | Y(t−1)) ) ] = 0</code>
</div>

with `φ(ν | Y(t−1)) = E[e^{iν′X(t−1)} | Y(t−1)]`. Estimate `m` with a
multilayer perceptron and `φ` with a mixture density network, evaluate the
resulting empirical process at `L` random directions, take the
Kolmogorov–Smirnov functional of its real and imaginary parts, and get
critical values from a multiplier bootstrap.

<div class="note"><strong>Why the correction term is the whole point</strong>
<p>Drop <code>φ̂</code> and the bias is first order in the neural-network error,
so the process has no valid limiting distribution. Keep it and the bias depends
on the <b>product</b> of the two estimation errors. Two estimators that each
converge slower than <code>n^{−1/2}</code> still multiply into something fast
enough — which is exactly what lets deep networks live inside a classical
testing framework.</p></div>

Because only the bootstrap multipliers are redrawn, the two networks are
trained **once per test**: `B = 999` costs milliseconds against seconds for the
networks. Never economise on `B`.

Full derivation and the line-by-line source map: [Theory](theory.html).

## Sixty seconds

```python
import numpy as np
from drgct import drgc_test

rng = np.random.default_rng(0)
n = 600

# X is an AR(1).  Y depends on X only through sin(.), so the linear
# projection of Y on lagged X is essentially zero: a VAR F-test sees nothing.
x = np.zeros(n); y = np.zeros(n)
e1, e2 = rng.normal(0, 0.7, n), rng.normal(0, 0.7, n)
for t in range(1, n):
    x[t] = -0.5 * x[t - 1] + e1[t]
    y[t] =  0.5 * y[t - 1] + np.sin(-x[t - 1]) + e2[t]

drgc_test(x, y, lag=1, seed=1).print()
```

```
  H0 : X does not Granger-cause Y in mean
  Lag orders          : p = 1 (cause), q = 1 (effect)
  Sample              : n = 600,  effective n - q = 599
  KS_n statistic      : 8.271043
  Bootstrap p-value   : 0.0000 ***
  Critical values     : 1%: 1.7654  5%: 1.5361  10%: 1.4224
  Decision at 5%      : REJECT H0
```

Scan every lag order in both directions, and get a tidy table back:

```python
from drgct import drgc_lag_scan
scan, results = drgc_lag_scan(x, y, lags=range(1, 11), seed=1)
```

## Does it work? Yes — and here is the evidence

The single most informative size diagnostic is the p-value plot: under the
null, a correctly sized test produces p-values that are uniform on `[0, 1]`, so
their empirical CDF should lie on the 45-degree line.

{{figure: fig2_pvalue_ecdf.png | **The doubly robust test is correctly sized.** Empirical CDF of the bootstrap p-value across 200 replications of the paper's two null designs, *n* = 500. The DRGCT (blue) tracks the diagonal at every lag order. The naive deep plug-in (gold) and the smoothing benchmark (red) do not. Full detail on the <a href="simulation.html">simulation page</a>.}}

## Two real applications

<div class="grid two">
  <a class="card" href="macro.html">
    <h3>US macroeconomics, 1959–2025 →</h3>
    <p>Eight monthly FRED series, six relations, both directions, lags of 1 to 18 months, plus a Great Inflation versus Great Moderation split. Oil prices Granger-cause inflation at <i>p</i> &lt; 0.001 out to six months; industrial production drives unemployment; monetary transmission is weak in the pooled sample and concentrated in the pre-1984 era.</p>
  </a>
  <a class="card" href="finance.html">
    <h3>Price–volume in three markets →</h3>
    <p>The paper's own application: 180 tests across the S&amp;P 500, CSI 300 and Nikkei 225, three overlapping sub-samples and ten lag orders — plus a rolling-window extension the paper points to but does not carry out.</p>
  </a>
</div>

## The one caveat you must not skip

Step 2(d) of the algorithm draws the `L` evaluation directions **at random**.
Every draw is a valid test, but with `L = 20` in a twenty-dimensional
conditioning space, the draws differ enough that the p-value carries simulation
noise on top of sampling noise.

This is not hypothetical. In the price–volume application, one seed put a test
at `p = 0.008`; across thirty independent draws the median was **0.52** and only
**7%** of draws rejected.

```python
from drgct import drgc_stability
stab = drgc_stability(x, y, lag=10, n_draws=30, seed=1)
stab["median"], stab["q05"], stab["q95"], stab["merged_pvalue"]
```

Three remedies, in order: **raise `L`** (nearly free, and it also raises
power); **report the distribution**, not one draw; **quote `merged_pvalue`**,
which is `min(1, 2 × median)` and is a valid — conservative — p-value under
arbitrary dependence.

`drgc_stability` is an addition to the paper, not part of it. It exists because
running the method at scale on real data made the issue impossible to ignore.

## What is in the package

<div class="grid three">
  <div class="card"><h3>The test</h3><p><code>drgc_test</code>, <code>drgc_lag_scan</code>, <code>drgc_both_directions</code>, <code>drgc_stability</code> — Algorithm 1 in full.</p></div>
  <div class="card"><h3>Networks</h3><p><code>MLPConfig</code>, <code>MDNConfig</code> — the paper's architecture, or the theoretical rates of Lemmas 1–2.</p></div>
  <div class="card"><h3>Benchmark</h3><p><code>nhkj_test</code> — a smoothing-based nonparametric comparator.</p></div>
  <div class="card"><h3>Simulation</h3><p><code>simulate_dgp</code>, <code>monte_carlo</code> — the six designs of Table 1 and a parallel replication driver.</p></div>
  <div class="card"><h3>Data</h3><p><code>load_index</code>, <code>load_macro</code> — bundled daily market and monthly FRED series, offline.</p></div>
  <div class="card"><h3>Output</h3><p><code>drgct.tables</code>, <code>drgct.plots</code> — booktabs LaTeX, Markdown, CSV; thirteen figures as vector PDF and PNG.</p></div>
</div>

There is also a command line:

```bash
drgct info
drgct test data.csv -x credit -y gdp --lag-scan --lag-max 12 --save
drgct simulate --dgps S1 P2 --reps 200 --jobs 10
drgct app --indices spx500 --lag-max 10 --jobs 10
```

## Honest limitations

<div class="warn"><strong>Read before you cite results from this implementation</strong>
<p>Four things diverge from the source paper, all documented in full on the
<a href="simulation.html">simulation page</a>: the naive plug-in fails
<i>downwards</i> here rather than upwards; the smoothing benchmark does not
reproduce the paper's NHKJ finite-sample behaviour, so its power column is not
a fair comparison; power in design P4 is below the paper's at high lag; and the
bundled market data are not the paper's (unnamed) vendor's.</p></div>

The test is also bivariate and targets the conditional **mean** only. It says
nothing about causality in variance or in quantiles, and — as always with
Granger causality — nothing about structural causality in the sense of an
intervention.

## Citation

Cite the paper. Cite the software too, if it saved you time.

```bibtex
@article{HuiLiuSong2025DRGCT,
  title   = {Deep learning based doubly robust test for Granger causality},
  author  = {Hui, Yongchang and Liu, Chijin and Song, Xiaojun},
  journal = {arXiv preprint arXiv:2509.15798},
  year    = {2025},
  url     = {https://arxiv.org/abs/2509.15798}
}

@software{Roudane2026drgct,
  title   = {drgct: Deep-learning based doubly robust Granger causality testing in Python},
  author  = {Roudane, Merwan},
  version = {1.0.0},
  year    = {2026},
  url     = {https://github.com/merwanroudane/DRGCT}
}
```

`drgct.cite("bibtex")` prints both.

---

**Dr Merwan Roudane** ·
[merwanroudane920@gmail.com](mailto:merwanroudane920@gmail.com) ·
[github.com/merwanroudane](https://github.com/merwanroudane) ·
[PyPI](https://pypi.org/project/drgct/) ·
[Source](https://github.com/merwanroudane/DRGCT)

The method is the intellectual property of its authors, Yongchang Hui, Chijin
Liu and Xiaojun Song. This repository is an independent open-source
implementation and is not affiliated with them.
