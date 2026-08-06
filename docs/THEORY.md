# From the paper to the code

An equation-by-equation map between Hui, Liu and Song (2025),
[arXiv:2509.15798v2](https://arxiv.org/abs/2509.15798), and the `drgct`
source. Read this if you need to verify that the implementation is faithful,
or if you want to extend it.

---

## 0. Notation

| Paper | Code | Meaning |
|---|---|---|
| `{(X_t, Y_t)}_{t=1}^n` | `x`, `y` | The bivariate series. |
| `p`, `q` | `p`, `q` (or `lag` when equal) | Lag orders of the cause and the effect. |
| `X_{t−1} = (X_{t−1},…,X_{t−p})'` | `design.xlag[t]` | `LagDesign.xlag`, shape `(n_eff, p)`. |
| `Y_{t−1} = (Y_{t−1},…,Y_{t−q})'` | `design.ylag[t]` | `LagDesign.ylag`, shape `(n_eff, q)`. |
| `W_{t−1} = (X_{t−1}', Y_{t−1}')'` | `design.w` | shape `(n_eff, p+q)`. |
| `n − q` | `design.n_eff` | Effective sample. |
| `m(Y_{t−1}) = E[Y_t \| Y_{t−1}]` | `mean_fit.fitted` | MLP output. |
| `φ(ν \| Y_{t−1}) = E[e^{iν'X_{t−1}} \| Y_{t−1}]` | `phi_hat` | MDN Monte-Carlo average. |
| `w = (μ, ν)` | `mu`, `nu` | Random directions, shapes `(L, q)` and `(L, p)`. |
| `G, L, M, B` | `G, L, M, B` | Mixture components, directions, pseudo-samples, bootstrap draws. |
| `ξ_t` | `xi` | Bootstrap multipliers. |

The lag bookkeeping lives in `drgct.utils.build_lag_design`. For
`t = q+1,…,n` (1-based), i.e. `t = max(p,q),…,n−1` in 0-based Python, it
stacks `y[t]`, `y[t−1..t−q]` and `x[t−1..t−p]`. `tests/test_utils.py::test_lag_design_alignment`
pins the alignment exactly.

---

## 1. The Markov property, equation (1)

> `P(X_t ≤ x, Y_t ≤ y | F_{t−1}) = P(X_t ≤ x, Y_t ≤ y | W_{t−1})`

This is a *maintained assumption*, not something the code tests: it says the
first `p` lags of `X` and the first `q` lags of `Y` exhaust the relevant
history. In practice it is the reason to choose `p` and `q` generously —
which is precisely what the DRGCT makes affordable.

**Code:** nothing to implement; documented in `drgct.core`'s module docstring
and enforced by the user's choice of `p, q`.

---

## 2. The null hypothesis

### Equation (2) — conditional moment form

> `H0 : E[Y_t − m(Y_{t−1}) | W_{t−1}] = 0` a.s.

### Equation (3) — after Stinchcombe and White (1998)

> `H0 : E[(Y_t − m(Y_{t−1})) e^{i w′W_{t−1}}] = 0` for all `w ∈ W`

`φ(W, w) = exp(i w′W)` is *generically comprehensively revealing* (GCR): if
the conditional moment restriction fails, the unconditional moment is non-zero
for **every** `w` outside a set of measure zero. That is why a finite random
sample of `L` directions retains consistency, and why `W` can be any compact
set with non-empty interior. Indicator-type *comprehensively revealing* (CR)
alternatives such as `1(W_{t−1} ≤ w)` would require `W` to be all of `R^{p+q}`.

**Code:** the choice of GCR family is hard-wired as the complex exponential;
`w_lower`/`w_upper` set the compact box.

### Equation (6) — the doubly robust form

> `H0 : E[(Y_t − m(Y_{t−1})) e^{i μ′Y_{t−1}} (e^{i ν′X_{t−1}} − φ(ν | Y_{t−1}))] = 0`

Proposition 1 of the paper establishes that (3) and (6) define the same null
and the same alternative. Under `H0`, `E[Y_t − m(Y_{t−1}) | W_{t−1}] = 0`, so
conditioning on `W_{t−1}` kills the whole expression; conversely, if (6) holds
for all `(μ, ν)`, the GCR property forces the conditional moment to vanish.

**Why bother.** Write `Δm = m̂ − m` and `Δφ = φ̂ − φ`. In the *naive* process
(5) the leading estimation term is `(n−q)^{−1/2} Σ_t Δm(Y_{t−1}) e^{i w′W_{t−1}}`,
which is `O_p(√n · ‖Δm‖)` and needs `‖Δm‖ = o(n^{−1/2})` — faster than any
deep-network rate in Lemmas 1–2. In the doubly robust process (8) the same
term is multiplied by `(e^{iν′X_{t−1}} − φ̂)`, whose conditional mean given
`Y_{t−1}` is `−Δφ`. The bias therefore scales with `√n · ‖Δm‖ · ‖Δφ‖`, and
Assumption 7 (`‖Δm‖, ‖Δφ‖ = O(n^{−κ₀})` with `κ₀ > 1/4`) makes that `o(1)`.
Two "too slow" estimators multiply into a fast enough product.

**Code:** `drgct/core.py`, the `b` array —

```python
b = np.exp(1j * (xlag_s @ nu.T)) - phi_hat        # doubly robust
# with doubly_robust=False, phi_hat is identically zero -> the naive process (5)
```

---

## 3. Estimation — Steps 1 and 2 of Algorithm 1

### Step 1: the conditional mean

> Reorganise into `{Y_t, Y_{t−1}}_{t=q+1}^n`, train an MLP with `Y_{t−1}` as
> covariates and `Y_t` as response.

**Rates.** Lemma 1 (bounded support, Assumption 2) and Lemma 2 (sub-Gaussian
`Y`, Assumption 3, following Brown 2024) give, for width
`H_n ≍ n^{q/(2(β₀+q))} log²n` and depth `L_n ≍ log n`,

```
E_n (m̂ − m)² ≤ C ( n^{−β₀/(β₀+q)} log⁹n + log n · log log n / n )
```

with high probability, where `β₀` is the Sobolev smoothness of Assumption 4.
Note the exponent: the rate degrades in `q`, which is exactly the curse of
dimensionality — but only in the *first-stage* estimator, whose error is
squared away by double robustness.

**Code:** `drgct/nets.py`

- `MLP(d_in, width, depth, dropout)` — ReLU network `R^q → R`.
- `theory_width(n, q, beta0)` implements `n^{q/(2(β₀+q))} log²n`;
  `theory_depth(n)` implements `round(log n)`; select them with
  `MLPConfig(width="theory", depth="theory")`.
- `paper_width(lag)` implements the `H_n = 5·lag` used in Sections 4–5, with
  `L_n = 1`. This is the default (`width="paper"`).
- `fit_conditional_mean` trains by Adam on the `L2` (or smooth-`L1`) loss and
  returns the **in-sample** fitted values. No sample splitting and no
  cross-fitting, matching Section 1: *"our proposed test does not rely on
  sample splitting or cross-fitting … This offers the advantage of using the
  entire dataset for model training."*

### Step 2: the conditional characteristic function

> Train an MDN with `G` components on `{X_{t−1}, Y_{t−1}}`; draw `M` samples
> `X*_j` from `f̂(· | Y_{t−1})`; set `φ̂(ν | Y_{t−1}) = M⁻¹ Σ_j e^{iν′X*_j}`.

**Rates.** Lemma 3 (following Zhou et al. 2023) bounds the `L²` error of the
MDN density estimate by `C q { G^{−ω₁} + G^{(γ₀+q)/(2γ₀)+4ω₂} n^{−γ₀/(2γ₀+q)} log³(nG) }`,
and Lemma 4 converts it into `max_ℓ ∫ (φ̂(ν_ℓ|y) − φ(ν_ℓ|y))² F(dy) = O_p(n^{−κ})`
for some `κ > 1/2`. The two competing terms in Lemma 3 are why `G` must be
neither too small (approximation bias, first term) nor too large (estimation
variance, second term); `G = 10` is the paper's practical compromise.

**Code:** `drgct/nets.py`

- `MixtureDensityNetwork(d_y, d_x, G, width, depth, min_sigma)` — a shared
  ReLU trunk feeding three heads: `log α_g(y)` via log-softmax, `μ_g(y)`, and
  `σ_g(y)` via softplus plus a floor. The floor is the finite-sample analogue
  of Assumption 6(i)'s `σ_g(y) ≥ C⁻¹G^{−ω₂}`.
- `X_{t−1}` is `p`-dimensional, so the components are diagonal-covariance
  Gaussians, `Π_d N(x_d; μ_{g,d}(y), σ_{g,d}(y)²)`. For `p = 1` this reduces
  exactly to the univariate mixture written in Assumption 6(i).
- Trained by maximum likelihood (`-log_prob().mean()`), as the paper states,
  with gradient-norm clipping at 5.
- `.sample(y, M)` implements Step 2(c) via a categorical draw over components
  followed by a Gaussian draw.

**Step 2(e):** the Monte-Carlo average, vectorised over `t`, `j` and `ℓ`:

```python
proj    = np.einsum("nmp,lp->nml", dens_fit.samples, nu)   # ν_ℓ′X*_j
phi_hat = np.exp(1j * proj).mean(axis=1)                   # (n_eff, L)
```

### Step 2(d): the random directions

> Sample `L` i.i.d. pairs `(μ_ℓ, ν_ℓ)` from a multivariate uniform over a
> compact interval.

```python
mu = rng.uniform(w_lower, w_upper, size=(L, q))
nu = rng.uniform(w_lower, w_upper, size=(L, p))
```

Assumption 7(iv) requires `L` to grow polynomially in `n`. In finite samples
`L` is a genuine tuning parameter: it governs both power and the *simulation*
noise in the p-value. See `drgc_stability` and §5.8 of the guide.

---

## 4. The statistic — Step 3

### Equations (7)–(9)

```
S_n(μ,ν)  = (n−q)^{−1/2} Σ_t (Y_t − m(Y_{t−1}))  e^{iμ′Y_{t−1}} (e^{iν′X_{t−1}} − φ(ν|Y_{t−1}))     (7)
Ŝ_n(μ,ν)  = (n−q)^{−1/2} Σ_t (Y_t − m̂(Y_{t−1})) e^{iμ′Y_{t−1}} (e^{iν′X_{t−1}} − φ̂(ν|Y_{t−1}))     (8)
Ŝ_n(μ_ℓ,ν_ℓ) evaluated at the L random pairs                                                        (9)
```

**Code:**

```python
resid = y_s - mean_fit.fitted                       # Y_t − m̂(Y_{t−1})
a = np.exp(1j * (ylag_s @ mu.T))                    # (n_eff, L)
b = np.exp(1j * (xlag_s @ nu.T)) - phi_hat          # (n_eff, L)
z = resid[:, None] * a * b                          # influence terms z_{t,ℓ}
S_hat = z.sum(axis=0) / np.sqrt(design.n_eff)       # (L,) complex
```

### Equation (10) — the Kolmogorov–Smirnov functional

```
KS_n = max_{ℓ≤L} max( |Re Ŝ_n(μ_ℓ,ν_ℓ)| , |Im Ŝ_n(μ_ℓ,ν_ℓ)| )
```

```python
ks_stat = float(np.max(np.maximum(np.abs(S_hat.real), np.abs(S_hat.imag))))
```

`tests/test_core.py::test_statistic_matches_its_definition` recomputes both
from `result.influence` and asserts equality to `1e-10`.

---

## 5. Asymptotics

| Result | Statement | What it buys you |
|---|---|---|
| **Proposition 2** | `sup_{(μ,ν)} \|S_n − Ŝ_n\| = o_p(1)` under `H0` | Plugging in the two network estimators is asymptotically free — *because* of the doubly robust construction. |
| **Theorem 1** | `Ŝ_n ⇝ S_∞`, a zero-mean Gaussian process | The statistic has a limit. |
| **Corollary 1** | `KS_n ⇝ sup_{(μ,ν)} max(\|S_R\|, \|S_I\|)` | The limit of the statistic itself. Data-dependent covariance kernel ⇒ bootstrap needed. |
| **Theorem 2** | Under `H1`, `n^{−1/2} Ŝ_n →_p E[(Y_t − m) e^{iw′W}] ≠ 0` | `KS_n → ∞`, so power → 1: the test is consistent. |
| **Theorem 3 / Corollary 2** | Under `H_{1n} : E[Y_t − m \| W_{t−1}] = n^{−1/2}Δ(W_{t−1})`, `Ŝ_n ⇝ S_∞ + Q` | Non-trivial power against `n^{−1/2}` local alternatives — the parametric rate, far faster than the `n^{−1/2}h^{−d/4}` typical of smoothing tests. |
| **Theorem 4** | `Ŝ*_n →_p* S_∞` under `H0`, `H1` and `H_{1n}` | The multiplier bootstrap is valid everywhere, so critical values are right and power is preserved. |

The paper also notes explicitly that `Ŝ⁰_n` — the naive process (5) — **does
not** converge to a Gaussian process under `H0`. That is the theoretical
statement behind the empirical size failure this package reproduces with
`doubly_robust=False`.

---

## 6. The multiplier bootstrap — Step 4

### Equation (12)

```
Ŝ*_n(μ,ν) = (n−q)^{−1/2} Σ_t ξ_t (Y_t − m̂(Y_{t−1})) e^{iμ′Y_{t−1}} (e^{iν′X_{t−1}} − φ̂(ν|Y_{t−1}))
```

with `{ξ_t}` i.i.d., mean 0, variance 1, **bounded support**, independent of
the data. Assumption 1 makes `{Y_t − E(Y_t|W_{t−1})}` a martingale difference
sequence, which is what lets the multipliers reproduce the covariance kernel.

**Code:** because the summand is exactly `result.influence`, the entire
bootstrap is one matrix product:

```python
xi         = draw_multipliers(rng, (B, n_eff), multiplier)   # (B, n_eff)
S_boot     = (xi @ z) / np.sqrt(n_eff)                       # (B, L) complex
boot_stats = np.max(np.maximum(np.abs(S_boot.real), np.abs(S_boot.imag)), axis=1)
pvalue     = float(np.mean(boot_stats >= ks_stat))
```

Nothing is re-estimated. This is the computational point of Section 3.4:
*"this procedure allows the critical value to be determined using the already
estimated quantities, thereby avoiding redundant calculations."* With
`n = 750`, `L = 20`, `B = 999`, the bootstrap costs about 15 ms against
roughly 10 s for the two networks.

**Choice of `ξ`.** `"rademacher"` (±1 with equal probability) is the default:
mean 0, variance 1, support `{−1, 1}`. `"mammen"` is the skewed two-point
distribution with the same first two moments and bounded support.
`"normal"` is offered for experimentation but violates the bounded-support
condition of Theorem 4.

### Step 5

> Reject `H0` if `p*_n < α`, where `p*_n = B⁻¹ Σ_b 1{KS*_{n,b} ≥ KS_n}`.

---

## 7. Assumptions and what they mean for practice

| Assumption | Statement | Practical reading |
|---|---|---|
| **1** | Stationarity; `β`-mixing with `β(t) ≤ C₁ exp(−C₂t)`; `{Y_t − E(Y_t\|W_{t−1})}` is an m.d.s. | **The binding one.** Difference or transform to stationarity, and verify with `check_stationarity`. Exponential mixing rules out long memory. |
| **2** | Compact support of `X`, `Y`; `‖m‖_∞` and `‖f‖_∞` bounded | Used by Lemma 1. Financial returns are not compactly supported — which is exactly why Lemma 2 exists. |
| **3** | Density of `Y_t` is sub-Gaussian | The unbounded-data route (Brown 2024). Fat-tailed returns strictly speaking violate this too; the test's empirical size holds up well in practice, but it is worth a robustness check with winsorised data. |
| **4** | `m(·)` lies in a Sobolev ball of smoothness `β₀` | Sets the MLP rate. `MLPConfig(beta0=...)` feeds `width="theory"`. |
| **5** | `f_{X\|Y}` is well approximated by a `G`-component Gaussian mixture; components bounded; `α_g, μ_g, σ_g` Sobolev-smooth of order `γ₀` | Justifies the MDN. Discrete or bounded `X` (e.g. a 0/1 policy indicator) sits badly here. |
| **6** | The MDN function class and its parameter count | `min_sigma` implements the `σ_g ≥ C⁻¹G^{−ω₂}` part. |
| **7** | `‖m̂ − m‖ = O(n^{−κ₀})` and `‖f̂ − f‖ = O(n^{−κ₀})` with `κ₀ > 1/4`; `M = κ₁n^{κ₂}` with `κ₂ ≥ 1/2`; `L` polynomial in `n` | **The double-robustness budget.** `κ₀ > 1/4` for each, so the product beats `n^{−1/2}`. Note 7(iii) formally wants `M` to grow with `n`; the paper fixes `M = 20` and reports insensitivity. Raise `M` if `p` is large. |
| **8** | Bounded densities for `X_t`, `Y_t`; sub-Gaussian `ε_t = Y_t − E(Y_t\|W_{t−1})` | Standard regularity for the weak-convergence argument. |
| **9** | `Δ(w)` bounded; a uniform LLN for the local drift | Only needed for the local-power result. |

---

## 8. Deliberate implementation choices

Places where the paper leaves a choice open and this package had to make one.
All are exposed as arguments.

| Choice | What we do | Why | Override |
|---|---|---|---|
| Support of `(μ, ν)` | `U[−1, 1]^{p+q}` on standardised inputs | The paper says "a specified compact interval" without fixing it. With z-scored inputs, `[−1,1]` puts `w′W` on a scale where `e^{iw′W}` oscillates informatively without aliasing. | `w_lower`, `w_upper` |
| Input scaling | z-score `X`, `Y` and the lag blocks | Leaves the p-value invariant (both `KS_n` and every `KS*_n` scale identically) but makes network training far more reliable. | `standardize=False` |
| Multiplier distribution | Rademacher | Bounded support, as Theorem 4 requires; the canonical two-point choice. | `multiplier="mammen"` / `"normal"` |
| MDN for `p > 1` | Diagonal-covariance Gaussian mixture in `R^p` | Assumption 6(i) is written for scalar `x`; the diagonal mixture is the standard Bishop (1994) multivariate MDN and reduces to it at `p = 1`. | subclass `MixtureDensityNetwork` |
| Optimiser | Adam, early stopping on the running training loss | Not specified in the paper. | `MLPConfig`, `MDNConfig` |
| `N(0, 0.5)` in the DGPs | read as a **variance** | The standard convention; the paper does not disambiguate. | `simulate_dgp(..., innovation_scale="sd")` |
| NHKJ benchmark | Zheng (1996) / Fan–Li (1996) degenerate U-statistic with a fourth-order Gaussian kernel and `h = c·n^{−0.15}` | The paper states the kernel order and bandwidth schedule but not the estimator's algebra. Same family, but **different finite-sample behaviour** from the paper's NHKJ — see `results/README.md`. Not a transcription. | `nhkj_test(kernel_order=..., bandwidth=...)` |
| `p > q` | allowed, effective sample starts at `max(p,q)+1` | The paper maintains `p ≤ q`; relaxing it is harmless and sometimes useful. | `allow_p_gt_q=False` in `build_lag_design` |

---

## 9. Source map

| Paper element | File | Symbol |
|---|---|---|
| Lag bookkeeping, Steps 1(a), 2(a) | `utils.py` | `build_lag_design`, `LagDesign` |
| Step 1(b), Lemmas 1–2 | `nets.py` | `MLP`, `fit_conditional_mean`, `theory_width`, `theory_depth` |
| Steps 2(b)–2(c), Lemma 3 | `nets.py` | `MixtureDensityNetwork`, `fit_conditional_density` |
| Step 2(d) | `core.py` | `mu = rng.uniform(...)`, `nu = rng.uniform(...)` |
| Step 2(e), Lemma 4 | `core.py` | `phi_hat = np.exp(1j * proj).mean(axis=1)` |
| Equations (8)–(9), Step 3(a) | `core.py` | `z`, `S_hat` |
| Equation (10), Step 3(b) | `core.py` | `ks_stat` |
| Equation (12), Step 4, Theorem 4 | `core.py`, `utils.py` | `S_boot`, `draw_multipliers` |
| Step 5 | `core.py` | `reject = pvalue < alpha` |
| Naive process (5) | `core.py` | `doubly_robust=False` |
| Table 1, Table 2 | `dgp.py` | `simulate_dgp`, `PARAMETERS`, `dgp_table`, `parameter_table` |
| Table 3, Table 4 | `simulate.py`, `tables.py` | `monte_carlo`, `table_size`, `table_power` |
| Table 5, Table 6 | `applications.py`, `tables.py` | `price_volume_study`, `table_detection`, `table_lag_orders` |
| NHKJ benchmark | `nhkj.py` | `nhkj_test` |

---

## 10. References

- Bishop, C. M. (1994). *Mixture density networks.* Aston University.
- Brown, C. (2024). *Statistical properties of deep neural networks with dependent data.* arXiv:2410.11113.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W. and Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal* 21, C1–C68.
- Delgado, M. A. and González Manteiga, W. (2001). Significance testing in nonparametric regression based on the bootstrap. *The Annals of Statistics* 29, 1469–1507.
- Fan, Y. and Li, Q. (1996). Consistent model specification tests: omitted variables and semiparametric functional forms. *Econometrica* 64, 865–890.
- Farrell, M. H., Liang, T. and Misra, S. (2021). Deep neural networks for estimation and inference. *Econometrica* 89, 181–213.
- Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica* 37, 424–438.
- Hui, Y., Liu, C. and Song, X. (2025). *Deep learning based doubly robust test for Granger causality.* arXiv:2509.15798.
- Mammen, E. (1993). Bootstrap and wild bootstrap for high-dimensional linear models. *The Annals of Statistics* 21, 255–285.
- Nishiyama, Y., Hitomi, K., Kawasaki, Y. and Jeong, K. (2011). A consistent nonparametric test for nonlinear causality — specification in time series regression. *Journal of Econometrics* 165, 112–127.
- Robins, J. M., Rotnitzky, A. and Zhao, L. P. (1994). Estimation of regression coefficients when some regressors are not always observed. *JASA* 89, 846–866.
- Stinchcombe, M. B. and White, H. (1998). Consistent specification testing with nuisance parameters present only under the alternative. *Econometric Theory* 14, 295–325.
- Vovk, V. and Wang, R. (2020). Combining p-values via averaging. *Biometrika* 107, 791–808.
- Zheng, J. X. (1996). A consistent test of functional form via nonparametric estimation techniques. *Journal of Econometrics* 75, 263–289.
- Zhou, Y., Shi, C., Li, L. and Yao, Q. (2023). Testing for the Markov property in time series via deep conditional generative learning. *JRSS-B* 85, 1204–1222.
