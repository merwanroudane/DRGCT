"""Deep-learning components of the DRGCT.

Two networks are needed by Algorithm 1 of Hui, Liu and Song (2025):

``MLP``  (Step 1)
    A multilayer perceptron with ReLU activations estimating the conditional
    mean ``m(Y_{t-1}) = E[Y_t | Y_{t-1}]``.  Lemma 1 (bounded ``Y``) and
    Lemma 2 (sub-Gaussian ``Y``) give its convergence rate; the architecture
    that attains those rates is width ``H_n ~ n^{q/(2(beta0+q))} log^2(n)``
    and depth ``L_n ~ log(n)``.  In the paper's numerical work the far
    simpler choice ``L_n = 1``, ``H_n = 5 * lag`` is used, and that is our
    default (``width="paper"``).

``MixtureDensityNetwork``  (Step 2)
    A Bishop (1994) mixture density network estimating the conditional
    density ``f_{X_{t-1}|Y_{t-1}}(x|y)`` with ``G`` Gaussian components,
    trained by maximum likelihood.  Draws from the fitted density supply the
    Monte-Carlo estimate of the conditional characteristic function

        phihat(nu | Y_{t-1}) = M^{-1} sum_{j=1}^{M} exp(i nu' X*_j).

    ``X_{t-1}`` is ``p``-dimensional, so the mixture uses diagonal-covariance
    Gaussian components; for ``p = 1`` this is exactly the univariate mixture
    of Assumption 6(i).

Both trainers use the *full* sample -- the test deliberately avoids sample
splitting and cross-fitting (Section 1 of the paper), which is what makes the
doubly robust construction worth having.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "MLPConfig",
    "ConvergenceWarning",
    "convergence_check",
    "MDNConfig",
    "paper_width",
    "theory_width",
    "theory_depth",
    "MLP",
    "MixtureDensityNetwork",
    "fit_conditional_mean",
    "fit_conditional_density",
]


# --------------------------------------------------------------------------- #
# Architecture rules of thumb
# --------------------------------------------------------------------------- #
def paper_width(lag: int, multiplier: int = 5) -> int:
    """``H_n = 5 * lag`` -- the width used in Sections 4 and 5 of the paper."""
    return max(4, int(multiplier) * int(lag))


def theory_width(n: int, q: int, beta0: int = 2) -> int:
    """``H_n ~ n^{q / (2 (beta0 + q))} log^2(n)`` -- the rate of Lemma 1.

    ``beta0`` is the Sobolev smoothness of ``m(.)`` in Assumption 4.  Larger
    ``beta0`` (smoother conditional mean) gives a narrower network.
    """
    n, q, beta0 = int(n), int(q), int(beta0)
    return max(4, int(round(n ** (q / (2.0 * (beta0 + q))) * math.log(n) ** 2)))


def theory_depth(n: int) -> int:
    """``L_n ~ log(n)`` -- the depth of Lemmas 1 and 2."""
    return max(1, int(round(math.log(max(int(n), 3)))))


def _resolve_width(width, *, lag: int, n: int, beta0: int) -> int:
    if isinstance(width, str):
        if width == "paper":
            return paper_width(lag)
        if width == "theory":
            return theory_width(n, lag, beta0)
        raise ValueError("width must be an int, 'paper' or 'theory'.")
    return max(1, int(width))


def _resolve_depth(depth, *, n: int) -> int:
    if isinstance(depth, str):
        if depth == "paper":
            return 1
        if depth == "theory":
            return theory_depth(n)
        raise ValueError("depth must be an int, 'paper' or 'theory'.")
    return max(1, int(depth))


# --------------------------------------------------------------------------- #
# Configuration objects
# --------------------------------------------------------------------------- #
@dataclass
class MLPConfig:
    """Hyper-parameters of the conditional-mean network (Step 1).

    Parameters
    ----------
    width, depth : int or {'paper', 'theory'}
        Hidden width ``H_n`` and number of hidden layers ``L_n``.
        ``'paper'`` reproduces Section 4 (``H_n = 5*lag``, ``L_n = 1``);
        ``'theory'`` uses the rates of Lemma 1.
    beta0 : int
        Sobolev smoothness used only when ``width='theory'``.
    loss : {'l2', 'smooth_l1'}
        The paper allows either; ``'l2'`` is the default.
    epochs, lr, batch_size, weight_decay : training controls.
    patience, min_delta : early stopping on the running training loss.
    dropout : float
        Optional dropout applied after each hidden layer (0 disables).
    device : str or None
        ``'cpu'``, ``'cuda'``, or ``None`` for automatic selection.
    warn_convergence : bool
        Emit a :class:`ConvergenceWarning` when the epoch cap is reached while
        the loss is still falling appreciably.  Set ``False`` inside long
        Monte-Carlo loops, where the warning would repeat thousands of times.
    """

    width: int | Literal["paper", "theory"] = "paper"
    depth: int | Literal["paper", "theory"] = "paper"
    beta0: int = 2
    loss: Literal["l2", "smooth_l1"] = "l2"
    epochs: int = 400
    lr: float = 5e-3
    batch_size: int = 512
    weight_decay: float = 0.0
    patience: int = 60
    min_delta: float = 1e-6
    dropout: float = 0.0
    device: str | None = None
    verbose: bool = False
    warn_convergence: bool = True


@dataclass
class MDNConfig:
    """Hyper-parameters of the mixture density network (Step 2).

    Parameters
    ----------
    n_components : int
        ``G`` in the paper.  ``G = 10`` "is suitable for most scenarios"
        (Section 4).  Too small inflates the type I error through MDN bias;
        too large inflates the variance of ``KS_n``.
    width, depth, beta0 : as in :class:`MLPConfig` (``'paper'`` -> ``5*lag``, 1).
    min_sigma : float
        Lower bound on the component standard deviations, the finite-sample
        counterpart of the constraint ``sigma_g(y) >= C^{-1} G^{-omega2}`` in
        Assumption 6(i).  Prevents mixture components from collapsing.
    epochs, lr, batch_size, weight_decay, patience, min_delta, device, verbose
        Training controls, as in :class:`MLPConfig`.
    """

    n_components: int = 10
    width: int | Literal["paper", "theory"] = "paper"
    depth: int | Literal["paper", "theory"] = "paper"
    beta0: int = 2
    min_sigma: float = 1e-2
    epochs: int = 500
    lr: float = 5e-3
    batch_size: int = 512
    weight_decay: float = 0.0
    patience: int = 70
    min_delta: float = 1e-6
    device: str | None = None
    verbose: bool = False
    warn_convergence: bool = True


def _pick_device(name: str | None) -> torch.device:
    if name is not None:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Networks
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """Fully connected ReLU network ``R^{d_in} -> R``."""

    def __init__(self, d_in: int, width: int, depth: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        prev = d_in
        for _ in range(depth):
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(x).squeeze(-1)


class MixtureDensityNetwork(nn.Module):
    """Bishop (1994) MDN with ``G`` diagonal-covariance Gaussian components.

    The network maps ``y in R^{d_y}`` to mixture weights ``alpha_g(y)``,
    means ``mu_g(y) in R^{d_x}`` and standard deviations
    ``sigma_g(y) in R_+^{d_x}``, so that

        fhat(x | y) = sum_g alpha_g(y) prod_{d} N(x_d ; mu_{g,d}(y), sigma_{g,d}(y)^2).
    """

    def __init__(
        self,
        d_y: int,
        d_x: int,
        n_components: int,
        width: int,
        depth: int,
        min_sigma: float = 1e-2,
    ):
        super().__init__()
        self.d_x = int(d_x)
        self.G = int(n_components)
        self.min_sigma = float(min_sigma)

        trunk: list[nn.Module] = []
        prev = d_y
        for _ in range(depth):
            trunk.append(nn.Linear(prev, width))
            trunk.append(nn.ReLU())
            prev = width
        self.trunk = nn.Sequential(*trunk)
        self.head_logits = nn.Linear(prev, self.G)
        self.head_mu = nn.Linear(prev, self.G * self.d_x)
        self.head_sigma = nn.Linear(prev, self.G * self.d_x)

    def forward(self, y: torch.Tensor):
        """Return ``(log_alpha, mu, sigma)`` with shapes ``(N,G)``, ``(N,G,dx)``, ``(N,G,dx)``."""
        h = self.trunk(y)
        log_alpha = F.log_softmax(self.head_logits(h), dim=-1)
        mu = self.head_mu(h).view(-1, self.G, self.d_x)
        sigma = F.softplus(self.head_sigma(h)).view(-1, self.G, self.d_x) + self.min_sigma
        return log_alpha, mu, sigma

    def log_prob(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Log conditional density ``log fhat(x | y)`` for each row."""
        log_alpha, mu, sigma = self(y)
        x = x.unsqueeze(1)  # (N, 1, dx)
        comp = -0.5 * ((x - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
        comp = comp.sum(dim=-1)  # (N, G)
        return torch.logsumexp(log_alpha + comp, dim=-1)

    @torch.no_grad()
    def sample(self, y: torch.Tensor, n_samples: int, generator=None) -> torch.Tensor:
        """Draw ``M = n_samples`` pseudo-observations of ``X`` per row of ``y``.

        Returns
        -------
        torch.Tensor, shape ``(N, M, d_x)``
            Step 2(c) of Algorithm 1.
        """
        log_alpha, mu, sigma = self(y)
        n, g, dx = mu.shape
        probs = log_alpha.exp()
        idx = torch.multinomial(
            probs, num_samples=n_samples, replacement=True, generator=generator
        )  # (N, M)
        idx_exp = idx.unsqueeze(-1).expand(n, n_samples, dx)
        mu_sel = torch.gather(mu, 1, idx_exp)
        sigma_sel = torch.gather(sigma, 1, idx_exp)
        eps = torch.randn(mu_sel.shape, generator=generator, device=mu_sel.device)
        return mu_sel + sigma_sel * eps


# --------------------------------------------------------------------------- #
# Training loops
# --------------------------------------------------------------------------- #
class ConvergenceWarning(UserWarning):
    """Raised when a network hits its epoch cap while the loss is still falling."""


def convergence_check(history, epochs: int, tag: str, tol: float = 0.05) -> str | None:
    """Return a warning message when training stopped while still improving.

    If the loop ran to ``epochs`` without early stopping *and* the loss over
    the final tenth of the run still fell by more than ``tol`` (relative), the
    network is under-trained: the conditional-mean or conditional-density
    estimate has not settled, and the resulting test loses power (and, for the
    MDN, size control).  Raise ``epochs``.
    """
    if len(history) < max(20, int(0.2 * epochs)) or len(history) < epochs:
        return None  # early stopping fired: the loss had already plateaued
    tail = max(5, len(history) // 10)
    first, last = float(history[-2 * tail]), float(history[-1])
    if not np.isfinite(first) or abs(first) < 1e-12:
        return None
    improvement = (first - last) / abs(first)
    if improvement > tol:
        return (
            f"[drgct] {tag} hit the {epochs}-epoch cap while the loss was still "
            f"falling ({100 * improvement:.1f}% over the last {2 * tail} epochs). "
            f"The estimator is under-trained; raise epochs "
            f"({'MLPConfig' if tag == 'MLP' else 'MDNConfig'}(epochs=...)) "
            "and inspect drgct.plots.plot_training_curves."
        )
    return None


def _train(model, closure, *, epochs, lr, weight_decay, patience, min_delta, verbose, tag):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, wait, history = math.inf, 0, []
    for epoch in range(int(epochs)):
        loss_value = closure(opt)
        history.append(loss_value)
        if loss_value < best - min_delta:
            best, wait = loss_value, 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"[{tag}] early stop at epoch {epoch + 1}, loss={loss_value:.6f}")
                break
        if verbose and (epoch + 1) % 50 == 0:
            print(f"[{tag}] epoch {epoch + 1:4d}  loss={loss_value:.6f}")
    return history


@dataclass
class MeanFit:
    """Output of :func:`fit_conditional_mean`."""

    fitted: np.ndarray  # mhat(Y_{t-1}) on the *standardised* Y scale
    model: MLP
    history: list = field(default_factory=list)
    width: int = 0
    depth: int = 0
    warning: str | None = None


def fit_conditional_mean(
    ylag: np.ndarray,
    y: np.ndarray,
    config: MLPConfig | None = None,
    *,
    lag: int | None = None,
) -> MeanFit:
    """Step 1 of Algorithm 1: train the MLP and return ``{mhat(Y_{t-1})}``.

    Parameters
    ----------
    ylag : ndarray, shape (N, q)
        Input covariates ``Y_{t-1}`` (already standardised by the caller).
    y : ndarray, shape (N,)
        Response ``Y_t`` (already standardised by the caller).
    config : MLPConfig, optional
    lag : int, optional
        Value used to resolve ``width='paper'``; defaults to ``ylag.shape[1]``.

    Returns
    -------
    MeanFit
        ``.fitted`` holds the in-sample conditional-mean estimates.  No sample
        splitting is used, in line with Section 1 of the paper.
    """
    cfg = config or MLPConfig()
    device = _pick_device(cfg.device)
    n, d_in = ylag.shape
    lag = int(lag if lag is not None else d_in)
    width = _resolve_width(cfg.width, lag=lag, n=n, beta0=cfg.beta0)
    depth = _resolve_depth(cfg.depth, n=n)

    xb = torch.as_tensor(ylag, dtype=torch.float32, device=device)
    yb = torch.as_tensor(y, dtype=torch.float32, device=device)

    model = MLP(d_in, width, depth, dropout=cfg.dropout).to(device)
    loss_fn = F.mse_loss if cfg.loss == "l2" else F.smooth_l1_loss
    batch = min(int(cfg.batch_size), n)

    def closure(opt):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, batch):
            sel = perm[start : start + batch]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb[sel]), yb[sel])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * sel.numel()
        return total / n

    history = _train(
        model,
        closure,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
        min_delta=cfg.min_delta,
        verbose=cfg.verbose,
        tag="MLP",
    )

    warning = convergence_check(history, cfg.epochs, "MLP")
    if warning and cfg.warn_convergence:
        warnings.warn(warning, ConvergenceWarning, stacklevel=2)

    model.eval()
    with torch.no_grad():
        fitted = model(xb).detach().cpu().numpy().astype(float)
    return MeanFit(fitted=fitted, model=model, history=history, width=width, depth=depth,
                   warning=warning)


@dataclass
class DensityFit:
    """Output of :func:`fit_conditional_density`."""

    samples: np.ndarray  # (N, M, p) draws X*_j from fhat(. | Y_{t-1})
    model: MixtureDensityNetwork
    history: list = field(default_factory=list)
    width: int = 0
    depth: int = 0
    n_components: int = 0
    warning: str | None = None


def fit_conditional_density(
    ylag: np.ndarray,
    xlag: np.ndarray,
    config: MDNConfig | None = None,
    *,
    n_samples: int = 20,
    lag: int | None = None,
    seed: int | None = None,
) -> DensityFit:
    """Steps 2(b)-2(c) of Algorithm 1: train the MDN and draw pseudo-samples.

    Parameters
    ----------
    ylag : ndarray, shape (N, q)
        Input covariates ``Y_{t-1}`` (standardised by the caller).
    xlag : ndarray, shape (N, p)
        Response block ``X_{t-1}`` (standardised by the caller).
    config : MDNConfig, optional
    n_samples : int
        ``M``, the number of draws per observation.  The paper fixes ``M = 20``
        and reports that results are insensitive to it.
    lag : int, optional
        Used to resolve ``width='paper'``.
    seed : int, optional
        Seed of the torch generator used for the pseudo-samples only.

    Returns
    -------
    DensityFit
    """
    cfg = config or MDNConfig()
    device = _pick_device(cfg.device)
    n, d_y = ylag.shape
    d_x = xlag.shape[1]
    lag = int(lag if lag is not None else max(d_y, d_x))
    width = _resolve_width(cfg.width, lag=lag, n=n, beta0=cfg.beta0)
    depth = _resolve_depth(cfg.depth, n=n)

    yb = torch.as_tensor(ylag, dtype=torch.float32, device=device)
    xb = torch.as_tensor(xlag, dtype=torch.float32, device=device)

    model = MixtureDensityNetwork(
        d_y=d_y,
        d_x=d_x,
        n_components=cfg.n_components,
        width=width,
        depth=depth,
        min_sigma=cfg.min_sigma,
    ).to(device)
    batch = min(int(cfg.batch_size), n)

    def closure(opt):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, batch):
            sel = perm[start : start + batch]
            opt.zero_grad(set_to_none=True)
            nll = -model.log_prob(yb[sel], xb[sel]).mean()
            nll.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(nll.detach()) * sel.numel()
        return total / n

    history = _train(
        model,
        closure,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
        min_delta=cfg.min_delta,
        verbose=cfg.verbose,
        tag="MDN",
    )

    warning = convergence_check(history, cfg.epochs, "MDN")
    if warning and cfg.warn_convergence:
        warnings.warn(warning, ConvergenceWarning, stacklevel=2)

    model.eval()
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
    samples = model.sample(yb, int(n_samples), generator=gen).cpu().numpy().astype(float)
    return DensityFit(
        samples=samples,
        model=model,
        history=history,
        width=width,
        depth=depth,
        n_components=cfg.n_components,
        warning=warning,
    )
