r"""Publication-quality figures for the DRGCT.

The visual system
-----------------
One serif typeface, one restrained palette, no chartjunk, everything legible
in greyscale and at single-column width -- the house style of *Econometrica*,
the *Journal of Econometrics* and *JBES*.  Call :func:`use_journal_style` once
(every plotting function does it for you) and :func:`save_figure` to write a
vector PDF plus a 400-dpi PNG in one go.

Available figures
-----------------
==================================  ==============================================
Function                            What it shows
==================================  ==============================================
:func:`plot_size`                   Empirical size vs lag, with the nominal level
                                    and a Monte-Carlo confidence band
:func:`plot_power`                  Empirical power vs lag, by sample size
:func:`plot_size_power_grid`        Both, as one multi-panel figure
:func:`plot_pvalue_ecdf`            Davidson-MacKinnon p-value plot (uniformity
                                    of the bootstrap p-value under H0)
:func:`plot_bootstrap_distribution` Bootstrap null distribution of KS*_n with the
                                    observed statistic and critical values
:func:`plot_empirical_process`      |Re| and |Im| of Shat_n over the L directions,
                                    against the bootstrap envelope
:func:`plot_lag_profile`            p-value as a function of the lag order
:func:`plot_pvalue_heatmap`         Lag x period p-value map for the application
:func:`plot_series_overview`        Prices, volumes and their transformed series
:func:`plot_mdn_fit`                MDN conditional-density diagnostic
:func:`plot_training_curves`        MLP / MDN training loss
:func:`plot_rolling_pvalue`         Rolling-window causality over calendar time
==================================  ==============================================
"""

from __future__ import annotations

import pathlib
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "PALETTE",
    "METHOD_STYLE",
    "use_journal_style",
    "save_figure",
    "plot_size",
    "plot_power",
    "plot_size_power_grid",
    "plot_pvalue_ecdf",
    "plot_bootstrap_distribution",
    "plot_empirical_process",
    "plot_lag_profile",
    "plot_pvalue_heatmap",
    "plot_series_overview",
    "plot_mdn_fit",
    "plot_training_curves",
    "plot_rolling_pvalue",
    "plot_stability",
]

#: Colour-blind-safe, greyscale-separable palette.
PALETTE = {
    "ink": "#1A1A1A",
    "muted": "#6B7280",
    "rule": "#B9BEC7",
    "blue": "#1F4E79",
    "terracotta": "#B4462F",
    "green": "#3E7B5E",
    "gold": "#B08300",
    "purple": "#6C4A96",
    "sky": "#7FA8C9",
    "sand": "#E4D7BE",
}

#: Consistent look for the three estimators across every figure.
METHOD_STYLE = {
    "drgc": dict(color=PALETTE["blue"], marker="o", ls="-", label="DRGC"),
    "DRGC": dict(color=PALETTE["blue"], marker="o", ls="-", label="DRGC"),
    "nhkj": dict(color=PALETTE["terracotta"], marker="s", ls="--", label="NHKJ"),
    "NHKJ": dict(color=PALETTE["terracotta"], marker="s", ls="--", label="NHKJ"),
    "drgc_naive": dict(color=PALETTE["gold"], marker="^", ls=":", label="DRGC (naive)"),
    "DRGC-naive": dict(color=PALETTE["gold"], marker="^", ls=":", label="DRGC (naive)"),
}

_N_STYLE = ["-", "--", ":", "-."]


# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
def use_journal_style(*, base_size: float = 9.5, serif: bool = True) -> None:
    """Install the house Matplotlib style (idempotent)."""
    import matplotlib as mpl

    family = (
        ["DejaVu Serif", "Times New Roman", "Nimbus Roman", "serif"]
        if serif
        else ["DejaVu Sans", "Arial", "sans-serif"]
    )
    mpl.rcParams.update(
        {
            "font.family": "serif" if serif else "sans-serif",
            "font.serif" if serif else "font.sans-serif": family,
            "font.size": base_size,
            "axes.titlesize": base_size + 0.5,
            "axes.titleweight": "normal",
            "axes.labelsize": base_size,
            "axes.edgecolor": PALETTE["ink"],
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": PALETTE["rule"],
            "grid.linewidth": 0.4,
            "grid.alpha": 0.55,
            "xtick.labelsize": base_size - 1,
            "ytick.labelsize": base_size - 1,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.frameon": False,
            "legend.fontsize": base_size - 1,
            "legend.handlelength": 2.2,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.0,
            "figure.dpi": 120,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavuserif" if serif else "dejavusans",
        }
    )


def save_figure(
    fig,
    stem: str,
    outdir: str | pathlib.Path = "results/figures",
    *,
    formats: Sequence[str] = ("pdf", "png"),
    close: bool = True,
    quiet: bool = False,
) -> dict:
    """Save ``fig`` as ``<outdir>/<stem>.<fmt>`` for each requested format."""
    import matplotlib.pyplot as plt

    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for fmt in formats:
        p = outdir / f"{stem}.{fmt}"
        fig.savefig(p, format=fmt)
        paths[fmt] = str(p)
    if not quiet:
        print(f"  [figure] {stem}: " + ", ".join(pathlib.Path(p).name for p in paths.values()))
    if close:
        plt.close(fig)
    return paths


def _grid_off_x(ax):
    ax.grid(axis="x", visible=False)


def _panel_tag(ax, text):
    ax.set_title(text, loc="left", pad=6, fontsize=ax.title.get_fontsize())


# --------------------------------------------------------------------------- #
# Simulation figures
# --------------------------------------------------------------------------- #
def plot_size(
    summary_df,
    *,
    nominal: float = 0.05,
    dgps: Sequence[str] = ("S1", "S2"),
    methods: Sequence[str] = ("drgc", "nhkj"),
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    band: bool = True,
):
    """Empirical size against lag order, one panel per (DGP, sample size).

    Parameters
    ----------
    summary_df : DataFrame
        Output of :func:`drgct.simulate.summarize`.
    nominal : float, default 0.05
        Nominal level; drawn as a horizontal reference.
    dgps, methods : sequences of str
    ylim : tuple, optional
    figsize : tuple, optional
    band : bool
        Shade the +/-1.96 Monte-Carlo standard-error band around ``nominal``,
        i.e. the region in which an exactly-sized test should land.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    use_journal_style()
    d = summary_df[summary_df["dgp"].isin(dgps)]
    ns = sorted(d["n"].unique())
    dgps = [g for g in dgps if g in set(d["dgp"])]
    nrow, ncol = len(dgps), len(ns)
    figsize = figsize or (2.5 * ncol + 1.0, 2.35 * nrow + 0.6)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, sharey=True, sharex=True, squeeze=False)

    for i, g in enumerate(dgps):
        for j, n in enumerate(ns):
            ax = axes[i][j]
            sub = d[(d["dgp"] == g) & (d["n"] == n)]
            reps = int(sub["reps"].max()) if len(sub) else 0
            if band and reps:
                se = np.sqrt(nominal * (1 - nominal) / reps)
                ax.axhspan(nominal - 1.96 * se, nominal + 1.96 * se,
                           color=PALETTE["sand"], alpha=0.55, lw=0, zorder=0)
            ax.axhline(nominal, color=PALETTE["muted"], lw=0.8, ls=(0, (4, 3)), zorder=1)
            for m in methods:
                s = sub[sub["method"] == m].sort_values("lag")
                if s.empty:
                    continue
                st = dict(METHOD_STYLE.get(m, {}))
                st.setdefault("label", m)
                ax.plot(s["lag"], s["rejection_rate"], zorder=3, **st)
            ax.set_xticks(sorted(d["lag"].unique()))
            _grid_off_x(ax)
            if i == 0:
                _panel_tag(ax, f"$n = {int(n)}$")
            if j == 0:
                ax.set_ylabel(f"DGP {g}\nempirical size")
            if i == nrow - 1:
                ax.set_xlabel("lag order")
            if ylim:
                ax.set_ylim(*ylim)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(labels), loc="lower center",
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Empirical size at the 5% nominal level", y=1.0, fontsize=11)
    fig.tight_layout()
    return fig


def plot_power(
    summary_df,
    *,
    dgps: Sequence[str] = ("P1", "P2", "P3", "P4"),
    methods: Sequence[str] = ("drgc", "nhkj"),
    figsize: tuple[float, float] | None = None,
    ncol: int = 2,
):
    """Empirical power against lag order, one panel per DGP, one line per ``n``.

    Method is encoded by colour and marker, sample size by line style and
    opacity, so the figure stays readable in black and white.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    use_journal_style()
    d = summary_df[summary_df["dgp"].isin(dgps)]
    dgps = [g for g in dgps if g in set(d["dgp"])]
    ns = sorted(d["n"].unique())
    nrow = int(np.ceil(len(dgps) / ncol))
    figsize = figsize or (3.1 * ncol + 0.6, 2.6 * nrow + 0.8)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, sharey=True, sharex=True, squeeze=False)
    flat = [a for row in axes for a in row]

    for ax, g in zip(flat, dgps):
        sub = d[d["dgp"] == g]
        for m in methods:
            base = METHOD_STYLE.get(m, {"color": PALETTE["ink"], "marker": "o"})
            for k, n in enumerate(ns):
                s = sub[(sub["method"] == m) & (sub["n"] == n)].sort_values("lag")
                if s.empty:
                    continue
                ax.plot(
                    s["lag"],
                    s["rejection_rate"],
                    color=base["color"],
                    marker=base["marker"],
                    ls=_N_STYLE[k % len(_N_STYLE)],
                    alpha=0.45 + 0.55 * (k + 1) / len(ns),
                    markersize=3.6,
                )
        ax.axhline(0.05, color=PALETTE["muted"], lw=0.8, ls=(0, (4, 3)))
        ax.set_ylim(-0.03, 1.05)
        ax.set_xticks(sorted(d["lag"].unique()))
        _grid_off_x(ax)
        _panel_tag(ax, f"DGP {g}")
        ax.set_xlabel("lag order")
    for ax in flat[len(dgps):]:
        ax.set_visible(False)
    for row in axes:
        row[0].set_ylabel("empirical power")

    hm = [Line2D([], [], **{k: v for k, v in METHOD_STYLE.get(m, {}).items() if k != "ls"},
                 ls="-") for m in methods]
    hn = [Line2D([], [], color=PALETTE["ink"], ls=_N_STYLE[k % len(_N_STYLE)],
                 alpha=0.45 + 0.55 * (k + 1) / len(ns), label=f"$n = {int(n)}$")
          for k, n in enumerate(ns)]
    fig.legend(handles=hm + hn, ncol=len(hm) + len(hn), loc="lower center",
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Empirical power at the 5% nominal level", y=1.0, fontsize=11)
    fig.tight_layout()
    return fig


def plot_size_power_grid(summary_df, **kwargs):
    """Convenience wrapper returning ``(size_figure, power_figure)``."""
    return plot_size(summary_df, **kwargs.get("size", {})), plot_power(
        summary_df, **kwargs.get("power", {})
    )


def plot_pvalue_ecdf(
    mc_df,
    *,
    dgps: Sequence[str] = ("S1", "S2"),
    methods: Sequence[str] = ("drgc", "nhkj"),
    n: int | None = None,
    lags: Iterable[int] | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Davidson-MacKinnon p-value plot: ECDF of the p-values under ``H0``.

    A correctly sized test has p-values that are uniform on ``[0, 1]``, so the
    ECDF should lie on the 45-degree line.  Curves above the diagonal indicate
    over-rejection, curves below indicate a conservative test.  This is the
    single most informative size diagnostic and complements Table 3.

    Parameters
    ----------
    mc_df : DataFrame
        Replication-level output of :func:`drgct.simulate.monte_carlo`.
    dgps : sequence of str
        Null designs only.
    methods : sequence of str
    n : int, optional
        Restrict to one sample size (defaults to the largest available).
    lags : iterable of int, optional
        Panels; defaults to every lag in ``mc_df``.
    """
    import matplotlib.pyplot as plt

    use_journal_style()
    d = mc_df[mc_df["dgp"].isin(dgps)]
    if n is None:
        n = int(d["n"].max())
    d = d[d["n"] == n]
    lags = sorted(d["lag"].unique()) if lags is None else list(lags)
    dgps = [g for g in dgps if g in set(d["dgp"])]

    nrow, ncol = len(dgps), len(lags)
    figsize = figsize or (2.0 * ncol + 0.8, 2.1 * nrow + 0.8)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, sharex=True, sharey=True, squeeze=False)

    for i, g in enumerate(dgps):
        for j, k in enumerate(lags):
            ax = axes[i][j]
            ax.plot([0, 1], [0, 1], color=PALETTE["muted"], lw=0.8, ls=(0, (4, 3)), zorder=1)
            for m in methods:
                s = d[(d["dgp"] == g) & (d["lag"] == k) & (d["method"] == m)]["pvalue"].to_numpy()
                if s.size == 0:
                    continue
                s = np.sort(s)
                ecdf = np.arange(1, s.size + 1) / s.size
                st = METHOD_STYLE.get(m, {})
                ax.step(s, ecdf, where="post", color=st.get("color", PALETTE["ink"]),
                        ls=st.get("ls", "-"), label=st.get("label", m), zorder=3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            if i == 0:
                _panel_tag(ax, f"lag {int(k)}")
            if j == 0:
                ax.set_ylabel(f"DGP {g}\nECDF")
            if i == nrow - 1:
                ax.set_xlabel("p-value")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(labels), loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"p-value plots under the null, $n = {n}$", y=1.0, fontsize=11)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Single-test diagnostics
# --------------------------------------------------------------------------- #
def plot_bootstrap_distribution(
    result,
    *,
    bins: int = 45,
    alphas: Sequence[float] = (0.10, 0.05, 0.01),
    figsize: tuple[float, float] = (5.4, 3.2),
):
    """Bootstrap null distribution of ``KS*_n`` with the observed ``KS_n``.

    This is the figure to put next to any single reported test: it shows the
    reader exactly where the statistic falls in its resampled null and makes
    the p-value visually auditable.
    """
    import matplotlib.pyplot as plt

    use_journal_style()
    fig, ax = plt.subplots(figsize=figsize)
    b = np.asarray(result.boot_stats, dtype=float)

    ax.hist(b, bins=bins, color=PALETTE["sky"], edgecolor="white", linewidth=0.4,
            alpha=0.85, zorder=2, label=r"bootstrap $KS_n^{*}$")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)  # headroom for the legend
    for a in alphas:
        cv = float(np.quantile(b, 1 - a))
        ax.axvline(cv, color=PALETTE["muted"], lw=0.8, ls=(0, (3, 2)), zorder=3)
        # Labels sit in the empty right tail, rotated, so they never collide
        # with the legend or the bars.
        ax.annotate(f"{100 * a:g}%", xy=(cv, 0), xytext=(-3, 4),
                    textcoords="offset points", rotation=90,
                    ha="right", va="bottom", fontsize=7.5, color=PALETTE["muted"])
    ax.axvline(result.ks_stat, color=PALETTE["terracotta"], lw=1.8, zorder=4,
               label=rf"observed $KS_n = {result.ks_stat:.3f}$")

    ax.set_xlabel(r"$KS_n$")
    ax.set_ylabel("frequency")
    _grid_off_x(ax)
    ax.legend(loc="upper right")
    ax.set_title(
        f"{result.direction},  lag {result.lag}   "
        rf"($p^*_n = {result.pvalue:.3f}$, $B = {len(b)}$)",
        loc="left",
    )
    fig.tight_layout()
    return fig


def plot_empirical_process(
    result,
    *,
    envelope: float = 0.95,
    figsize: tuple[float, float] = (6.6, 3.0),
):
    r"""``Re`` and ``Im`` of ``Shat_n(mu_l, nu_l)`` over the ``L`` directions.

    The grey band is the pointwise ``envelope``-level bootstrap envelope, so a
    spike escaping the band identifies *which* direction in ``W`` drives the
    rejection.  Because ``phi(W, w) = exp(i w' W)`` is generically
    comprehensively revealing, a departure at any direction is evidence
    against ``H0``.
    """
    import matplotlib.pyplot as plt

    use_journal_style()
    z = np.asarray(result.influence)
    xi_free = z.sum(axis=0) / np.sqrt(z.shape[0])
    L = xi_free.size
    ell = np.arange(1, L + 1)

    # Rebuild a direction-wise bootstrap envelope from the stored influence terms.
    rng = np.random.default_rng(12345)
    xi = rng.integers(0, 2, size=(2000, z.shape[0])).astype(float) * 2 - 1
    S_boot = (xi @ z) / np.sqrt(z.shape[0])
    hi_r = np.quantile(np.abs(S_boot.real), envelope, axis=0)
    hi_i = np.quantile(np.abs(S_boot.imag), envelope, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    for ax, vals, band, tag in (
        (axes[0], xi_free.real, hi_r, r"$\mathrm{Re}\,\hat{S}_n(\mu_\ell,\nu_\ell)$"),
        (axes[1], xi_free.imag, hi_i, r"$\mathrm{Im}\,\hat{S}_n(\mu_\ell,\nu_\ell)$"),
    ):
        ax.fill_between(ell, -band, band, color=PALETTE["rule"], alpha=0.45, lw=0,
                        label=f"{int(100 * envelope)}% bootstrap envelope", zorder=1)
        ax.vlines(ell, 0, vals, color=PALETTE["blue"], lw=1.1, zorder=2)
        ax.plot(ell, vals, "o", color=PALETTE["blue"], ms=3.2, zorder=3)
        out = np.abs(vals) > band
        if out.any():
            ax.plot(ell[out], vals[out], "o", color=PALETTE["terracotta"], ms=4.6, zorder=4)
        ax.axhline(0, color=PALETTE["ink"], lw=0.7)
        ax.set_xlabel(r"direction index $\ell$")
        _panel_tag(ax, tag)
        _grid_off_x(ax)
    axes[0].set_ylabel("process value")
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D

    handles.append(Line2D([], [], marker="o", ls="none", ms=4.6,
                          color=PALETTE["terracotta"]))
    labels.append("direction outside the envelope")
    fig.legend(handles, labels, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(
        f"Feasible empirical process, {result.direction}, lag {result.lag}", y=1.02, fontsize=11
    )
    fig.tight_layout()
    return fig


def plot_training_curves(result, *, figsize: tuple[float, float] = (5.6, 2.6)):
    """MLP and MDN training-loss trajectories, for convergence auditing."""
    import matplotlib.pyplot as plt

    use_journal_style()
    mlp = result.settings.get("mlp_history") or []
    mdn = result.settings.get("mdn_history") or []
    ncol = 1 + int(bool(mdn))
    fig, axes = plt.subplots(1, ncol, figsize=figsize, squeeze=False)
    axes = axes[0]

    axes[0].plot(np.arange(1, len(mlp) + 1), mlp, color=PALETTE["blue"])
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("training loss")
    _panel_tag(axes[0], r"MLP  ($\hat{m}(Y_{t-1})$, $L_2$)")
    _grid_off_x(axes[0])
    if mdn:
        axes[1].plot(np.arange(1, len(mdn) + 1), mdn, color=PALETTE["green"])
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("negative log-likelihood")
        _panel_tag(axes[1], r"MDN  ($\hat{f}_{X_{t-1}|Y_{t-1}}$)")
        _grid_off_x(axes[1])
    fig.tight_layout()
    return fig


def plot_mdn_fit(
    result,
    *,
    coordinate: int = 0,
    bins: int = 40,
    figsize: tuple[float, float] = (5.4, 3.0),
):
    """Diagnostic for Step 2: MDN pseudo-samples against the observed ``X`` lag.

    Requires ``drgc_test(..., return_networks=True)``.  If the marginal of the
    pooled pseudo-samples does not track the empirical marginal of
    ``X_{t-coordinate-1}``, ``G`` is probably too small.
    """
    import matplotlib.pyplot as plt

    use_journal_style()
    nets = result.settings.get("networks")
    if not nets or nets.get("mdn_samples") is None:
        raise ValueError("Re-run drgc_test(..., return_networks=True) to use plot_mdn_fit.")
    obs = np.asarray(nets["xlag_std"])[:, coordinate]
    sim = np.asarray(nets["mdn_samples"])[:, :, coordinate].ravel()

    fig, ax = plt.subplots(figsize=figsize)
    lo, hi = np.percentile(np.r_[obs, sim], [0.2, 99.8])
    grid = np.linspace(lo, hi, bins + 1)
    ax.hist(obs, bins=grid, density=True, color=PALETTE["sky"], alpha=0.8,
            edgecolor="white", lw=0.4, label=rf"observed $X_{{t-{coordinate + 1}}}$ (standardised)")
    ax.hist(sim, bins=grid, density=True, histtype="step", color=PALETTE["terracotta"],
            lw=1.5, label=rf"MDN draws, $G = {result.settings.get('G')}$, $M = {result.settings.get('M')}$")
    ax.set_xlabel("standardised value")
    ax.set_ylabel("density")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.28)  # headroom for the legend
    _grid_off_x(ax)
    ax.legend(loc="upper right")
    ax.set_title("Mixture density network fit (Step 2)", loc="left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Application figures
# --------------------------------------------------------------------------- #
def plot_lag_profile(
    scan_df,
    *,
    alpha: float = 0.05,
    label: str = "",
    figsize: tuple[float, float] = (5.6, 3.0),
    show_stat: bool = True,
):
    """p-value (and optionally ``KS_n``) as a function of the lag order.

    Parameters
    ----------
    scan_df : DataFrame
        First element of the tuple returned by
        :func:`drgct.drgc_lag_scan`; needs ``lag``, ``pvalue`` and,
        for ``show_stat``, ``ks_stat``.
    """
    import matplotlib.pyplot as plt

    use_journal_style()
    d = scan_df.sort_values("lag")
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhspan(0, alpha, color=PALETTE["sand"], alpha=0.6, lw=0, zorder=0)
    ax.axhline(alpha, color=PALETTE["muted"], lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.plot(d["lag"], d["pvalue"], "-o", color=PALETTE["blue"], zorder=3,
            label="bootstrap p-value")
    sig = d[d["pvalue"] < alpha]
    if len(sig):
        ax.plot(sig["lag"], sig["pvalue"], "o", color=PALETTE["terracotta"], ms=6,
                zorder=4, label=f"reject at {100 * alpha:g}%")
    ax.set_xlabel("lag order  $p = q$")
    ax.set_ylabel("bootstrap p-value")
    ax.set_xticks(list(d["lag"]))
    ax.set_ylim(-0.02, 1.02)
    _grid_off_x(ax)

    if show_stat and "ks_stat" in d:
        ax2 = ax.twinx()
        ax2.plot(d["lag"], d["ks_stat"], "-s", color=PALETTE["green"], ms=3.2,
                 lw=1.0, alpha=0.85, label=r"$KS_n$")
        ax2.set_ylabel(r"$KS_n$", color=PALETTE["green"])
        ax2.tick_params(axis="y", colors=PALETTE["green"])
        ax2.grid(False)
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(PALETTE["green"])
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = h1 + h2, l1 + l2
    else:
        handles, labels = ax.get_legend_handles_labels()
    # Below the axes: a p-value profile often runs right through the top-right
    # corner, so an in-axes legend would sit on top of the data.
    fig.legend(handles, labels, ncol=len(labels), loc="lower center",
               bbox_to_anchor=(0.5, -0.04))
    if label:
        ax.set_title(label, loc="left")
    fig.tight_layout()
    return fig


def plot_pvalue_heatmap(
    app_df,
    *,
    alpha: float = 0.05,
    directions: Sequence[str] | None = None,
    index_order: Sequence[str] = ("SPX 500", "CSI 300", "NI 225"),
    figsize: tuple[float, float] | None = None,
    annotate: bool = True,
):
    """Lag-by-period map of DRGCT p-values, one panel per causality direction.

    This is the graphical counterpart to Table 6: a reader sees at a glance
    which index, which sub-sample and which lag orders carry the causality.
    Significant cells are outlined so the figure survives greyscale printing.

    Parameters
    ----------
    app_df : DataFrame
        Long results with ``index_label, period, direction, lag, pvalue``.
    alpha : float
    directions : sequence of str, optional
    index_order : sequence of str
    annotate : bool
        Print the p-value inside each cell.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.patches import Rectangle

    use_journal_style()
    directions = directions or list(dict.fromkeys(app_df["direction"]))
    lags = sorted(app_df["lag"].unique())

    # A light, airy diverging scheme: warm coral on the rejection side, warm
    # off-white at the threshold, soft dusty blue above it.  TwoSlopeNorm maps
    # [0, alpha] -> [0, 0.5] and [alpha, 1] -> [0.5, 1] in colormap space, so
    # the stops below are in *normalised* units and everything left of 0.5 is
    # the rejection region.
    cmap = LinearSegmentedColormap.from_list(
        "drgct_p",
        [(0.00, "#E27C68"), (0.22, "#F0A48F"), (0.42, "#F8CFBF"),
         (0.50, "#FBF7F2"), (0.60, "#E2ECF0"), (0.80, "#BBD3DE"), (1.00, "#93B4C6")],
    )
    norm = TwoSlopeNorm(vmin=0.0, vcenter=alpha, vmax=1.0)

    rows = []
    for d in directions:
        sub = app_df[app_df["direction"] == d]
        for idx in [i for i in index_order if i in set(sub["index_label"])]:
            for per in sorted(sub[sub["index_label"] == idx]["period"].unique()):
                rows.append((d, idx, per))
    per_dir = {d: [r for r in rows if r[0] == d] for d in directions}
    nrow_max = max(len(v) for v in per_dir.values())
    figsize = figsize or (0.62 * len(lags) + 3.2, 0.42 * nrow_max * len(directions) + 1.6)

    fig, axes = plt.subplots(
        len(directions), 1, figsize=figsize,
        gridspec_kw={"height_ratios": [len(per_dir[d]) for d in directions]},
        squeeze=False,
    )
    im = None
    for ax, d in zip((a[0] for a in axes), directions):
        keys = per_dir[d]
        mat = np.full((len(keys), len(lags)), np.nan)
        for r, (_, idx, per) in enumerate(keys):
            for c, k in enumerate(lags):
                sel = app_df[
                    (app_df["direction"] == d)
                    & (app_df["index_label"] == idx)
                    & (app_df["period"] == per)
                    & (app_df["lag"] == k)
                ]["pvalue"]
                if len(sel):
                    mat[r, c] = float(sel.iloc[0])
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(lags)), [str(k) for k in lags])
        ax.set_yticks(range(len(keys)), [f"{idx}  {per}" for _, idx, per in keys])
        ax.set_ylabel("")
        _panel_tag(ax, d)
        # Thin white separators between cells: airier, and they keep the
        # rejection outlines legible where several adjacent cells reject.
        ax.set_xticks(np.arange(-0.5, len(lags), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(keys), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.6)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                v = mat[r, c]
                if not np.isfinite(v):
                    continue
                if v < alpha:
                    ax.add_patch(
                        Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                  edgecolor="#A8412C", lw=1.3, zorder=5)
                    )
                if annotate:
                    # Every cell is light, so a single dark ink reads cleanly
                    # throughout; bold marks the rejections.
                    ax.text(c, r, f"{v:.2f}".lstrip("0") if v < 1 else "1.0",
                            ha="center", va="center", fontsize=6.8, zorder=6,
                            color="#5E2415" if v < alpha else "#33414B",
                            fontweight="bold" if v < alpha else "normal")
    axes[-1][0].set_xlabel("lag order")
    cbar = fig.colorbar(im, ax=[a[0] for a in axes], fraction=0.03, pad=0.02)
    cbar.set_label("bootstrap p-value")
    cbar.ax.axhline(alpha, color=PALETTE["ink"], lw=1.0)
    fig.suptitle(
        f"DRGCT p-values by lag order (outlined cells reject at {100 * alpha:g}%)",
        y=1.0, fontsize=11,
    )
    return fig


def plot_series_overview(
    raw_map: dict,
    transformed_map: dict | None = None,
    *,
    figsize: tuple[float, float] | None = None,
):
    """Four-column overview of the raw and transformed data for each index.

    Parameters
    ----------
    raw_map : dict
        ``{label: DataFrame with Close and Volume}``.
    transformed_map : dict, optional
        ``{label: DataFrame with P and V}`` from
        :func:`drgct.datasets.to_percentage_changes`.
    """
    import matplotlib.pyplot as plt

    use_journal_style()
    labels = list(raw_map)
    ncol = 2 + (2 if transformed_map else 0)
    figsize = figsize or (3.0 * ncol, 1.9 * len(labels) + 0.7)
    fig, axes = plt.subplots(len(labels), ncol, figsize=figsize, squeeze=False)

    for i, lab in enumerate(labels):
        raw = raw_map[lab]
        axes[i][0].plot(raw.index, raw["Close"], color=PALETTE["blue"], lw=0.9)
        axes[i][0].set_ylabel(f"{lab}")
        axes[i][1].plot(raw.index, raw["Volume"] / 1e9, color=PALETTE["green"], lw=0.6, alpha=0.9)
        if i == 0:
            _panel_tag(axes[i][0], "closing level")
            _panel_tag(axes[i][1], "volume (bn)")
        if transformed_map:
            tr = transformed_map[lab]
            axes[i][2].plot(tr.index, tr["P"], color=PALETTE["blue"], lw=0.5, alpha=0.9)
            axes[i][3].plot(tr.index, tr["V"], color=PALETTE["terracotta"], lw=0.5, alpha=0.9)
            if i == 0:
                _panel_tag(axes[i][2], r"$P_t$  (% change)")
                _panel_tag(axes[i][3], r"$V_t$  (% change / 10)")
        for j in range(ncol):
            _grid_off_x(axes[i][j])
            if i < len(labels) - 1:
                axes[i][j].set_xticklabels([])
            else:
                for t in axes[i][j].get_xticklabels():
                    t.set_rotation(30)
                    t.set_horizontalalignment("right")
    fig.tight_layout()
    return fig


def plot_stability(
    stability,
    *,
    label: str = "",
    bins: int = 20,
    figsize: tuple[float, float] = (5.8, 3.0),
):
    """Distribution of the p-value across independent random-direction draws.

    Parameters
    ----------
    stability : dict
        Output of :func:`drgct.drgc_stability`.
    label : str
    bins : int

    Notes
    -----
    A tight cluster well below ``alpha`` (or well above) means the conclusion
    is robust to the ``(mu_l, nu_l)`` draw.  A distribution straddling
    ``alpha`` is a warning: raise ``L``, and report the merged p-value rather
    than a single run.
    """
    import matplotlib.pyplot as plt

    use_journal_style()
    p = np.asarray(stability["pvalues"], dtype=float)
    alpha = float(stability.get("alpha", 0.05))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvspan(0, alpha, color=PALETTE["sand"], alpha=0.6, lw=0, zorder=0)
    ax.hist(p, bins=np.linspace(0, 1, bins + 1), color=PALETTE["sky"],
            edgecolor="white", lw=0.4, zorder=2, label="p-value per draw")
    ax.axvline(stability["median"], color=PALETTE["blue"], lw=1.6, zorder=3,
               label=f"median = {stability['median']:.3f}")
    ax.axvline(stability["merged_pvalue"], color=PALETTE["terracotta"], lw=1.6,
               ls=(0, (4, 2)), zorder=3,
               label=f"merged (2$\\times$median) = {stability['merged_pvalue']:.3f}")
    ax.axvline(alpha, color=PALETTE["muted"], lw=0.9, ls=(0, (2, 2)), zorder=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("bootstrap p-value")
    ax.set_ylabel("draws")
    _grid_off_x(ax)
    ax.legend(loc="upper right")
    share = 100 * stability["share_reject"]
    ax.set_title(
        (label or stability.get("direction", ""))
        + f"   lag {stability['lag']},  {stability['n_draws']} direction draws,  "
        f"{share:.0f}% reject at {100 * alpha:g}%",
        loc="left",
    )
    fig.tight_layout()
    return fig


def plot_rolling_pvalue(
    roll_df,
    *,
    alpha: float = 0.05,
    label: str = "",
    figsize: tuple[float, float] = (6.4, 3.0),
):
    """Rolling-window DRGCT p-values plotted against the window end date.

    Parameters
    ----------
    roll_df : DataFrame
        Needs ``end_date``, ``pvalue`` and (optionally) ``direction``.
    """
    import matplotlib.pyplot as plt

    use_journal_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhspan(0, alpha, color=PALETTE["sand"], alpha=0.6, lw=0, zorder=0)
    ax.axhline(alpha, color=PALETTE["muted"], lw=0.8, ls=(0, (4, 3)), zorder=1)

    if "direction" in roll_df.columns and roll_df["direction"].nunique() > 1:
        colors = [PALETTE["blue"], PALETTE["terracotta"], PALETTE["green"], PALETTE["purple"]]
        for c, (name, g) in zip(colors, roll_df.groupby("direction")):
            g = g.sort_values("end_date")
            ax.plot(g["end_date"], g["pvalue"], lw=1.2, color=c, label=name, zorder=3)
        ax.legend(loc="upper right")
    else:
        g = roll_df.sort_values("end_date")
        ax.plot(g["end_date"], g["pvalue"], lw=1.2, color=PALETTE["blue"], zorder=3)

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("window end date")
    ax.set_ylabel("bootstrap p-value")
    _grid_off_x(ax)
    for t in ax.get_xticklabels():
        t.set_rotation(30)
        t.set_horizontalalignment("right")
    if label:
        ax.set_title(label, loc="left")
    fig.tight_layout()
    return fig
