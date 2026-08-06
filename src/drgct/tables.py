r"""Publication-quality tables.

Every builder here returns a plain :class:`pandas.DataFrame` (so you can keep
working with the numbers) and can be pushed through :func:`export_table`,
which writes three files at once:

* ``<stem>.tex`` -- a ``booktabs`` ``table`` environment ready to
  ``\input{}`` into a manuscript (``\usepackage{booktabs}`` required);
* ``<stem>.md``  -- a GitHub-flavoured Markdown table for READMEs and issues;
* ``<stem>.csv`` -- the raw numbers.

The layouts mirror the six tables of Hui, Liu and Song (2025):

===========  ==========================================================
Paper table  Builder
===========  ==========================================================
Table 1      :func:`table_dgp_definitions`
Table 2      :func:`table_parameter_settings`
Table 3      :func:`table_size`
Table 4      :func:`table_power`
Table 5      :func:`table_detection`
Table 6      :func:`table_lag_orders`
(new)        :func:`table_descriptives`, :func:`table_hyperparameters`
===========  ==========================================================
"""

from __future__ import annotations

import pathlib
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "TICK",
    "CROSS",
    "fmt_rate",
    "to_latex_booktabs",
    "export_table",
    "table_dgp_definitions",
    "table_parameter_settings",
    "table_size",
    "table_power",
    "table_detection",
    "table_lag_orders",
    "table_descriptives",
    "table_hyperparameters",
]

TICK = "✓"  # check mark: causality detected
CROSS = "✗"  # ballot X: no causality detected

_TEX_TICK = r"\checkmark"
_TEX_CROSS = r"$\times$"


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def fmt_rate(v, digits: int = 3) -> str:
    """Format a rejection frequency as the paper does (``0.051``, ``1.000``)."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "/"
    return f"{float(v):.{digits}f}"


def _escape_latex(s: str) -> str:
    if s.startswith("$") and s.endswith("$"):
        return s  # already math
    out = str(s)
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        out = out.replace(a, b)
    return out.replace(TICK, _TEX_TICK).replace(CROSS, _TEX_CROSS)


def to_latex_booktabs(
    df,
    *,
    caption: str = "",
    label: str = "",
    notes: str = "",
    align: str | None = None,
    index: bool = False,
    float_format: str = "%.3f",
    position: str = "htbp",
    small: bool = True,
    escape: bool = True,
) -> str:
    r"""Render ``df`` as a ``booktabs`` table.

    Parameters
    ----------
    df : DataFrame
    caption, label : str
        ``\caption{}`` text and ``\label{}`` key.
    notes : str
        Table notes typeset below the rule in ``\footnotesize``.
    align : str, optional
        Column specification, e.g. ``"lcccc"``.  Defaults to left-aligning
        object columns and centring numeric ones.
    index : bool
        Include the DataFrame index (flattened if it is a MultiIndex).
    float_format : str
    position : str
    small : bool
        Wrap the tabular in ``\small``.
    escape : bool
        Escape ``&``, ``%``, ``_``, ``#`` and translate the tick/cross glyphs.

    Returns
    -------
    str
    """
    import pandas as pd

    d = df.copy()
    if index:
        d = d.reset_index()
    if align is None:
        align = "".join("l" if d[c].dtype == object else "c" for c in d.columns)

    def cell(v):
        if isinstance(v, float):
            return "" if not np.isfinite(v) else float_format % v
        s = str(v)
        return _escape_latex(s) if escape else s

    head = " & ".join(_escape_latex(str(c)) if escape else str(c) for c in d.columns)
    body = " \\\\\n".join(" & ".join(cell(v) for v in row) for row in d.itertuples(index=False))

    lines = [
        f"\\begin{{table}}[{position}]",
        "\\centering",
    ]
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    if small:
        lines.append("\\small")
    lines += [
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        head + " \\\\",
        "\\midrule",
        body + " \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    if notes:
        lines += [
            "",
            "\\begin{minipage}{\\linewidth}\\footnotesize",
            notes,
            "\\end{minipage}",
        ]
    lines.append("\\end{table}")
    return "\n".join(lines)


def export_table(
    df,
    stem: str,
    outdir: str | pathlib.Path = "results/tables",
    *,
    caption: str = "",
    label: str = "",
    notes: str = "",
    index: bool = False,
    float_format: str = "%.3f",
    quiet: bool = False,
    **latex_kwargs,
) -> dict:
    """Write ``df`` to ``<outdir>/<stem>.{tex,md,csv}`` and return the paths."""
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": outdir / f"{stem}.csv",
        "md": outdir / f"{stem}.md",
        "tex": outdir / f"{stem}.tex",
    }
    df.to_csv(paths["csv"], index=index)

    md = df.to_markdown(index=index, floatfmt=float_format.lstrip("%").replace("f", "f"))
    header = f"**{caption}**\n\n" if caption else ""
    footer = f"\n\n_{notes}_\n" if notes else "\n"
    paths["md"].write_text(header + md + footer, encoding="utf-8")

    paths["tex"].write_text(
        to_latex_booktabs(
            df,
            caption=caption,
            label=label,
            notes=notes,
            index=index,
            float_format=float_format,
            **latex_kwargs,
        )
        + "\n",
        encoding="utf-8",
    )
    if not quiet:
        print(f"  [table] {stem}: " + ", ".join(str(p.name) for p in paths.values()))
    return {k: str(v) for k, v in paths.items()}


# --------------------------------------------------------------------------- #
# Table 1 and Table 2 -- the designs
# --------------------------------------------------------------------------- #
def table_dgp_definitions():
    """Table 1: the six data generating processes."""
    from .dgp import dgp_table

    return dgp_table()


def table_parameter_settings(lags: Sequence[int] = (1, 2, 3, 4, 5)):
    """Table 2: coefficient settings by lag order."""
    from .dgp import parameter_table

    return parameter_table(lags)


# --------------------------------------------------------------------------- #
# Table 3 and Table 4 -- size and power
# --------------------------------------------------------------------------- #
def _rejection_block(mc_df, dgps, methods, *, alpha, digits):
    """Long -> wide with rows ``(lag, n)`` and columns ``(DGP, method)``."""
    import pandas as pd

    from .simulate import summarize

    s = summarize(mc_df, alpha=alpha)
    label = {"drgc": "DRGC", "drgc_naive": "DRGC-naive", "nhkj": "NHKJ"}
    s = s[s["method"].isin(methods) & s["dgp"].isin(dgps)].copy()
    s["method"] = s["method"].map(lambda m: label.get(m, m))
    if s.empty:
        return pd.DataFrame()

    wide = s.pivot_table(index=["lag", "n"], columns=["dgp", "method"],
                         values="rejection_rate").sort_index()
    # Preserve the caller's DGP order and the paper's method order.
    cols = [(d, label.get(m, m)) for d in dgps for m in methods
            if (d, label.get(m, m)) in wide.columns]
    wide = wide[cols]

    out = wide.reset_index()
    out.columns = (
        ["Lag", "Sample size"]
        + [f"{d} {m}" for d, m in cols]
    )
    out["Sample size"] = out["Sample size"].map(lambda v: f"n = {int(v)}")
    for c in out.columns[2:]:
        out[c] = out[c].map(lambda v: fmt_rate(v, digits))
    return out


def table_size(
    mc_df,
    *,
    dgps: Sequence[str] = ("S1", "S2"),
    methods: Sequence[str] = ("drgc", "nhkj"),
    alpha: float | None = 0.05,
    digits: int = 3,
):
    """Table 3: empirical sizes under varying lags.

    Parameters
    ----------
    mc_df : DataFrame
        Output of :func:`drgct.simulate.monte_carlo`.
    dgps : sequence of str
        Null designs to include.
    methods : sequence of str
        Any of ``'drgc'``, ``'drgc_naive'``, ``'nhkj'``.  Adding
        ``'drgc_naive'`` reproduces the paper's point that a naive deep
        plug-in loses control of the type I error.
    alpha : float, optional
        Recompute rejections at this level (``None`` keeps the stored flags).
    digits : int
    """
    return _rejection_block(mc_df, list(dgps), list(methods), alpha=alpha, digits=digits)


def table_power(
    mc_df,
    *,
    dgps: Sequence[str] = ("P1", "P2", "P3", "P4"),
    methods: Sequence[str] = ("drgc", "nhkj"),
    alpha: float | None = 0.05,
    digits: int = 3,
):
    """Table 4: empirical powers under varying lags."""
    return _rejection_block(mc_df, list(dgps), list(methods), alpha=alpha, digits=digits)


# --------------------------------------------------------------------------- #
# Table 5 and Table 6 -- the application
# --------------------------------------------------------------------------- #
def table_detection(app_df, *, alpha: float = 0.05, rule: str = "any"):
    r"""Table 5: price-volume Granger causality detection, one tick per cell.

    Parameters
    ----------
    app_df : DataFrame
        Long results with columns ``index_label, period, direction, lag,
        pvalue``.  This is what
        :func:`drgct.applications.price_volume_study` returns.
    alpha : float, default 0.05
        "The upper 5% critical value is employed as the threshold."
    rule : {'any', 'majority', 'all'}
        How lag-specific decisions aggregate into the single tick of Table 5.
        ``'any'`` (the paper's reading -- causality is declared if it shows up
        at some lag order), ``'majority'`` (more than half the lags reject),
        ``'all'`` (every lag rejects).

    Returns
    -------
    pandas.DataFrame
        Rows = (direction, period), columns = index label, cells = tick/cross.
    """
    import pandas as pd

    d = app_df.copy()
    d["reject"] = d["pvalue"] < float(alpha)
    agg = {"any": "max", "all": "min", "majority": "mean"}[rule]
    g = d.groupby(["direction", "period", "index_label"], as_index=False)["reject"].agg(agg)
    if rule == "majority":
        g["reject"] = g["reject"] > 0.5
    g["mark"] = np.where(g["reject"].astype(bool), TICK, CROSS)

    wide = g.pivot_table(index=["direction", "period"], columns="index_label",
                         values="mark", aggfunc="first")
    order = [c for c in ("SPX 500", "CSI 300", "NI 225") if c in wide.columns]
    wide = wide[order + [c for c in wide.columns if c not in order]]
    out = wide.reset_index().rename(columns={"direction": "Causality direction",
                                             "period": "Period"})
    out.columns.name = None
    return out


def table_lag_orders(app_df, *, alpha: float = 0.05, lags: Iterable[int] | None = None):
    """Table 6: price-volume Granger causality under specific lag orders.

    Parameters
    ----------
    app_df : DataFrame
        As in :func:`table_detection`.
    alpha : float, default 0.05
    lags : iterable of int, optional
        Column order; defaults to the sorted lags present in ``app_df``.

    Returns
    -------
    pandas.DataFrame
        Rows = (direction, index, period), one column per lag order.
    """
    import pandas as pd

    d = app_df.copy()
    d["mark"] = np.where(d["pvalue"] < float(alpha), TICK, CROSS)
    wide = d.pivot_table(index=["direction", "index_label", "period"], columns="lag",
                         values="mark", aggfunc="first")
    if lags is not None:
        wide = wide.reindex(columns=list(lags))
    wide = wide.sort_index()
    out = wide.reset_index().rename(
        columns={"direction": "Causality direction", "index_label": "Index", "period": "Period"}
    )
    out.columns = [c if not isinstance(c, (int, np.integer)) else int(c) for c in out.columns]
    out.columns.name = None
    return out


def table_pvalues(app_df, *, digits: int = 3, lags: Iterable[int] | None = None):
    """A p-value companion to Table 6 -- same layout, numbers instead of ticks."""
    import pandas as pd

    wide = app_df.pivot_table(
        index=["direction", "index_label", "period"], columns="lag", values="pvalue"
    )
    if lags is not None:
        wide = wide.reindex(columns=list(lags))
    out = wide.sort_index().round(digits).reset_index()
    out = out.rename(columns={"direction": "Causality direction",
                              "index_label": "Index", "period": "Period"})
    out.columns.name = None
    return out


# --------------------------------------------------------------------------- #
# Supporting tables
# --------------------------------------------------------------------------- #
def table_descriptives(series_map: dict, *, digits: int = 3):
    """Descriptive statistics block for the empirical section."""
    from .datasets import describe

    d = describe(series_map)
    return d.round(digits).reset_index().rename(columns={"index": "Statistic"})


def table_hyperparameters(result_or_settings) -> "object":
    """One-column table recording every hyper-parameter of a fitted test.

    Reviewers ask for this; give it to them.
    """
    import pandas as pd

    s = getattr(result_or_settings, "settings", result_or_settings)
    rows = [
        ("Mixture components $G$", s.get("G")),
        ("Random directions $L$", s.get("L")),
        ("Pseudo-samples $M$", s.get("M")),
        ("Bootstrap replications $B$", s.get("B")),
        ("Support of $(\\mu,\\nu)$", f"U[{s.get('w_lower')}, {s.get('w_upper')}]"),
        ("Bootstrap multipliers", s.get("multiplier")),
        ("MLP width / depth", f"{s.get('mlp_width')} / {s.get('mlp_depth')}"),
        ("MDN width / depth", f"{s.get('mdn_width')} / {s.get('mdn_depth')}"),
        ("MLP loss", s.get("mlp_loss")),
        ("Inputs standardised", s.get("standardize")),
        ("Seed", s.get("seed")),
    ]
    return pd.DataFrame(rows, columns=["Setting", "Value"])
