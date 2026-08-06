r"""Bundled economic data and the transformations of Section 5 of the paper.

Three daily index series covering **27 September 2019 - 26 September 2024**
ship with the package so that every example in this repository runs offline:

=============  ============================  =============================
Key            Index                         Source ticker
=============  ============================  =============================
``spx500``     S&P 500                       ``^GSPC``
``csi300``     CSI 300 (ETF tracker)         ``510300.SS``
``nikkei225``  Nikkei 225                    ``^N225``
=============  ============================  =============================

See ``src/drgct/data/SOURCES.md`` for the provenance record and
``data/fetch_data.py`` to refresh them or add your own series.

Transformation
--------------
Section 5: "To ensure stationarity, both prices and volumes are transformed
into percentage changes.  For volumes, the percentage change is divided by 10
to conform model training and achieve a scale comparable to that of stock
prices."  :func:`to_percentage_changes` implements exactly that and returns
``P_t`` (price changes) and ``V_t`` (scaled volume changes).

Sub-samples
-----------
The paper splits the five-year window into three overlapping three-year
sub-samples -- 2019-2022, 2020-2023, 2021-2024, each of roughly ``n = 750``
observations -- to check whether causality patterns are stable.
:data:`PAPER_PERIODS` and :func:`subsample` reproduce that split.
"""

from __future__ import annotations

import pathlib
from importlib import resources
from typing import Iterable

import numpy as np

__all__ = [
    "INDEX_KEYS",
    "MACRO_SERIES",
    "MACRO_TRANSFORMS",
    "MACRO_PERIODS",
    "load_macro",
    "INDEX_LABELS",
    "PAPER_PERIODS",
    "PAPER_START",
    "PAPER_END",
    "data_dir",
    "available_datasets",
    "load_index",
    "load_all",
    "to_percentage_changes",
    "subsample",
    "describe",
]

INDEX_KEYS = ("spx500", "csi300", "nikkei225")

INDEX_LABELS = {
    "spx500": "SPX 500",
    "csi300": "CSI 300",
    "nikkei225": "NI 225",
}

#: FRED series in the bundled US macroeconomic dataset, with the label used in
#: tables and figures and the transformation that renders them stationary.
MACRO_SERIES = {
    "INDPRO":   ("Industrial production", "logdiff"),
    "CPIAUCSL": ("CPI inflation",         "logdiff"),
    "PCEPI":    ("PCE inflation",         "logdiff"),
    "FEDFUNDS": ("Fed funds rate",        "diff"),
    "M2SL":     ("M2 money growth",       "logdiff"),
    "UNRATE":   ("Unemployment rate",     "diff"),
    "PAYEMS":   ("Nonfarm payrolls",      "logdiff"),
    "WTISPLC":  ("WTI oil price",         "logdiff"),
}

#: How each transformation is described in a table note.
MACRO_TRANSFORMS = {
    "logdiff": r"$100\,\Delta\log$ (monthly growth, %)",
    "diff": r"$\Delta$ (monthly change, percentage points)",
    "level": "level",
}

#: Sub-samples used in the macro application.  The break is the conventional
#: Great Moderation date: Volcker's disinflation is complete, and the
#: volatility of US output growth drops sharply (McConnell and Perez-Quiros,
#: 2000; Stock and Watson, 2002).
MACRO_PERIODS = {
    "Full sample": ("1959-02-01", "2025-12-01"),
    "Great Inflation 1959-1983": ("1959-02-01", "1983-12-01"),
    "Great Moderation 1984-2025": ("1984-01-01", "2025-12-01"),
}

PAPER_START = "2019-09-27"
PAPER_END = "2024-09-26"

#: The three overlapping three-year windows of Section 5.
PAPER_PERIODS = {
    "2019-2022": ("2019-09-27", "2022-09-26"),
    "2020-2023": ("2020-09-27", "2023-09-26"),
    "2021-2024": ("2021-09-27", "2024-09-26"),
}


# --------------------------------------------------------------------------- #
# Locating the files
# --------------------------------------------------------------------------- #
def data_dir() -> pathlib.Path:
    """Directory holding the bundled CSVs.

    Resolution order: the installed package's ``drgct/data``, then a
    ``data/`` folder next to the current working directory (handy when
    running straight from a git clone).
    """
    try:
        p = pathlib.Path(str(resources.files("drgct") / "data"))
        if p.exists():
            return p
    except Exception:  # pragma: no cover
        pass
    for cand in (pathlib.Path.cwd() / "data", pathlib.Path(__file__).resolve().parent / "data"):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "Could not locate the drgct data directory.  Run `python data/fetch_data.py` "
        "from the repository root to (re)build it."
    )


def available_datasets() -> list[str]:
    """Keys of every CSV found in :func:`data_dir`."""
    return sorted(p.stem for p in data_dir().glob("*.csv"))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_index(
    name: str,
    *,
    start: str | None = None,
    end: str | None = None,
    path: str | pathlib.Path | None = None,
):
    """Load one bundled index as a ``DatetimeIndex``-ed DataFrame.

    Parameters
    ----------
    name : str
        One of :data:`INDEX_KEYS`, or any stem returned by
        :func:`available_datasets`.
    start, end : str, optional
        ISO dates used to slice the sample (both inclusive).
    path : path-like, optional
        Load an arbitrary CSV instead; it must have a date column plus
        ``Close`` and ``Volume`` columns.

    Returns
    -------
    pandas.DataFrame
        Columns ``Close`` and ``Volume``, indexed by date, sorted, with
        non-positive volumes and missing closes dropped.

    Examples
    --------
    >>> from drgct.datasets import load_index
    >>> spx = load_index("spx500")
    >>> list(spx.columns)
    ['Close', 'Volume']
    """
    import pandas as pd

    if path is None:
        f = data_dir() / f"{name}.csv"
        if not f.exists():
            raise FileNotFoundError(
                f"No dataset {name!r}.  Available: {available_datasets()}."
            )
    else:
        f = pathlib.Path(path)

    df = pd.read_csv(f, index_col=0, parse_dates=True)
    df.index.name = "Date"
    df = df.sort_index()
    keep = [c for c in ("Close", "Volume") if c in df.columns]
    if len(keep) < 2:
        raise ValueError(f"{f} must contain 'Close' and 'Volume' columns; found {list(df.columns)}.")
    df = df[keep].astype(float)
    df = df[(df["Volume"] > 0) & df["Close"].notna()]
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    df.attrs["name"] = name
    df.attrs["label"] = INDEX_LABELS.get(name, name)
    return df


def load_all(keys: Iterable[str] = INDEX_KEYS, **kwargs) -> dict:
    """``{key: DataFrame}`` for several indices at once."""
    return {k: load_index(k, **kwargs) for k in keys}


def load_macro(
    series=None,
    *,
    start: str | None = None,
    end: str | None = None,
    transform: bool = True,
    path: str | pathlib.Path | None = None,
):
    r"""Load the bundled monthly US macroeconomic dataset from FRED.

    Eight series, 1959-01 to 2025-12, listed in :data:`MACRO_SERIES`.  See
    ``src/drgct/data/SOURCES_MACRO.md`` for provenance and
    ``data/fetch_macro.py`` to refresh or extend them.

    Parameters
    ----------
    series : sequence of str, optional
        FRED ids to keep.  Defaults to all of :data:`MACRO_SERIES`.
    start, end : str, optional
        ISO dates, both inclusive.
    transform : bool, default True
        Apply the stationarity-inducing transformation attached to each series
        in :data:`MACRO_SERIES`: ``100 * dlog`` for indices, prices, money and
        employment; a first difference for the two series already measured in
        percent (the federal funds rate and the unemployment rate).  Columns
        are renamed to the human-readable labels and the first row is dropped.
    path : path-like, optional
        Load an arbitrary CSV instead; it must have a date column and the
        requested FRED ids as columns.

    Returns
    -------
    pandas.DataFrame
        Monthly, date-indexed.  With ``transform=True`` every column is
        stationary by construction and non-finite rows are dropped.

    Examples
    --------
    >>> from drgct.datasets import load_macro
    >>> m = load_macro(["FEDFUNDS", "INDPRO"])
    >>> list(m.columns)
    ['Fed funds rate', 'Industrial production']
    >>> len(m) > 700
    True
    """
    import pandas as pd

    f = pathlib.Path(path) if path is not None else data_dir() / "us_macro.csv"
    if not f.exists():
        raise FileNotFoundError(
            f"{f} not found.  Run `python data/fetch_macro.py` to build it."
        )
    df = pd.read_csv(f, index_col=0, parse_dates=True).sort_index()
    df.index.name = "Date"

    keys = list(series) if series is not None else [k for k in MACRO_SERIES if k in df.columns]
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise ValueError(f"{missing} not in {f}; available: {list(df.columns)}.")
    df = df[keys].astype(float)

    if transform:
        out = {}
        for k in keys:
            label, how = MACRO_SERIES.get(k, (k, "logdiff"))
            col = df[k]
            if how == "logdiff":
                out[label] = 100.0 * np.log(col).diff()
            elif how == "diff":
                out[label] = col.diff()
            else:
                out[label] = col
        df = pd.DataFrame(out, index=df.index)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    df.attrs["name"] = "us_macro"
    df.attrs["transformed"] = transform
    return df


# --------------------------------------------------------------------------- #
# Transformations
# --------------------------------------------------------------------------- #
def to_percentage_changes(
    df,
    *,
    price_col: str = "Close",
    volume_col: str = "Volume",
    volume_divisor: float = 10.0,
    in_percent: bool = True,
):
    r"""Section 5 transformation: percentage changes, volume scaled by 1/10.

    .. math::
        P_t = 100 \left(\frac{\text{Close}_t}{\text{Close}_{t-1}} - 1\right),
        \qquad
        V_t = \frac{1}{10}\,100 \left(\frac{\text{Volume}_t}{\text{Volume}_{t-1}} - 1\right).

    Parameters
    ----------
    df : DataFrame
        Output of :func:`load_index`.
    price_col, volume_col : str
    volume_divisor : float, default 10.0
        The paper divides the volume percentage change by 10 so that the two
        series have a comparable scale, which helps the networks train.
    in_percent : bool, default True
        Multiply the simple returns by 100.  Set ``False`` for raw fractions;
        the DRGCT p-value is unaffected either way.

    Returns
    -------
    pandas.DataFrame
        Columns ``P`` and ``V``, with the first row (and any non-finite rows)
        removed.
    """
    import pandas as pd

    scale = 100.0 if in_percent else 1.0
    out = pd.DataFrame(
        {
            "P": scale * df[price_col].pct_change(),
            "V": scale * df[volume_col].pct_change() / float(volume_divisor),
        },
        index=df.index,
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out.attrs.update(df.attrs)
    return out


def subsample(df, period: str | tuple[str, str]):
    """Slice ``df`` to one of :data:`PAPER_PERIODS` or an explicit date pair.

    Examples
    --------
    >>> from drgct.datasets import load_index, to_percentage_changes, subsample
    >>> pv = to_percentage_changes(load_index("spx500"))
    >>> len(subsample(pv, "2019-2022")) > 700
    True
    """
    import pandas as pd

    lo, hi = PAPER_PERIODS[period] if isinstance(period, str) else period
    out = df[(df.index >= pd.Timestamp(lo)) & (df.index <= pd.Timestamp(hi))].copy()
    out.attrs.update(getattr(df, "attrs", {}))
    out.attrs["period"] = period if isinstance(period, str) else f"{lo}..{hi}"
    return out


# --------------------------------------------------------------------------- #
# Descriptives
# --------------------------------------------------------------------------- #
def describe(series_map: dict, *, add_tests: bool = True):
    """Journal-style descriptive statistics table for a dict of series.

    Parameters
    ----------
    series_map : dict
        ``{label: 1-D array_like}``.
    add_tests : bool
        Append Jarque-Bera, Ljung-Box(10) on the level and on the square,
        and ADF / KPSS p-values -- the standard block of an empirical
        finance paper's Table 1.

    Returns
    -------
    pandas.DataFrame
        Rows = statistics, columns = series.
    """
    import pandas as pd
    from scipy import stats

    cols = {}
    for label, v in series_map.items():
        s = np.asarray(v, dtype=float)
        s = s[np.isfinite(s)]
        rec = {
            "Obs.": int(s.size),
            "Mean": s.mean(),
            "Median": float(np.median(s)),
            "Std. dev.": s.std(ddof=1),
            "Min.": s.min(),
            "Max.": s.max(),
            "Skewness": float(stats.skew(s)),
            "Kurtosis": float(stats.kurtosis(s, fisher=False)),
        }
        if add_tests:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            from statsmodels.tsa.stattools import adfuller, kpss
            import warnings

            jb, jb_p = stats.jarque_bera(s)[:2]
            lb = acorr_ljungbox(s, lags=[10], return_df=True)
            lb2 = acorr_ljungbox(s**2, lags=[10], return_df=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adf_p = adfuller(s, autolag="AIC")[1]
                kpss_p = kpss(s, regression="c", nlags="auto")[1]
            rec.update(
                {
                    "Jarque-Bera": float(jb),
                    "JB p-value": float(jb_p),
                    "Ljung-Box(10)": float(lb["lb_stat"].iloc[0]),
                    "LB p-value": float(lb["lb_pvalue"].iloc[0]),
                    "Ljung-Box$^2$(10)": float(lb2["lb_stat"].iloc[0]),
                    "LB$^2$ p-value": float(lb2["lb_pvalue"].iloc[0]),
                    "ADF p-value": float(adf_p),
                    "KPSS p-value": float(kpss_p),
                }
            )
        cols[label] = rec
    return pd.DataFrame(cols)
