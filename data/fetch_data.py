#!/usr/bin/env python
"""Rebuild the bundled index price/volume CSVs used by the DRGCT application.

The application in Section 5 of Hui, Liu and Song (2025) uses **daily closing
prices and trading volumes** for three indices over
**27 September 2019 - 26 September 2024** (T = 1257 in the paper):

======================  ====================  =======================================
Index                   Yahoo! Finance ticker File written
======================  ====================  =======================================
S&P 500 (SPX 500)       ``^GSPC``             ``spx500.csv``
CSI 300                 ``510300.SS``         ``csi300.csv``
Nikkei 225 (NI 225)     ``^N225``             ``nikkei225.csv``
======================  ====================  =======================================

Why ``510300.SS`` for the CSI 300
---------------------------------
Yahoo! Finance truncates the history of the CSI 300 *index* (``000300.SS``) to
roughly the last three years, so it cannot cover 2019-2024.  ``510300.SS`` is
the Huatai-PineBridge CSI 300 ETF, the largest exchange-traded tracker of the
index: it spans the whole window and -- crucially for a price-*volume* study --
carries genuine exchange turnover.  The script tries the index first and falls
back to the tracker, recording which source it used in the CSV header comment
and in ``data/SOURCES.md``.  If you have access to a CSI 300 index feed with
volume (WIND, CSMAR, Bloomberg), drop it in as ``csi300.csv`` with the same
column names and every example in this repository will use it instead.

Usage
-----
    python data/fetch_data.py                       # paper window, default files
    python data/fetch_data.py --start 2010-01-01 --end 2024-12-31
    python data/fetch_data.py --tickers ^GSPC=spx500 ^FTSE=ftse100

Requires ``yfinance`` (``pip install "drgct[data]"``) and an internet
connection.  The repository already ships the resulting CSVs, so this script
is only needed to refresh or extend them.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
#: Canonical location of the bundled CSVs -- inside the installable package,
#: so that ``drgct.datasets.load_index`` finds them after ``pip install``.
PACKAGE_DATA = HERE.parent / "src" / "drgct" / "data"

PAPER_START = "2019-09-27"
PAPER_END = "2024-09-26"

DEFAULT_TARGETS = [
    # (preferred ticker, fallback ticker or None, output stem, pretty name)
    ("^GSPC", None, "spx500", "S&P 500 Index"),
    ("000300.SS", "510300.SS", "csi300", "CSI 300 Index"),
    ("^N225", None, "nikkei225", "Nikkei 225 Index"),
]


def _download(ticker: str, start: str, end: str):
    import pandas as pd
    import yfinance as yf

    # yfinance's `end` is exclusive; pad by one day so the last date is kept.
    end_pad = (_dt.date.fromisoformat(end) + _dt.timedelta(days=1)).isoformat()
    df = yf.download(ticker, start=start, end=end_pad, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close", "Volume"]].copy()
    df.index.name = "Date"
    df = df[(df["Volume"] > 0) & df["Close"].notna()]
    return df


def _covers_start(df, start, tolerance_days=10) -> bool:
    """True if the series actually begins at (or just after) the requested start."""
    if df is None or df.empty:
        return False
    want = _dt.date.fromisoformat(start)
    return (df.index.min().date() - want).days <= tolerance_days


def fetch_one(preferred, fallback, stem, pretty, start, end, outdir, min_rows=600):
    print(f"\n  {pretty:<20s} <- {preferred}", end="", flush=True)
    df = _download(preferred, start, end)
    source = preferred
    if df is None or len(df) < min_rows or not _covers_start(df, start):
        got = 0 if df is None else len(df)
        first = "n/a" if df is None or df.empty else str(df.index.min().date())
        if fallback is None:
            print(f"  FAILED ({got} rows, starts {first})")
            return None
        print(f"  history truncated ({got} rows, starts {first}); "
              f"falling back to {fallback}", end="", flush=True)
        df = _download(fallback, start, end)
        source = fallback
        if df is None or df.empty:
            print("  FAILED")
            return None
    path = outdir / f"{stem}.csv"
    df.round(6).to_csv(path)
    print(f"  -> {path.name}  ({len(df)} rows, {df.index.min().date()} .. {df.index.max().date()})")
    return {"stem": stem, "pretty": pretty, "ticker": source, "rows": len(df),
            "start": str(df.index.min().date()), "end": str(df.index.max().date())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=PAPER_START)
    ap.add_argument("--end", default=PAPER_END)
    ap.add_argument("--outdir", default=str(PACKAGE_DATA if PACKAGE_DATA.exists() else HERE))
    ap.add_argument("--tickers", nargs="*", default=None,
                    help="Extra TICKER=stem pairs, e.g. ^FTSE=ftse100")
    args = ap.parse_args(argv)

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    targets = list(DEFAULT_TARGETS)
    for spec in args.tickers or []:
        tick, _, stem = spec.partition("=")
        targets.append((tick, None, stem or tick.strip("^").lower(), tick))

    print(f"[drgct] downloading {args.start} .. {args.end}")
    manifest = []
    for preferred, fallback, stem, pretty in targets:
        try:
            rec = fetch_one(preferred, fallback, stem, pretty, args.start, args.end, outdir)
        except Exception as exc:  # network hiccups are the common case
            print(f"  {pretty}: ERROR {exc}")
            rec = None
        if rec:
            manifest.append(rec)

    if manifest:
        lines = ["# Data sources", "",
                 f"Downloaded from Yahoo! Finance on {_dt.date.today().isoformat()} "
                 f"by `data/fetch_data.py`.", "",
                 "| File | Index | Ticker used | Rows | First | Last |",
                 "|---|---|---|---|---|---|"]
        for m in manifest:
            lines.append(
                f"| `{m['stem']}.csv` | {m['pretty']} | `{m['ticker']}` | "
                f"{m['rows']} | {m['start']} | {m['end']} |"
            )
        lines += [
            "",
            "Columns: `Date`, `Close` (daily closing level), `Volume` (daily share/contract volume).",
            "",
            "Yahoo! Finance data is provided for personal, non-commercial use; see",
            "<https://policies.yahoo.com/us/en/yahoo/terms/product-atos/apiforydn/index.htm>.",
        ]
        (outdir / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[drgct] wrote {outdir / 'SOURCES.md'}")
    return 0 if len(manifest) == len(targets) else 1


if __name__ == "__main__":
    sys.exit(main())
