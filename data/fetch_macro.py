#!/usr/bin/env python
"""Rebuild the bundled US macroeconomic dataset used by the macro application.

Downloads monthly series from the Federal Reserve Bank of St. Louis (FRED) and
writes them as a single tidy CSV, ``us_macro.csv``, inside the installable
package.

=============  ======================================================  ==========
FRED id        Series                                                  Units
=============  ======================================================  ==========
``INDPRO``     Industrial Production: Total Index                      index
``CPIAUCSL``   Consumer Price Index for All Urban Consumers            index
``PCEPI``      Personal Consumption Expenditures Price Index           index
``FEDFUNDS``   Effective Federal Funds Rate                            percent
``M2SL``       M2 Money Stock                                          $ billions
``UNRATE``     Unemployment Rate                                       percent
``PAYEMS``     All Employees, Total Nonfarm                            thousands
``WTISPLC``    Spot Crude Oil Price: West Texas Intermediate           $ / barrel
=============  ======================================================  ==========

FRED data are in the public domain; the series are redistributed here so that
every example in the repository runs offline.  See
<https://fred.stlouisfed.org/> for the primary source and the individual
series pages for their release notes.

Usage
-----
    python data/fetch_macro.py
    python data/fetch_macro.py --start 1970-01-01 --end 2024-12-31
    python data/fetch_macro.py --add GDPC1 TB3MS

Requires ``pandas-datareader`` (``pip install "drgct[data]"``) and an internet
connection.  The repository already ships the resulting CSV.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
PACKAGE_DATA = HERE.parent / "src" / "drgct" / "data"

DEFAULT_START = "1959-01-01"
DEFAULT_END = "2025-12-31"

SERIES = {
    "INDPRO": "Industrial production index",
    "CPIAUCSL": "Consumer price index (all urban consumers)",
    "PCEPI": "PCE price index",
    "FEDFUNDS": "Effective federal funds rate",
    "M2SL": "M2 money stock",
    "UNRATE": "Unemployment rate",
    "PAYEMS": "Total nonfarm payrolls",
    "WTISPLC": "Spot crude oil price, WTI",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--add", nargs="*", default=[], help="extra FRED series ids")
    ap.add_argument("--outdir", default=str(PACKAGE_DATA if PACKAGE_DATA.exists() else HERE))
    a = ap.parse_args(argv)

    import pandas_datareader.data as web

    ids = list(SERIES) + list(a.add)
    print(f"[drgct] downloading {len(ids)} FRED series, {a.start} .. {a.end}")
    df = web.DataReader(ids, "fred", a.start, a.end)
    df.index.name = "Date"
    df = df.dropna(how="all")

    outdir = pathlib.Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "us_macro.csv"
    df.round(6).to_csv(path)
    print(f"[drgct] wrote {path}  ({len(df)} monthly observations, "
          f"{df.index.min().date()} .. {df.index.max().date()})")
    for col in df.columns:
        n = int(df[col].notna().sum())
        print(f"    {col:<10s} {n:>4d} obs   {SERIES.get(col, '')}")

    lines = [
        "# US macroeconomic data",
        "",
        f"Downloaded from FRED (Federal Reserve Bank of St. Louis) on "
        f"{_dt.date.today().isoformat()} by `data/fetch_macro.py`.",
        "",
        f"Monthly, {df.index.min().date()} to {df.index.max().date()}, "
        f"{len(df)} observations.",
        "",
        "| FRED id | Series | Observations | Source page |",
        "|---|---|---|---|",
    ]
    for col in df.columns:
        lines.append(
            f"| `{col}` | {SERIES.get(col, '')} | {int(df[col].notna().sum())} | "
            f"<https://fred.stlouisfed.org/series/{col}> |"
        )
    lines += [
        "",
        "FRED series are in the public domain unless the individual series page",
        "says otherwise; see <https://fred.stlouisfed.org/legal/>.",
    ]
    (outdir / "SOURCES_MACRO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[drgct] wrote {outdir / 'SOURCES_MACRO.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
