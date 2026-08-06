# Data

The CSV files themselves live inside the installable package, at
[`src/drgct/data/`](../src/drgct/data), so that `drgct.datasets.load_index`
finds them after a plain `pip install`. This folder holds the downloader that
builds them.

## What ships

Daily closing levels and trading volumes for three indices over the exact
window used in Section 5 of Hui, Liu and Song (2025) —
**27 September 2019 to 26 September 2024**:

| File | Index | Ticker used | Rows |
|---|---|---|---|
| `spx500.csv` | S&P 500 | `^GSPC` | 1257 |
| `csi300.csv` | CSI 300 (exchange-traded tracker) | `510300.SS` | 1211 |
| `nikkei225.csv` | Nikkei 225 | `^N225` | 1220 |

Columns: `Date`, `Close`, `Volume`. The S&P 500 file has exactly the paper's
`T = 1257` observations.

The authoritative provenance record, regenerated on every download, is
[`src/drgct/data/SOURCES.md`](../src/drgct/data/SOURCES.md).

## Why the CSI 300 uses a tracker

Yahoo! Finance truncates the history of the CSI 300 *index* (`000300.SS`) to
roughly the last three years, so it cannot cover 2019–2024. `510300.SS` is the
Huatai-PineBridge CSI 300 ETF, the largest exchange-traded tracker of the
index: it spans the whole window and — decisive for a price-*volume* study —
carries genuine exchange turnover. `fetch_data.py` tries the index first,
checks that the returned series actually starts at the requested date, and
falls back only when it does not, recording which source it used.

If you have a CSI 300 index feed with volume (WIND, CSMAR, Bloomberg,
Refinitiv), export it as `Date, Close, Volume` and drop it in as
`src/drgct/data/csi300.csv`. Every script and example in the repository will
then use it without any code change.

## Rebuilding or extending

```bash
python data/fetch_data.py                                  # the paper's window
python data/fetch_data.py --start 2005-01-01 --end 2024-12-31
python data/fetch_data.py --tickers ^FTSE=ftse100 ^GDAXI=dax ^HSI=hangseng
python data/fetch_data.py --outdir /somewhere/else
```

Requires `pip install "drgct[data]"` (which adds `yfinance`) and an internet
connection. The repository already ships the CSVs, so this is only needed to
refresh or extend them.

## Using your own data

You do not have to touch this folder at all:

```python
from drgct.datasets import load_index, to_percentage_changes

df = load_index("anything", path="/path/to/my_series.csv")   # needs Date, Close, Volume
pv = to_percentage_changes(df, volume_divisor=10.0)
```

or skip the loaders entirely and hand two aligned arrays to `drgc_test`.

## Terms

Yahoo! Finance data is provided for personal, non-commercial use; see
<https://policies.yahoo.com/us/en/yahoo/terms/product-atos/apiforydn/index.htm>.
The CSVs are redistributed here to make the examples reproducible. Check the
terms before using them for anything else.
