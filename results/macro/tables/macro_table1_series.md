**US macroeconomic series, transformations and stationarity screens**

| FRED id   | Series                | Transformation   |   Obs. |   ADF p |   KPSS p | Stationary   |
|:----------|:----------------------|:-----------------|-------:|--------:|---------:|:-------------|
| INDPRO    | Industrial production | 100 x dlog       |    801 |   0.000 |    0.019 | no           |
| CPIAUCSL  | CPI inflation         | 100 x dlog       |    801 |   0.026 |    0.010 | no           |
| PCEPI     | PCE inflation         | 100 x dlog       |    801 |   0.040 |    0.010 | no           |
| FEDFUNDS  | Fed funds rate        | first difference |    801 |   0.000 |    0.100 | yes          |
| M2SL      | M2 money growth       | 100 x dlog       |    801 |   0.000 |    0.078 | yes          |
| UNRATE    | Unemployment rate     | first difference |    801 |   0.000 |    0.100 | yes          |
| PAYEMS    | Nonfarm payrolls      | 100 x dlog       |    801 |   0.000 |    0.010 | no           |
| WTISPLC   | WTI oil price         | 100 x dlog       |    801 |   0.000 |    0.100 | yes          |

_Monthly data from FRED (Federal Reserve Bank of St. Louis), February 1959 to December 2025.  ADF reports the p-value of the augmented Dickey--Fuller unit-root null; KPSS the p-value of the stationarity null.  A series is marked stationary only when ADF rejects and KPSS does not.  Assumption 1 of Hui, Liu and Song (2025) requires stationarity._
