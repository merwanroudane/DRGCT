**Descriptive statistics of the transformed price and volume series**

| Statistic         |   SPX 500  $P_t$ |   SPX 500  $V_t$ |   CSI 300  $P_t$ |   CSI 300  $V_t$ |   NI 225  $P_t$ |   NI 225  $V_t$ |
|:------------------|-----------------:|-----------------:|-----------------:|-----------------:|----------------:|----------------:|
| Obs.              |         1256.000 |         1256.000 |         1210.000 |         1210.000 |        1219.000 |        1219.000 |
| Mean              |            0.062 |            0.160 |            0.001 |            0.920 |           0.056 |           0.229 |
| Median            |            0.095 |           -0.033 |           -0.032 |           -0.101 |           0.090 |          -0.069 |
| Std. dev.         |            1.340 |            1.865 |            1.158 |            4.897 |           1.353 |           2.249 |
| Min.              |          -11.984 |           -5.769 |           -8.433 |           -7.004 |         -12.396 |          -5.385 |
| Max.              |            9.383 |           12.705 |            7.337 |           33.939 |          10.226 |          15.710 |
| Skewness          |           -0.526 |            1.730 |           -0.116 |            1.836 |          -0.257 |           1.391 |
| Kurtosis          |           16.293 |           11.525 |            8.007 |            9.152 |          13.431 |           7.773 |
| Jarque-Bera       |         9304.744 |         4430.385 |         1266.828 |         2587.604 |        5539.850 |        1550.387 |
| JB p-value        |            0.000 |            0.000 |            0.000 |            0.000 |           0.000 |           0.000 |
| Ljung-Box(10)     |          247.569 |          152.737 |            5.312 |          135.051 |          12.399 |         115.812 |
| LB p-value        |            0.000 |            0.000 |            0.869 |            0.000 |           0.259 |           0.000 |
| Ljung-Box$^2$(10) |         1780.124 |           60.641 |           79.513 |            1.479 |         423.691 |          10.567 |
| LB$^2$ p-value    |            0.000 |            0.000 |            0.000 |            0.999 |           0.000 |           0.392 |
| ADF p-value       |            0.000 |            0.000 |            0.000 |            0.000 |           0.000 |           0.000 |
| KPSS p-value      |            0.100 |            0.100 |            0.100 |            0.100 |           0.100 |           0.100 |

_$P_t$ is the daily percentage change in the closing level; $V_t$ is the daily percentage change in trading volume divided by 10, following Section 5 of Hui, Liu and Song (2025).  Ljung--Box statistics use 10 lags.  ADF and KPSS report p-values for the unit-root and stationarity nulls._
