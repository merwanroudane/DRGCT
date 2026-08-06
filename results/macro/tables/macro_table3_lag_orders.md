**Granger causality in mean among US macroeconomic series, by lag order**

| Causality direction                        | Index                  | Period      | 1   | 3   | 6   | 9   | 12   | 18   |
|:-------------------------------------------|:-----------------------|:------------|:----|:----|:----|:----|:-----|:-----|
| CPI inflation -> Fed funds rate            | MP to prices           | Full sample | ✗   | ✗   | ✗   | ✗   | ✗    | ✗    |
| CPI inflation -> M2 money growth           | Money to prices        | Full sample | ✗   | ✓   | ✗   | ✗   | ✗    | ✗    |
| CPI inflation -> WTI oil price             | Oil to prices          | Full sample | ✗   | ✗   | ✓   | ✗   | ✗    | ✗    |
| Fed funds rate -> CPI inflation            | MP to prices           | Full sample | ✗   | ✗   | ✗   | ✗   | ✗    | ✗    |
| Fed funds rate -> Industrial production    | MP to output           | Full sample | ✓   | ✗   | ✗   | ✗   | ✗    | ✗    |
| Industrial production -> Fed funds rate    | MP to output           | Full sample | ✗   | ✗   | ✗   | ✗   | ✗    | ✗    |
| Industrial production -> Unemployment rate | Output to unemployment | Full sample | ✓   | ✓   | ✓   | ✗   | ✗    | ✗    |
| Industrial production -> WTI oil price     | Oil to output          | Full sample | ✗   | ✗   | ✗   | ✗   | ✗    | ✗    |
| M2 money growth -> CPI inflation           | Money to prices        | Full sample | ✓   | ✗   | ✓   | ✗   | ✗    | ✗    |
| Unemployment rate -> Industrial production | Output to unemployment | Full sample | ✓   | ✗   | ✗   | ✗   | ✓    | ✗    |
| WTI oil price -> CPI inflation             | Oil to prices          | Full sample | ✓   | ✓   | ✓   | ✗   | ✗    | ✓    |
| WTI oil price -> Industrial production     | Oil to output          | Full sample | ✓   | ✗   | ✗   | ✗   | ✗    | ✗    |

_Ticks mark rejection of the null of non-causality at the 5\% level, bootstrap critical values with $B=999$, $G=10$, $L=60$, $M=20$.  Lag orders are months._
