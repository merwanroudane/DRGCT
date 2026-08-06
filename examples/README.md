# Examples

Four runnable scripts, in increasing order of scope. Each is self-contained
and writes its output to its own `results_example*/` folder.

| Script | Runtime | What it does |
|---|---|---|
| [`01_quickstart.py`](01_quickstart.py) | ~1 min | A series where the truth is known: purely nonlinear causality that a linear VAR *F*-test misses entirely. Shows every field of the result object. |
| [`02_simulation_size_power.py`](02_simulation_size_power.py) | ~6 min on 8 cores | A miniature Section 4: size and power for DGP S1 and P2, comparing the doubly robust test, the naive deep plug-in, and the smoothing-based benchmark. Produces Tables 3–4 and the size/power/p-value-plot figures. |
| [`03_real_data_price_volume.py`](03_real_data_price_volume.py) | ~4 min on 8 cores | The price–volume application on the bundled CSI 300 data: stationarity screen, descriptive table, a two-directional lag scan over 1–10, benchmarks, a stability check, and all the figures. |
| [`04_your_own_data.py`](04_your_own_data.py) | ~5 min on 8 cores | **The template to copy.** Edit one CONFIG block and point it at your CSV. Ships with a synthetic macro dataset (an asymmetric policy shock transmitting to output at lag 6) so it runs out of the box. |

```bash
python examples/01_quickstart.py
python examples/02_simulation_size_power.py
python examples/03_real_data_price_volume.py
python examples/04_your_own_data.py
```

For the full replication of the paper, use the scripts in
[`../scripts/`](../scripts) instead. For the narrative walk-through, read
[`../docs/GUIDE.md`](../docs/GUIDE.md).
