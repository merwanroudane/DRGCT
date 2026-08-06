# Replication scripts

Two scripts reproduce everything in [`../results/`](../results).

## `run_simulation.py` — Section 4

Two experiments:

* **A.** All six designs of Table 1, lag orders 1–5, comparing the doubly
  robust test, the naive deep plug-in and the smoothing benchmark. Produces
  Tables 1–4 and figures 1–3.
* **B.** Design S1 at larger sample sizes, isolating the type-I-error blow-up
  of the naive plug-in that Section 4 documents. Produces Table 3b and
  figure 4.

```bash
python scripts/run_simulation.py                       # the shipped configuration
python scripts/run_simulation.py --reps 1000 --ns 500 1000 2000 -B 1000 --jobs 10
python scripts/run_simulation.py --quick               # 2-minute plumbing check
python scripts/run_simulation.py --skip-b              # experiment A only
```

Key options: `--reps --ns --lags --dgps --methods -B --alpha --jobs --seed
--outdir --b-ns --b-lags --b-reps`.

## `run_application.py` — Section 5

Loads the three bundled indices, screens for stationarity, writes descriptive
statistics, runs 3 indices × 3 sub-samples × 2 directions × 10 lags = **180
DRGCTs**, writes Tables 5–6 plus a p-value companion, draws all the figures,
runs a stability check on the headline specification, and optionally a
rolling-window study.

```bash
python scripts/run_application.py --jobs 10 --rolling --stability-draws 30
python scripts/run_application.py --quick
python scripts/run_application.py --indices spx500 --lags 1 2 3 4 5
```

Key options: `--indices --periods --lags -B -G -L -M --alpha --jobs --seed
--outdir --stability-draws --rolling --roll-index --roll-window --roll-step
--roll-lag`.

## `run_macro_application.py` — US macroeconomics

Eight monthly FRED series, 1959–2025. Six relations, both directions, lag
orders 1–18 months, a Great Inflation versus Great Moderation split, a
three-way comparison against a linear VAR *F*-test and the smoothing benchmark,
and a direction-draw stability check. Writes to `results/macro/`.

```bash
python scripts/run_macro_application.py --jobs -1
python scripts/run_macro_application.py --quick
python scripts/run_macro_application.py --lags 1 3 6 12 --skip-subsamples
```

Key options: `--lags --sub-lags -B -G -L -M --alpha --jobs --seed
--stability-draws --skip-subsamples --outdir`.

`-L` defaults to 60 rather than the paper's 20, because at lag 18 the
conditioning set has 36 dimensions and twenty random directions cover it far
too thinly.

## `build_site.py` — the documentation website

Converts `site/pages/*.md` and `docs/*.md` into the static site served by
GitHub Pages from `/docs`, copying every result figure into `docs/assets/`.
Tables and figures are embedded from the committed run output with
`{{table: ...}}` and `{{figure: ...}}` directives, so nothing on the site is
retyped by hand.

```bash
python scripts/build_site.py
python scripts/build_site.py --serve      # preview on http://localhost:8000
```

## Notes

* Both scripts guard their entry point with `if __name__ == "__main__":`,
  which is required on Windows and macOS for the process pool.
* `--jobs -1` uses `cpu_count() - 1`. Workers set `torch.set_num_threads(1)`:
  these networks are far too small for intra-op threading to pay off.
* Long Monte-Carlo runs stream partial results to CSV, so an interrupted run
  is not a total loss.
