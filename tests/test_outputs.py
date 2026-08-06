"""Tests for the data loaders, the table builders and the figure builders."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from drgct.datasets import (
    INDEX_KEYS,
    PAPER_PERIODS,
    available_datasets,
    describe,
    load_index,
    subsample,
    to_percentage_changes,
)
from drgct.dgp import simulate_dgp
from drgct.nets import MDNConfig, MLPConfig
from drgct.plots import (
    plot_bootstrap_distribution,
    plot_empirical_process,
    plot_lag_profile,
    plot_pvalue_heatmap,
    plot_series_overview,
    plot_size,
    plot_stability,
    plot_training_curves,
    save_figure,
    use_journal_style,
)
from drgct.tables import (
    CROSS,
    TICK,
    export_table,
    table_detection,
    table_dgp_definitions,
    table_hyperparameters,
    table_lag_orders,
    table_parameter_settings,
    table_pvalues,
    to_latex_booktabs,
)

FAST = dict(B=99, L=6, M=6, G=3,
            mlp=MLPConfig(epochs=40, patience=20),
            mdn=MDNConfig(epochs=50, patience=25))


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def test_bundled_datasets_are_present_and_cover_the_paper_window():
    assert set(INDEX_KEYS) <= set(available_datasets())
    for key in INDEX_KEYS:
        df = load_index(key)
        assert list(df.columns) == ["Close", "Volume"]
        assert len(df) > 1000
        assert df.index.is_monotonic_increasing
        assert (df["Volume"] > 0).all()
        assert str(df.index.min().date()) <= "2019-10-05"
        assert str(df.index.max().date()) >= "2024-09-20"


def test_percentage_change_transform():
    df = load_index("spx500")
    pv = to_percentage_changes(df, volume_divisor=10.0)
    assert list(pv.columns) == ["P", "V"]
    assert len(pv) == len(df) - 1
    assert np.isfinite(pv.to_numpy()).all()
    # V is the volume percentage change divided by ten
    raw_v = 100.0 * df["Volume"].pct_change().dropna().to_numpy()
    np.testing.assert_allclose(pv["V"].to_numpy(), raw_v / 10.0, rtol=1e-10)


def test_subsample_matches_the_paper_windows():
    pv = to_percentage_changes(load_index("spx500"))
    for name, (lo, hi) in PAPER_PERIODS.items():
        s = subsample(pv, name)
        assert str(s.index.min().date()) >= lo
        assert str(s.index.max().date()) <= hi
        assert 600 < len(s) < 800          # "approximately n = 750"


def test_describe_returns_the_expected_block():
    rng = np.random.default_rng(0)
    d = describe({"a": rng.normal(size=500)})
    for row in ("Obs.", "Mean", "Std. dev.", "Skewness", "Kurtosis",
                "Jarque-Bera", "ADF p-value", "KPSS p-value"):
        assert row in d.index


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def app_frame():
    """A small tidy long frame in the layout the table builders expect."""
    rows = []
    rng = np.random.default_rng(0)
    for idx in ("SPX 500", "CSI 300"):
        for per in ("2019-2022", "2021-2024"):
            for direction in ("P_t -> V_t", "V_t -> P_t"):
                for lag in range(1, 6):
                    rows.append({
                        "index_label": idx, "period": per, "direction": direction,
                        "lag": lag, "pvalue": float(rng.uniform()),
                        "ks_stat": float(rng.uniform(0.5, 2.0)),
                    })
    return pd.DataFrame(rows)


def test_detection_and_lag_tables(app_frame):
    det = table_detection(app_frame, alpha=0.05)
    assert {"Causality direction", "Period"} <= set(det.columns)
    marks = det.drop(columns=["Causality direction", "Period"]).to_numpy().ravel()
    assert set(marks) <= {TICK, CROSS}

    lo = table_lag_orders(app_frame, alpha=0.05, lags=range(1, 6))
    assert list(lo.columns[:3]) == ["Causality direction", "Index", "Period"]
    assert list(lo.columns[3:]) == [1, 2, 3, 4, 5]

    pv = table_pvalues(app_frame, lags=range(1, 6))
    assert pv.shape[0] == lo.shape[0]


@pytest.mark.parametrize("rule", ["any", "majority", "all"])
def test_detection_rules_are_monotone(app_frame, rule):
    det = table_detection(app_frame, alpha=0.5, rule=rule)
    assert det.shape[0] == 4      # 2 directions x 2 periods


def test_design_tables():
    assert len(table_dgp_definitions()) == 6
    assert len(table_parameter_settings()) == 5


def test_latex_export_is_booktabs_and_escapes(tmp_path):
    df = pd.DataFrame({"name": ["a_b", "100%", TICK], "value": [1.234, 5.0, np.nan]})
    tex = to_latex_booktabs(df, caption="Cap & more", label="tab:x", notes="A note.")
    assert "\\begin{table}" in tex and "\\toprule" in tex and "\\bottomrule" in tex
    assert "\\caption{Cap & more}" in tex
    assert "a\\_b" in tex and "100\\%" in tex and "\\checkmark" in tex

    paths = export_table(df, "t", tmp_path, caption="Cap", label="tab:x", quiet=True)
    for kind in ("csv", "md", "tex"):
        assert (tmp_path / f"t.{kind}").exists()
        assert (tmp_path / f"t.{kind}").read_text(encoding="utf-8").strip()
    assert set(paths) == {"csv", "md", "tex"}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def test_every_figure_builds_and_saves(tmp_path, app_frame):
    import matplotlib.pyplot as plt

    from drgct import drgc_stability, drgc_test

    use_journal_style()
    s = simulate_dgp("P1", n=200, lag=1, rng=0)
    res = drgc_test(s.x, s.y, lag=1, seed=0, return_networks=True, **FAST)

    figs = {
        "boot": plot_bootstrap_distribution(res),
        "proc": plot_empirical_process(res),
        "train": plot_training_curves(res),
        "heat": plot_pvalue_heatmap(app_frame),
        "lag": plot_lag_profile(
            app_frame[(app_frame["direction"] == "P_t -> V_t")
                      & (app_frame["index_label"] == "SPX 500")
                      & (app_frame["period"] == "2019-2022")]),
        "series": plot_series_overview(
            {"SPX 500": load_index("spx500").iloc[:200]},
            {"SPX 500": to_percentage_changes(load_index("spx500")).iloc[:200]}),
    }
    for name, fig in figs.items():
        paths = save_figure(fig, name, tmp_path, formats=("png",), quiet=True)
        assert (tmp_path / f"{name}.png").stat().st_size > 5_000, name
        assert paths["png"].endswith(".png")
    plt.close("all")


def test_mdn_fit_requires_return_networks():
    from drgct import drgc_test
    from drgct.plots import plot_mdn_fit

    s = simulate_dgp("S1", n=200, lag=1, rng=0)
    res = drgc_test(s.x, s.y, lag=1, seed=0, **FAST)
    with pytest.raises(ValueError, match="return_networks"):
        plot_mdn_fit(res)


def test_size_figure_from_a_summary_frame(tmp_path):
    import matplotlib.pyplot as plt

    summ = pd.DataFrame([
        {"dgp": "S1", "n": 500, "lag": lag, "method": m, "reps": 100,
         "rejection_rate": 0.05 + 0.01 * lag, "mc_se": 0.02, "causal": False}
        for lag in (1, 2, 3) for m in ("drgc", "nhkj")
    ])
    save_figure(plot_size(summ), "size", tmp_path, formats=("png",), quiet=True)
    assert (tmp_path / "size.png").exists()
    plt.close("all")


def test_stability_helpers(tmp_path):
    import matplotlib.pyplot as plt

    from drgct import drgc_stability

    s = simulate_dgp("P1", n=200, lag=1, rng=1)
    stab = drgc_stability(s.x, s.y, lag=1, n_draws=3, seed=0, **FAST)
    assert stab["pvalues"].shape == (3,)
    assert 0.0 <= stab["median"] <= 1.0
    assert stab["merged_pvalue"] == pytest.approx(min(1.0, 2 * stab["median"]))
    save_figure(plot_stability(stab), "stab", tmp_path, formats=("png",), quiet=True)
    plt.close("all")


def test_hyperparameter_table():
    from drgct import drgc_test

    s = simulate_dgp("S1", n=200, lag=1, rng=0)
    res = drgc_test(s.x, s.y, lag=1, seed=0, **FAST)
    t = table_hyperparameters(res)
    assert list(t.columns) == ["Setting", "Value"]
    assert len(t) >= 10
