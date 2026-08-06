r"""
drgct -- Deep-learning based Doubly Robust Granger Causality Test
=================================================================

A faithful, documented implementation of

    Hui, Y., Liu, C. and Song, X. (2025).
    *Deep learning based doubly robust test for Granger causality.*
    arXiv:2509.15798v2 [stat.ME].  https://arxiv.org/abs/2509.15798

Quick start
-----------
>>> import numpy as np
>>> from drgct import drgc_test
>>> rng = np.random.default_rng(0)
>>> n = 500
>>> x = rng.normal(0, 1, n)
>>> y = np.r_[0.0, np.sin(x[:-1])] + rng.normal(0, 0.5, n)   # X causes Y at lag 1
>>> res = drgc_test(x, y, lag=1, B=299, seed=1)
>>> res.pvalue < 0.05
True

What the test does
------------------
``H0``: ``X`` does not Granger-cause ``Y`` in mean, i.e.
``E[Y_t | X_{t-1},...,X_{t-p}, Y_{t-1},...,Y_{t-q}] = E[Y_t | Y_{t-1},...,Y_{t-q}]``.

The test statistic is a Kolmogorov-Smirnov functional of a *doubly robust*
empirical process built from two deep-learning estimators -- an MLP for the
conditional mean and a mixture density network for the conditional
characteristic function -- and its critical values come from a multiplier
bootstrap that reuses the fitted quantities, so the networks are trained once
per test.  Because the bias of the process depends on the *product* of the two
estimation errors, deep networks whose individual rates are slower than
``n^{-1/2}`` still deliver a test with correct asymptotic size and power
approaching one, at lag orders where kernel-smoothing tests have already
collapsed under the curse of dimensionality.

Public API
----------
``drgc_test``, ``drgc_lag_scan``, ``drgc_both_directions``, ``drgc_stability``,
``DRGCTResult``
    The test itself (:mod:`drgct.core`).
``nhkj_test``
    Smoothing-based nonparametric benchmark (:mod:`drgct.nhkj`).
``simulate_dgp``, ``monte_carlo``, ``summarize``, ``size_power_tables``
    Section 4 replication (:mod:`drgct.dgp`, :mod:`drgct.simulate`).
``load_index``, ``to_percentage_changes``, ``subsample``, ``describe``
    Bundled economic data (:mod:`drgct.datasets`).
``price_volume_study``, ``rolling_causality``, ``lag_scan_frame``
    Section 5 replication and extensions (:mod:`drgct.applications`).
``drgct.tables``, ``drgct.plots``
    Journal-ready tables (LaTeX/Markdown/CSV) and figures (PDF/PNG).
``MLPConfig``, ``MDNConfig``
    Network hyper-parameters (:mod:`drgct.nets`).
``check_stationarity``, ``set_seed``, ``build_lag_design``
    Utilities (:mod:`drgct.utils`).

Install
-------
``pip install drgct``  (or ``pip install "drgct[data]"`` for the data
downloader).  Package: https://pypi.org/project/drgct/

Full guide: https://github.com/merwanroudane/DRGCT/blob/main/docs/GUIDE.md
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Merwan Roudane"
__email__ = "merwanroudane920@gmail.com"
__license__ = "MIT"

PAPER = {
    "title": "Deep learning based doubly robust test for Granger causality",
    "authors": ("Yongchang Hui", "Chijin Liu", "Xiaojun Song"),
    "year": 2025,
    "arxiv": "2509.15798v2",
    "url": "https://arxiv.org/abs/2509.15798",
}

from .applications import (  # noqa: E402
    DIRECTIONS,
    lag_scan_frame,
    price_volume_study,
    rolling_causality,
)
from .core import (  # noqa: E402
    DRGCTResult,
    drgc_both_directions,
    drgc_lag_scan,
    drgc_stability,
    drgc_test,
)
from .datasets import (  # noqa: E402
    INDEX_KEYS,
    INDEX_LABELS,
    PAPER_PERIODS,
    available_datasets,
    describe,
    load_all,
    load_index,
    subsample,
    to_percentage_changes,
)
from .dgp import DGP_NAMES, POWER_DGPS, SIZE_DGPS, simulate_dgp  # noqa: E402
from .nets import MDNConfig, MLPConfig  # noqa: E402
from .nhkj import NHKJResult, nhkj_test  # noqa: E402
from .simulate import monte_carlo, size_power_tables, summarize  # noqa: E402
from .utils import build_lag_design, check_stationarity, set_seed  # noqa: E402
from . import plots, tables  # noqa: E402,F401

__all__ = [
    "__version__",
    "PAPER",
    # core
    "drgc_test",
    "drgc_lag_scan",
    "drgc_both_directions",
    "drgc_stability",
    "DRGCTResult",
    # benchmark
    "nhkj_test",
    "NHKJResult",
    # simulation
    "simulate_dgp",
    "monte_carlo",
    "summarize",
    "size_power_tables",
    "DGP_NAMES",
    "SIZE_DGPS",
    "POWER_DGPS",
    # data
    "load_index",
    "load_all",
    "available_datasets",
    "to_percentage_changes",
    "subsample",
    "describe",
    "INDEX_KEYS",
    "INDEX_LABELS",
    "PAPER_PERIODS",
    # applications
    "price_volume_study",
    "rolling_causality",
    "lag_scan_frame",
    "DIRECTIONS",
    # configuration and utilities
    "MLPConfig",
    "MDNConfig",
    "set_seed",
    "check_stationarity",
    "build_lag_design",
    # submodules
    "tables",
    "plots",
]


def cite(style: str = "text") -> str:
    """Return the citation for the underlying paper and for this software.

    Parameters
    ----------
    style : {'text', 'bibtex'}
    """
    if style == "bibtex":
        return (
            "@article{HuiLiuSong2025DRGCT,\n"
            "  title   = {Deep learning based doubly robust test for Granger causality},\n"
            "  author  = {Hui, Yongchang and Liu, Chijin and Song, Xiaojun},\n"
            "  journal = {arXiv preprint arXiv:2509.15798},\n"
            "  year    = {2025},\n"
            "  url     = {https://arxiv.org/abs/2509.15798}\n"
            "}\n\n"
            "@software{Roudane2026drgct,\n"
            "  title   = {drgct: Deep-learning based doubly robust Granger causality "
            "testing in Python},\n"
            "  author  = {Roudane, Merwan},\n"
            f"  version = {{{__version__}}},\n"
            "  year    = {2026},\n"
            "  url     = {https://github.com/merwanroudane/DRGCT}\n"
            "}"
        )
    return (
        "Hui, Y., Liu, C. and Song, X. (2025). Deep learning based doubly robust test "
        "for Granger causality. arXiv:2509.15798.\n"
        f"Roudane, M. (2026). drgct: Deep-learning based doubly robust Granger causality "
        f"testing in Python (version {__version__}). "
        "https://github.com/merwanroudane/DRGCT"
    )
