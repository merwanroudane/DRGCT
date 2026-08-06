"""Run the doctests embedded in the fast modules.

The doctests in ``drgct.core`` and ``drgct.nhkj`` train networks and take a
few seconds each; run them on demand with::

    pytest --doctest-modules src/drgct
"""

from __future__ import annotations

import doctest

import pytest

from drgct import datasets, dgp, utils


@pytest.mark.parametrize("module", [utils, dgp, datasets])
def test_module_doctests(module):
    result = doctest.testmod(module, verbose=False, raise_on_error=False)
    assert result.failed == 0, f"{result.failed} doctest failure(s) in {module.__name__}"
