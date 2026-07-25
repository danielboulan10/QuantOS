"""Shared fixtures.

Every fixture that involves randomness is seeded. There is no fixture that
returns an unseeded generator, because a flaky statistical test is worse than no
test: it trains the reader to ignore failures.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantos.core.rng import SeedBank

#: Tolerance for comparisons against SciPy. Justified by the measured worst-case
#: relative errors tabulated in quantos.core.special's module docstring.
ORACLE_RTOL = 1e-10


@pytest.fixture
def bank() -> SeedBank:
    """A fixed-root seed bank. Derive per-test streams with ``.child(name)``."""
    return SeedBank(root=20240719)


@pytest.fixture
def rng(bank: SeedBank) -> np.random.Generator:
    return bank.child("test").generator()


@pytest.fixture
def gaussian_returns(rng: np.random.Generator) -> np.ndarray:
    """2,000 i.i.d. Gaussian returns with a small positive drift."""
    return rng.standard_normal(2_000) * 0.01 + 0.0003


@pytest.fixture
def garch_returns(rng: np.random.Generator) -> np.ndarray:
    """GARCH(1,1) returns with known parameters (omega=2e-6, a=0.08, b=0.90).

    Used wherever a test needs volatility clustering *and* a known answer.
    """
    n = 6_000
    r = np.zeros(n)
    v = np.zeros(n)
    v[0] = 2e-6 / (1 - 0.98)
    shocks = rng.standard_normal(n)
    for t in range(1, n):
        v[t] = 2e-6 + 0.08 * r[t - 1] ** 2 + 0.90 * v[t - 1]
        r[t] = np.sqrt(v[t]) * shocks[t]
    return r


@pytest.fixture
def cointegrated_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two I(1) series sharing a stochastic trend, with hedge ratio 2/3."""
    common = np.cumsum(rng.standard_normal(1_500))
    a = common + rng.standard_normal(1_500) * 0.4
    b = 1.5 * common + rng.standard_normal(1_500) * 0.4
    return a, b


@pytest.fixture
def independent_walks(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two independent random walks -- the spurious-regression null."""
    return (
        np.cumsum(rng.standard_normal(1_500)),
        np.cumsum(rng.standard_normal(1_500)),
    )
