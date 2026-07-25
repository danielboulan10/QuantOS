# DDR-004: Critical values are simulated for our own estimator

- **Status:** Accepted
- **Affects:** `core/timeseries/cointegration.py`, `scripts/tabulate_johansen.py`

## Context

The Johansen trace statistic's asymptotic distribution is non-standard, so
critical values come from tables. Published tables are specific to the VECM's
*deterministic specification* — no constant, unrestricted constant, constant
restricted to the cointegrating space, and trend variants — and they are widely
reproduced **without that specification attached**.

## The incident

An Osterwald-Lenum table was used with our estimator. A size check on independent
random walks then showed a **28% rejection rate against a nominal 5%**: the test
claimed cointegration in roughly one in four unrelated pairs of random walks.

The statistic was correct. The table assumed a different treatment of the
constant, and its 95% value for `k−r=1` was 4.13 where our specification needs
about 8.2.

This is worth dwelling on because of how it would have failed in use. Nothing
crashed. Nothing looked wrong. A pairs-trading research programme built on that
estimator would have produced a large inventory of "cointegrated" pairs, every one
of them noise, and the error would have surfaced only as unexplained live losses.

## Decision

`scripts/tabulate_johansen.py` simulates the null distribution of the statistic
**as `_johansen_statistics` computes it**, and the resulting quantiles are pasted
into the module with the generating command recorded.

## Rationale

The asymptotic distribution depends only on `k − r` and the deterministic terms.
Simulating against our own estimator makes size correct *by construction*, and it
removes the entire class of error where a remembered constant does not match the
code it is applied to.

The simulated `k−r=1` value of 8.39 sits next to the 8.18 published for the
unrestricted-constant case, which both identifies our specification and confirms
the original table was the wrong one. The residual gap is finite-sample: ours are
T=1000 quantiles, which is the correct thing to compare a T=1000 statistic
against.

Empirical size after the change: **2.7%** against a nominal 5%, asserted by
`tests/core/test_statistics.py::test_johansen_size_is_controlled`.

## Consequences

- **Positive:** correct size, auditable provenance, and regenerable numbers.
- **Positive:** the same approach extends to any statistic whose null we can
  simulate but not derive.
- **Negative:** the table is finite-sample rather than asymptotic, so it is
  strictly correct only near T=1000. Documented at the table.
- **Negative:** 6,000 replications at six dimensions takes a few minutes, so the
  table is checked in rather than computed at import.

## The general rule this establishes

**Every statistical test in QuantOS has a size check.** Not an accuracy check
against a reference implementation — a measurement of its false-positive rate
under its own null. It is the only test that would have caught this, and it is
cheap. See the `@pytest.mark.statistical` tests.
