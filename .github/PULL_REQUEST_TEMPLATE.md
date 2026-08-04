## What this changes

<!-- One paragraph. What was wrong or missing, and what is different now. -->

## How it was validated

<!--
The bar here is not "the tests pass". It is that the new behaviour is checked
against something independent: a published benchmark, a closed form, an
analytic identity, SciPy as a test-only oracle, or a case constructed so the
answer is known in advance.

If a number is being added to the documentation, it needs a check in
scripts/verify_claims.py that fails when the number stops being true.
-->

## Checklist

- [ ] `make check` passes (lint, types, tests, claims, links)
- [ ] New behaviour is validated against something independent, and the PR says what
- [ ] Any number added to a README, docstring or research note is re-derived in `scripts/verify_claims.py`
- [ ] No new runtime dependency (DDR-002 — SciPy is test-only)
- [ ] Docstrings explain *why*, not only what
- [ ] If a defect was found and fixed, the fix is documented at its site rather than silently corrected

## Anything that came out badly

<!--
Optional and genuinely welcome. This repository publishes seven measurements
that were kept because they were unflattering. A negative result, a method that
did not work, or a limitation you hit is worth recording rather than dropping.
-->
