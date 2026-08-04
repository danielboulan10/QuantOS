# Research notes

Short papers written from results this repository produced. Every number in them
is re-derived by [`scripts/verify_claims.py`](../../scripts/verify_claims.py) on
every push, so a note cannot drift away from the code it describes without CI
failing.

The topics were not chosen and then investigated. Each one is something that came
out of building a module — usually something that came out *wrong* first, and was
worth writing down once it was understood.

| | Note | Result |
|---|---|---|
| 001 | [Nothing survives an 840-factor search](001-nothing-survives.md) | The best of 840 factors on SPY has t = 2.23 and survives no correction. Two defects in the search itself were more interesting than the search. |
| 002 | [The stock–bond hedge is not a constant](002-the-hedge-inverted.md) | It held in 2008 and 2020 and inverted in 2022. The statistic that would have shown this in advance is not the one usually reported. |
| 003 | [A t-statistic of 8 with the sign wrong](003-confidently-wrong.md) | A macro beta significant at every conventional level, whose direction reversed in the regime that followed. What a confidence interval does and does not measure. |

## What these are and are not

They are working notes: method, data, result, and what the result does not
support. They are not peer reviewed, they use a public price feed rather than a
research-grade database, and the samples are ten to twenty years of daily bars —
enough to say something, not enough to settle anything.

Where a note reports a negative result, that is the finding, not a failed
attempt at a different one. Three of the three below are negative.

## Reproducing them

```bash
pip install -e ".[test]"
python scripts/verify_claims.py     # re-derives every number in every note
```

Each note names the command that produces its own tables.
