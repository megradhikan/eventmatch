# Phase 5 — Calibration + Confidence Thresholds

## Calibration

Isotonic regression ([`model/calibrate.py`](../model/calibrate.py)) fit on
`val` raw LightGBM scores -> P(match), applied to `test`. Chosen over Platt
scaling since LightGBM's raw pairwise-margin scores aren't assumed to be
sigmoid-shaped, and isotonic regression makes no parametric assumption about
the score->probability mapping.

## Threshold tuning (on `val`, applied to held-out `test`)

Two thresholds, tuned only on `val`:

- **`match_threshold` = 0.96** — the lowest calibrated score that still hits
  **99% precision on val** (target precision reachable; val precision at
  this threshold = 0.9915). PRD 6.5: false positives (wrongly merging two
  different events) are worse than false negatives here.
- **`review_floor` = 0.071** — the calibrated score above which 98% of
  val's recoverable true positives are captured; below it, a listing is
  unlikely enough to be a real match that agentic review isn't worth the
  cost, so it goes straight to `NO_MATCH`.

## Tier decision is per LISTING, not per candidate pair

An earlier version of this script assigned tiers per (listing, candidate)
row — wrong, since a listing has ~19 candidate rows but exactly one real
prediction. Fixed to: for each listing, take its highest-calibrated-score
candidate among blocking candidates, and tier THAT.

## Result on held-out test (919 listings)

| Tier | Volume | % of traffic |
|---|---|---|
| auto (>= 0.96) | 690 | 75.1% |
| agentic (0.071 - 0.96) | 124 | 13.5% |
| no_match (< 0.071) | 105 | 11.4% |

**Auto-accept precision, actually measured on test: 99.71%** (688/690
correct) — above the 99% target, and consistent with val's 99.15%.
Auto-accept recall of recoverable positives (listings whose true match
survived blocking): 86.2% (690 of 798).

The agentic band is 13.5% of traffic — comfortably under the PRD's stated
red flag ("if it's running on 40% of listings, the ranker/thresholds need
work"), meaning the ranker is doing most of the discriminating work and
the agentic tier is handling a genuinely ambiguous minority, which is the
intended design.

## What "no_match" actually contains

105 test listings fall below `review_floor`. This isn't only genuine
`NO_MATCH` cases (held-out/fabricated events) — some are real matches
where blocking, string noise, or date corruption pushed every candidate's
score down (a listing whose true match never survived blocking at all,
Phase 2, is unrecoverable by construction here). These are the
unrecoverable-by-this-pipeline cases; Phase 7's failure analysis pulls
concrete examples and gives exact counts.
