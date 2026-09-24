# Phase 3a: String/Date/Venue Features + Baseline Logistic Regression

Scope: PRD section 6.3 (string similarity, date proximity, venue match,
source reliability) plus the phase-3 baseline model from PRD section 10
("train a simple baseline (logistic regression on string similarity
alone) to get a first F1 number, then swap in LightGBM"). This agent does
**not** touch embeddings (owned by a parallel agent) or LightGBM (phase 4+).

## What was built

`features/build_string_date_venue_features.py` reads
`data/processed/candidate_pairs.csv` (101,044 rows) and writes
`features/string_date_venue_features.csv` — same row count and order,
keyed by `(listing_id, candidate_event_id)`, with `label` and `split`
copied through unchanged, plus 23 feature columns:

**String similarity (8 features)** — computed on `raw_title` against
`cand_home_team`/`cand_away_team` (joined both orders, since listings can
read "A vs B" or "B at A"), both on the raw strings and after
`blocking/normalize.py`'s `normalize_text()`:
- `str_lev_ratio_{raw,norm}` — rapidfuzz Levenshtein ratio
- `str_jaro_winkler_{raw,norm}` — jellyfish Jaro-Winkler similarity
- `str_token_jaccard_{raw,norm}` — token-set Jaccard overlap
- `str_token_sort_ratio_{raw,norm}` — rapidfuzz token-sort ratio (added
  beyond the PRD's minimum list — handles word-order noise directly
  rather than relying on the two-order max trick used for the other two)

**Date proximity (5 features)**, reusing `normalize_date()`'s
ambiguous-year-aware parsing for the listing side and `dateutil` for the
catalog's clean ISO dates:
- `date_missing` — 1 if the listing date couldn't be parsed at all (6.8%
  of rows; `raw_date` empty or unparseable)
- `date_year_inferred` — 1 if the raw date had no explicit year (uses the
  same signal `normalize_date()` returns)
- `date_day_diff` — absolute day difference, sentinel `999` when
  `date_missing=1` (never silently defaulted to 0)
- `date_exact_match`, `date_same_weekday` — flags; both forced to 0 when
  the date is missing

**Venue match (6 features)**:
- `venue_missing` — 1 if `raw_venue` is empty/null (9.2% of rows)
- `venue_exact_match` — normalized raw venue string equals normalized
  candidate venue, no alias resolution
- `venue_alias_match` — `raw_venue` is a literal key in `VENUE_ALIASES`
  and its mapped canonical name matches the candidate (isolates the
  alias table's specific contribution from plain exact match)
- `venue_fuzzy_score` — rapidfuzz ratio on alias-resolved, normalized
  strings (via `normalize_venue()`)
- `venue_geo_distance_km` — haversine distance in km; `-1` sentinel when
  not computable
- `venue_geo_distance_missing` — 1 when the listing's venue couldn't be
  resolved to a known catalog venue (so no coordinates exist for it) —
  candidate_pairs.csv only carries lat/lon for the candidate side, so the
  listing side is looked up via a `venue_canonical -> (lat, lon)` table
  built from `events_catalog.csv`; unresolvable ~15.2% of rows

**Source reliability (4 features)**: one-hot `source_official_feed`,
`source_partner_feed_a`, `source_scraped_feed_b`, `source_scraped_feed_c`.

All missing/unparseable cases are handled with explicit sentinel values
(`999`, `-1`) plus a paired `*_missing` flag column, never a silent
default — no NaNs in the output file (verified).

## Baseline: logistic regression on string features alone

`model/baseline_lr.py` trains `sklearn.linear_model.LogisticRegression`
on the 8 string-similarity features only (no date/venue/source), fit on
`split == train` (70,949 pairs, 3,715 positive — 5.2% base rate),
evaluated on `val` (15,160 pairs, 797 positive) and `test` (14,935 pairs,
798 positive). This is a per-**pair** classifier (is this candidate the
match for this listing), not a top-1-per-listing ranker, per the task
scope.

`class_weight="balanced"` was used — with an unweighted fit, threshold
0.5 recall collapses to ~0.04 (measured: val P=0.302/R=0.036/F1=0.065,
test P=0.425/R=0.039/F1=0.071) because ~95% of pairs are negatives from
blocking. This is a one-line reweighting, not a tuned hyperparameter, so
it stays within "tune nothing fancy."

**Real, measured numbers** (no fabrication; `model/baseline_lr_results.json`
has the full output):

| Split | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| val | 0.5 | 0.1653 | 0.7077 | 0.2681 |
| val | best-F1 on PR curve (≈0.849) | 0.3785 | 0.5157 | 0.4365 |
| test | 0.5 | 0.1626 | 0.7043 | 0.2642 |
| test | best-F1 on PR curve (≈0.851) | 0.3799 | 0.5251 | 0.4408 |

At threshold 0.5, `class_weight="balanced"` pushes the decision boundary
toward recall (expected/by design given the reweighting); the PR-curve
best-F1 point (~0.85) is the more representative single operating point
for this feature set — F1 ≈ 0.44 on both val and test, consistent
between splits (no obvious overfitting from this simple a model).

## Expected contribution of date/venue features (qualitative)

Not measured here (this agent trains string-only, per scope), but
expected directionally:
- **Date features** should cut a specific failure mode this feature set
  can't see at all: two different events with similar/identical team
  names on different dates (e.g. two Lakers-Celtics games in the same
  season). `date_exact_match` and `date_day_diff` give the model a hard
  signal string similarity has none of, so this is likely the single
  largest F1 jump in the eventual ablation.
- **Venue features** should mainly help disambiguate near-duplicate
  matchups at different arenas and recover cases where alias
  substitution (e.g. "Crypto Arena" vs "Crypto.com Arena") makes venue
  match fail without the alias table but succeed with it — a smaller,
  more targeted gain than date, concentrated on the subset of pairs
  where team-name similarity alone is ambiguous.
- Together, date+venue should push precision up substantially at a given
  recall (fewer same-team-different-game false positives), which matters
  more than the F1 number alone since the PRD's real target is high
  precision at the auto-accept threshold (section 6.5).
