# Phase 1 — Data

## Canonical catalog

Real schedule data pulled from ESPN's public (unauthenticated) scoreboard API:
the full completed **2025-26 NBA season** (1,401 games, incl. a handful of
international preseason/exhibition games) and the completed **2025 NFL
regular season** (272 games). Combined: **1,673 real events** across 83
distinct venues. See [`fetch_catalog.py`](../data/fetch_catalog.py).

Venue coordinates are a hand-curated static lookup
([`venue_coords.py`](../data/venue_coords.py)) rather than live geocoding —
Nominatim's public 1 req/sec rate limit made a one-time ~80-venue batch
lookup take upwards of 10+ minutes with no visible progress, so a static
table (accurate to city level, sufficient for a geo-distance feature) was
used instead. All 83 venues resolved with no missing coordinates.

## Noise generation ([`noise_generator.py`](../data/noise_generator.py))

Per PRD 5.2, each real event is turned into several noisy "listing" variants
simulating messy multi-feed scraped data, drawn from 4 simulated sources with
different noise profiles (`official_feed` clean, `partner_feed_a` moderate,
`scraped_feed_b`/`scraped_feed_c` noisy):

- **Name noise**: team order (`A vs B` / `B at A`), full-name vs. abbreviation
  (e.g. `LAL vs GSW`), character-level typos (keyboard-neighbor substitution,
  deletion, adjacent swap), casing, `vs`/`v.`/`@` punctuation drift.
- **Date noise**: format drift (ISO / `M/D/YYYY` / `M/D/YY` / `Month D, YYYY` /
  `Month D` with no year), and an 8% rare "wrong day" shift (±1-2 days) to
  test tolerance of near-misses.
- **Venue noise**: alias substitution from a hand-built table of real
  sponsor-renamed arenas/stadiums (`venue_aliases.py`, e.g. Crypto.com Arena
  / Staples Center), initialism abbreviation, and ~10% venue omission.
- **Structural noise**: extra whitespace, HTML entities (`&nbsp;`, `&#39;`,
  `&amp;`), and a per-source missing-field rate (venue or date dropped
  entirely).

## Hard negatives (ground truth: `NO_MATCH`)

Without these, an eval only measures "can it match an obviously present
event," which is close to meaningless. Two kinds are generated:

1. **Held-out real events** — 20% of the real catalog (335 events) is
   deliberately excluded from the indexed catalog that blocking/matching
   runs against. Noisy listings are still generated for them. These are
   real games, with real teams/venues/dates, that simply aren't in the
   catalog slice being matched against — the most realistic kind of "new
   event we haven't seen yet."
2. **Fabricated events** — 200 synthetic events built by pairing real team
   names into combinations that never actually played each other, on
   fabricated future dates. Tests generalization beyond "real event we
   didn't index."

Because the underlying catalog is real sports data, additional hard
negatives emerge structurally without extra generation: teams play many
games across a season (so "right teams, wrong date" candidates are common),
and many games share a date (so blocking's ±3-day window surfaces same-day
distractors). These become hard negative *candidate pairs* during Phase 2
blocking, not separate NO_MATCH listings.

## Train/val/test split — by event, not listing

Indexed catalog events get a 70/15/15 split (936 / 201 / 201 events),
assigned once and inherited by every listing generated from that event, so
listings of the same event never leak across splits (PRD 6.4 requirement).

## Output

| File | Rows | Purpose |
|---|---|---|
| `data/raw/events_catalog.csv` | 1,673 | full real catalog before the indexed/held-out split |
| `data/processed/events_catalog.csv` | 1,338 | indexed catalog — what Elasticsearch indexes in Phase 2 |
| `data/processed/held_out_events.csv` | 535 | held-out real (335) + fabricated (200) events — never indexed |
| `data/processed/listings.csv` | 6,422 | noisy listings, never contains the true label |
| `data/processed/ground_truth.csv` | 6,422 | `listing_id -> true_event_id` (blank = `NO_MATCH`); eval-only, never a model feature |

Ground truth balance: **5,352 positive** listings (4 per indexed event x
1,338 events) and **1,070 `NO_MATCH`** listings (2 per held-out/fabricated
event x 535 events).

Integrity checks run after generation: every positive label resolves to an
event_id present in the indexed catalog (0 dangling references), and there
is zero overlap between held-out event IDs and indexed catalog event IDs.
