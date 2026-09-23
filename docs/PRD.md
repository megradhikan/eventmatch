# PRD: EventMatch — Entity Resolution for Live Event Listings

## 1. Summary

A machine learning pipeline that maps messy, inconsistently formatted event listings (from multiple simulated feeds) to canonical event records — the "event matching" problem named directly in SeatGeek's JD, alongside venue mapping and inventory tracking. Demonstrates a full production-shaped ML pipeline: candidate blocking, feature engineering, a learned ranker, calibrated confidence thresholds, and an agentic fallback for ambiguous cases.

Target audience: new-grad ML / applied AI engineering interviews (SeatGeek and similar marketplace/data companies).

## 2. Goals

- Given a noisy listing, correctly resolve it to its canonical event, or correctly abstain (flag as new/uncertain) when it shouldn't be forced.
- Show the recall/speed tradeoff of candidate blocking explicitly, not just the final classifier.
- Show, via ablation, how much each feature group (string similarity, date, venue, embeddings) contributes.
- Handle the ambiguous long tail with an agentic fallback rather than silently getting them wrong.
- Report real precision/recall/F1 and latency numbers, not vibes.

## 3. Non-goals

- Real production data feeds or a live ingestion pipeline — synthetic/derived data is fine and expected, documented as such.
- Multi-language listings.
- Real-time streaming ingestion. Batch processing is sufficient.
- A UI. This is a backend/ML project; a simple CLI or notebook-driven demo is enough, plus an optional small API.

## 4. Problem framing

**Input**: a "listing" — a noisy record with fields like `title` (e.g. "Lakers vs Celtics", "LAL v BOS 3/14", "Boston Celtics at LA Lakers"), `date` (multiple formats, occasional missing/wrong year), `venue` (aliased names, e.g. "Crypto.com Arena" vs "Staples Center"), `source` (which feed it came from).

**Output**: a match to one canonical `event_id` in the event catalog, with a confidence score, or `NO_MATCH` (new event / needs review) when confidence is below threshold.

This is entity resolution / record linkage, framed as a learning-to-rank problem over candidate pairs, not a naive classifier over the full catalog.

## 5. Data

### 5.1 Canonical event catalog
- Pull a real schedule (NBA, NFL, or a mix — e.g. via a public sports schedule API, or a static historical dataset) to get real team names, venues, and dates. A few thousand events is enough.
- Normalize into: `event_id, home_team, away_team (or artist/show name), venue_canonical, date, venue_lat, venue_lon`.

### 5.2 Synthetic noisy listings (ground truth is known because you generate the noise)
Write a noise-generation script that takes each canonical event and produces N noisy variants:
- **Name noise**: reordering ("A vs B" / "B at A"), abbreviation (full name → city code / initials), typos (character-level edit noise), casing, punctuation drift.
- **Date noise**: format drift (`2027-03-14`, `3/14/27`, `March 14`), timezone-naive shifts, occasional missing year (must infer from context), rare wrong-day errors (to test the model's tolerance/rejection of near-misses).
- **Venue noise**: known alias substitution (build a small alias table — arenas get renamed by sponsors regularly, this is realistic), abbreviation, occasional venue omission.
- **Structural noise**: missing fields, extra whitespace/HTML entities (simulating scraped data), duplicate near-identical listings from different sources.
- Also generate **true negatives**: noisy listings for events NOT in the small candidate-catalog slice being evaluated (e.g. a same-day different-city game with similar team name), plus genuinely new events that should resolve to `NO_MATCH`. Without hard negatives the eval is meaningless.
- Store ground truth (`listing_id -> true_event_id or NO_MATCH`) separately for evaluation only, never fed to the model.

## 6. Pipeline architecture

```
listing --> [normalize] --> [candidate blocking (Elasticsearch)] --> [feature extraction]
        --> [LightGBM ranker] --> top candidate + confidence
        --> if confidence >= threshold: MATCH
        --> if confidence < threshold and > review_floor: [agentic review] --> MATCH / NO_MATCH + reasoning
        --> else: NO_MATCH (new event)
```

### 6.1 Normalization
- Lowercase, strip punctuation/HTML entities, parse dates into a canonical `datetime` with explicit handling of ambiguous/missing years (infer from surrounding context or flag as low-confidence), resolve venue names through the alias table where known (partial credit even if not resolved — the model should still work off fuzzy string match for unseen aliases).

### 6.2 Candidate blocking (Elasticsearch)
- Index the canonical catalog in Elasticsearch with fuzzy matching on team/artist names and venue, and a date range filter (±3 days, since listings can be date-noisy).
- For each incoming listing, retrieve top-K candidates (e.g. K=10) instead of comparing against the full catalog — this is the production-realistic step and the one most portfolio projects skip.
- **Report blocking recall@K**: what fraction of the time does the true match survive into the top-K candidates, as a function of K. This is the headline tradeoff metric — show the curve (recall vs K vs downstream latency cost).

### 6.3 Feature extraction (per listing/candidate pair)
- String similarity: Levenshtein ratio, Jaro-Winkler, token-set overlap (Jaccard) on team/artist names — computed both on raw and normalized strings.
- Date proximity: absolute day difference, exact match flag, same-weekday flag (catches "right date wrong year" cases).
- Venue match: exact match, alias-table match, fuzzy string match, geo-distance if coordinates available.
- Embedding similarity: sentence-transformers (e.g. all-MiniLM-L6-v2, consistent with prior project experience) on the normalized listing text vs candidate's canonical description; cosine similarity as a feature.
- Source reliability (optional): a feature for which feed the listing came from, if some feeds are noisier than others — realistic and gives the model something structural to lean on.

### 6.4 Ranking model — LightGBM
- Frame as pairwise learning-to-rank (LightGBM's `lambdarank` objective) over the candidate set per listing, OR pointwise binary classification (match / not-match) per pair, scored and top-1 selected. Implement pointwise first (simpler, easier to calibrate a threshold), pairwise as a stretch comparison.
- Train/val/test split by **event**, not by listing — listings of the same event must not leak across splits, or the eval is inflated.
- Output: for each listing, ranked candidates with calibrated confidence scores (Platt scaling or isotonic regression on top of raw LightGBM scores, so "confidence" is actually interpretable as a probability).

### 6.5 Confidence thresholding
- Two thresholds: `match_threshold` (auto-accept) and `review_floor` (below this, straight to `NO_MATCH`; between the two, send to agentic review).
- Tune both thresholds on the validation set to hit a target precision (e.g. 99% precision on auto-accepted matches — false positives are worse than false negatives here, since wrongly merging two different events is a real product bug).

## 7. Agentic fallback (stretch, but a strong differentiator)

For listings in the "ambiguous" confidence band:
- A LangChain or Strands tool-calling agent given the listing, its top candidates, and their feature breakdowns.
- Tools: `search_recent_news(team_or_venue)` (catches venue renames, team relocations, schedule changes not in the static catalog), `get_candidate_details(event_id)`, `compare_listings(a, b)` (returns a structured diff).
- The agent either confirms a specific candidate with a written rationale, or explains why it's genuinely a new/unresolvable event.
- **Guardrail**: the agent's confirmation still has to be logged with its reasoning trace and treated as a lower-trust match tier in metrics (report agentic-tier precision separately from the auto-accepted LightGBM tier) — don't blend the numbers and claim one accuracy figure.
- Keep this scoped: it only runs on the ambiguous band, which should be a small fraction of traffic if the ranker is well-tuned. If it's running on 40% of listings, that's a signal the ranker/thresholds need work, not that the agent needs to do more.

## 8. Evaluation & metrics (the deliverable — write these up properly)

- **Blocking**: recall@K curve (K = 1, 3, 5, 10, 20) and retrieval latency at each K.
- **Ranking/classification**: precision, recall, F1 at the chosen threshold; also report the full PR curve so the threshold choice is justified, not arbitrary.
- **Feature ablation**: F1 with (a) string similarity only, (b) + date/venue features, (c) + embeddings — show the marginal contribution of each group. This is the single most interview-friendly artifact in the project — a table or chart of "which features actually mattered."
- **Auto-accept tier vs agentic tier**: precision/recall reported separately, plus what % of total volume falls in each tier.
- **Failure analysis**: pull 15-20 actual misclassifications, categorize them (venue alias not in table, near-duplicate events on the same date, genuinely ambiguous), and write up what would fix each category. This shows real engineering judgment, not just a metrics dump.
- **Latency**: end-to-end ms/listing at p50/p99 for the auto-accept path (blocking + features + LightGBM inference).

## 9. Data model (rough)

```
events(event_id, home_team, away_team, venue_canonical, event_date, venue_lat, venue_lon)
venue_aliases(alias, venue_canonical)
listings(listing_id, raw_title, raw_date, raw_venue, source)
ground_truth(listing_id, true_event_id_or_null)   # eval only, never used as a feature
match_results(listing_id, predicted_event_id_or_null, confidence, tier[auto|agentic|no_match], features jsonb)
agentic_reviews(listing_id, reasoning_trace, tools_called, final_decision)
```

## 10. Suggested build order (phases)

1. **Phase 1 — data**: pull/build the canonical catalog, write the noise generator, produce the labeled synthetic dataset with hard negatives. Do not skip hard negatives.
2. **Phase 2 — blocking**: stand up Elasticsearch, index the catalog, implement candidate retrieval, measure recall@K before any ML is involved. This gives you a baseline and the recall/K curve early.
3. **Phase 3 — features + baseline model**: implement feature extraction, train a simple baseline (logistic regression on string similarity alone) to get a first F1 number, then swap in LightGBM.
4. **Phase 4 — embeddings + full ranker**: add sentence-transformer embeddings, retrain, run the ablation study comparing feature groups.
5. **Phase 5 — calibration + thresholds**: calibrate confidence scores, tune the two thresholds against target precision, finalize the auto-accept / agentic / no-match tiers.
6. **Phase 6 — agentic fallback**: build the LangChain/Strands agent for the ambiguous band, log reasoning traces, evaluate its tier separately.
7. **Phase 7 — write-up**: failure analysis, all metrics/charts, README, optional small FastAPI wrapper (`POST /match` -> result) so it's runnable as a demo, not just a notebook.

## 11. Stack

- ML: LightGBM, sentence-transformers (PyTorch backend), scikit-learn (calibration, baselines)
- Retrieval: Elasticsearch
- Agentic layer: LangChain or Strands (matches SeatGeek's stated stack), any LLM provider for the agent
- Data/eval: pandas, matplotlib or the project's existing charting approach for the ablation/recall curves
- Optional serving: FastAPI wrapping the trained pipeline for a `POST /match` demo endpoint
- Infra: Docker for Elasticsearch + the API, if deployed
