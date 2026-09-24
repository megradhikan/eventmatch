# Phase 3b: Sentence-Transformer Embedding Feature

Scope: the sentence-transformer embedding slice of PRD 6.3 only -- embedding
each listing and candidate event, cosine similarity as a feature, and a
standalone embedding-only benchmark. String/date/venue features are being
built in parallel by another agent; LightGBM training and the combined
feature-group ablation study are Phase 4 (later), not this work.

## Text representation

- **Model**: `all-MiniLM-L6-v2` (named explicitly in PRD 6.3), via
  `sentence-transformers`, run on CPU.
- **Listing text**: `"{raw_title} {raw_venue} {raw_date}"`, e.g.
  `"Cleveland Cavaliers @ Phoenix Suns Mortgage Matchup Center 1/31/2026"`.
- **Candidate event text**: `"{cand_home_team} vs {cand_away_team} at {cand_venue_canonical} on {cand_event_date}"`,
  e.g. `"Phoenix Suns vs Cleveland Cavaliers at Mortgage Matchup Center on 2026-01-31T02:00Z"`.

**Design decision: raw text, not `normalize_text()`-normalized text.**
`blocking/normalize.py`'s `normalize_text()` lowercases and strips all
punctuation -- it's built for exact/fuzzy *string* matching downstream.
Feeding that into a sentence-transformer would collapse structurally
meaningful separators: `"@"` (implies "at"/away-at-home), `"vs"`, and the
slashes in `"1/31/2026"` all carry relational/structural information that a
pretrained sentence encoder can use, but that a bag of lowercased tokens
loses. MiniLM's own WordPiece tokenizer already case-folds internally, so
pre-lowercasing buys nothing there either. Net: normalization is the right
call for string-similarity features (that's a different, complementary
feature group), but for embeddings the raw listing string and a clean
natural-language candidate sentence are the better inputs. This is a
judgment call, not a measured result -- it wasn't A/B'd against normalized
text within this task's scope.

## Efficiency: encode once per unique item, not once per pair

`candidate_pairs.csv` has 101,044 rows but only 6,190 unique `listing_id`s
and 1,338 unique `candidate_event_id`s (the full catalog). Encoding text per
pair would mean ~101K encoder calls with massive redundancy (every listing's
text gets re-encoded once per candidate it was blocked against, up to 20x).
Instead: dedupe to unique listings and unique candidate events, encode each
set once, cache the resulting vectors, then compute cosine similarity per
pair as a dot product of the (L2-normalized) cached vectors -- O(pairs) for
the cheap step, O(unique items) for the expensive encode step. This is also
the realistic production shape: a catalog event's embedding doesn't change
per incoming listing, so it'd be precomputed and cached in a real system.

## Output

`features/embedding_features.csv`, 101,044 rows, one per input row in
`data/processed/candidate_pairs.csv`, columns:
`listing_id, candidate_event_id, label, split, embedding_cosine_sim`
(`label` and `split` copied through, not recomputed).

Build with: `python features/build_embedding_features.py`

## Wall-clock (embedding all unique items, CPU, MacBook)

| Step | Items | Time |
|---|---|---|
| Encode unique listings | 6,190 | 2.54s |
| Encode unique candidate events | 1,338 | 0.56s |
| **Total** | 7,528 | **3.10s** |

That's ~2,430 texts/sec on CPU for this short-text workload, with
`normalize_embeddings=True` and batch_size=64 (`sentence-transformers`
default mini-batching, no GPU). For the eventual end-to-end latency
write-up (PRD 8, p50/p99 ms/listing): this is the batch/offline embedding
cost, not per-request inference latency -- at serve time only the
*incoming listing* needs a fresh embed (candidate vectors are precomputed),
so a single-listing embed is the relevant number, not this batch figure.
A single-item encode call was not separately timed here (TBD if needed for
the latency write-up; batching amortizes model-load and vectorization
overhead that a single call wouldn't get for free).

## Standalone benchmark: embedding-only matching accuracy

Computed from `features/embedding_features.csv` alone -- no string/date/
venue features, no LightGBM. Run with: `python features/embedding_benchmark.py`

### 1. Top-1 accuracy (TEST split)

For each TEST listing, predicted match = the candidate with the highest
`embedding_cosine_sim` among that listing's blocked candidates. Restricted
to listings whose true match actually survived blocking (present among its
candidates) -- listings where blocking itself dropped the true match can't
be won by any downstream feature, so including them would understate the
embedding signal specifically rather than measure it.

- TEST listings total: 919
- TEST listings with true match surviving blocking: 798
- Top-1 correct: 538
- **Top-1 accuracy: 0.6742** (538/798)

### 2. Pairwise match/no-match classification at F1-optimal threshold

Treats each (listing, candidate) pair as a binary classification problem
using `embedding_cosine_sim` alone as the score. Threshold swept on VAL via
`sklearn.metrics.precision_recall_curve`, F1-optimal point selected on VAL,
then that *fixed* threshold applied to TEST. This framing (not top-1) is the
one comparable to the string-similarity baseline the other feature-extraction
agent is building, since both reduce to "is this pair a match" at a
threshold.

- F1-optimal threshold (chosen on VAL): **0.8619**
- VAL at that threshold: precision 0.6000, recall 0.4065, F1 0.4847
- **TEST at that threshold: precision 0.5572, recall 0.4148, F1 0.4756**
  (tp=331, fp=263, fn=467)

### Reading these numbers

Embedding-only matching on short, noisy templated listing text is a
moderate signal (F1 ~0.48), well above chance but clearly not sufficient
alone -- consistent with the PRD's framing of embeddings as one feature
group among several (string similarity, date proximity, venue match) that
LightGBM combines, not a standalone matcher. The gap between top-1 accuracy
(0.67, an easier framing -- only has to beat siblings in the same
listing's candidate set) and pairwise F1 (0.48, harder -- has to be
correctly separated from all pairs by one global threshold) is expected and
is itself a useful data point for Phase 4: embeddings rank reasonably well
within a candidate set but don't cleanly threshold in isolation, which is
exactly the kind of feature a learned ranker (LightGBM) is good at
combining with others rather than thresholding on its own.

All numbers above are measured directly from this run; none are estimated
or invented.
