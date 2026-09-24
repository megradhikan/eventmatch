"""
PRD 6.3 embedding similarity feature (Phase 4 scope, embedding slice only):
sentence-transformers (all-MiniLM-L6-v2) cosine similarity between each
listing's text and each candidate event's text, as a feature per
(listing_id, candidate_event_id) pair.

Design decision -- raw text, not normalize_text()-stripped text, goes into
the encoder. normalize_text() (blocking/normalize.py) is built for exact/
fuzzy STRING matching: it lowercases and strips all punctuation, which
collapses structurally meaningful separators like "@" ("at"), "vs", and
date slashes into blank space. A sentence-transformer is trained on real,
punctuated natural-language sentences, and MiniLM's WordPiece tokenizer is
itself case-folding, so lowercasing/punctuation-stripping ahead of time
buys nothing and actively removes signal the model can otherwise use
(e.g. "Lakers @ Celtics" reads as a relation, "lakers celtics" reads as a
bag of tokens). Raw listing text is used as-is; candidate event text is
built as a clean natural-language sentence template.

Efficiency: embeds each of the ~6,190 unique listings and ~1,338 unique
candidate events exactly ONCE (not once per of the 101,044 pairs), caches
the vectors, then computes cosine similarity per pair from the cached
vectors. This mirrors the realistic production design (listings and
catalog events are each encoded once; only the O(pairs) cosine-similarity
step scales with blocking fan-out) and avoids ~94x redundant encode calls.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
OUT_PATH = Path(__file__).parent / "embedding_features.csv"
MODEL_NAME = "all-MiniLM-L6-v2"


def listing_text(row):
    parts = [row["raw_title"], row["raw_venue"], row["raw_date"]]
    return " ".join(p for p in parts if isinstance(p, str) and p.strip())


def candidate_text(row):
    return (
        f"{row['cand_home_team']} vs {row['cand_away_team']} "
        f"at {row['cand_venue_canonical']} on {row['cand_event_date']}"
    )


def main():
    print("Loading candidate_pairs.csv ...", flush=True)
    pairs = pd.read_csv(DATA_DIR / "candidate_pairs.csv", dtype=str)
    pairs["label"] = pairs["label"].astype(int)
    print(f"  {len(pairs)} pairs loaded", flush=True)

    # --- Unique listings (encode once each) ---
    listing_cols = ["listing_id", "raw_title", "raw_venue", "raw_date"]
    unique_listings = pairs[listing_cols].drop_duplicates(subset="listing_id").reset_index(drop=True)
    unique_listings["text"] = unique_listings.apply(listing_text, axis=1)
    print(f"  {len(unique_listings)} unique listings to embed", flush=True)

    # --- Unique candidate events (encode once each) ---
    cand_cols = [
        "candidate_event_id", "cand_home_team", "cand_away_team",
        "cand_venue_canonical", "cand_event_date",
    ]
    unique_candidates = pairs[cand_cols].drop_duplicates(subset="candidate_event_id").reset_index(drop=True)
    unique_candidates["text"] = unique_candidates.apply(candidate_text, axis=1)
    print(f"  {len(unique_candidates)} unique candidate events to embed", flush=True)

    print(f"Loading model {MODEL_NAME} ...", flush=True)
    model = SentenceTransformer(MODEL_NAME)

    t0 = time.time()
    listing_vecs = model.encode(
        unique_listings["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    t1 = time.time()
    candidate_vecs = model.encode(
        unique_candidates["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    t2 = time.time()

    listing_encode_s = t1 - t0
    candidate_encode_s = t2 - t1
    total_encode_s = t2 - t0
    print(f"Listing encode: {listing_encode_s:.2f}s for {len(unique_listings)} texts", flush=True)
    print(f"Candidate encode: {candidate_encode_s:.2f}s for {len(unique_candidates)} texts", flush=True)
    print(f"Total unique-item embedding wall clock: {total_encode_s:.2f}s", flush=True)

    # id -> row index into the *_vecs arrays
    listing_idx = {lid: i for i, lid in enumerate(unique_listings["listing_id"])}
    candidate_idx = {cid: i for i, cid in enumerate(unique_candidates["candidate_event_id"])}

    li = pairs["listing_id"].map(listing_idx).to_numpy()
    ci = pairs["candidate_event_id"].map(candidate_idx).to_numpy()

    # embeddings are already L2-normalized, so cosine similarity == dot product
    cos_sim = np.sum(listing_vecs[li] * candidate_vecs[ci], axis=1)

    out = pd.DataFrame({
        "listing_id": pairs["listing_id"],
        "candidate_event_id": pairs["candidate_event_id"],
        "label": pairs["label"],
        "split": pairs["split"],
        "embedding_cosine_sim": cos_sim,
    })
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} rows to {OUT_PATH}", flush=True)

    # Persist timing so the benchmark script (run separately) can quote it
    # without re-embedding.
    timing_path = Path(__file__).parent / "_embedding_timing.txt"
    timing_path.write_text(
        f"listing_encode_s={listing_encode_s:.2f}\n"
        f"n_listings={len(unique_listings)}\n"
        f"candidate_encode_s={candidate_encode_s:.2f}\n"
        f"n_candidates={len(unique_candidates)}\n"
        f"total_encode_s={total_encode_s:.2f}\n"
    )


if __name__ == "__main__":
    main()
