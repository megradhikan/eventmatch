"""
Shared prep step ahead of Phase 3: runs every listing through blocking
(top-K=20) to produce the (listing, candidate) pairs both feature-extraction
subagents build their feature columns on top of. Centralizing this means
both agents' feature CSVs are row-aligned on (listing_id, candidate_event_id)
and can be joined directly in Phase 4, and the train/val/test split logic
(critical to get right per PRD 6.4) only has to be implemented once.

Split assignment is per LISTING, not per pair: a positive listing inherits
its true event's split (data/processed/events_catalog.csv `split` column);
a NO_MATCH listing (no true event) gets a deterministic hash-based 70/15/15
split of its own. Every pair generated from a given listing shares that
listing's split, so no listing's pairs cross train/val/test.
"""
import csv
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "blocking"))
from retrieve import get_client, retrieve_candidates  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
K = 20


def listing_split(listing_id, true_event_id, event_split):
    if true_event_id:
        return event_split[true_event_id]
    h = int(hashlib.md5(listing_id.encode()).hexdigest(), 16) % 100
    if h < 70:
        return "train"
    elif h < 85:
        return "val"
    return "test"


def main():
    with (DATA_DIR / "events_catalog.csv").open() as f:
        catalog_rows = list(csv.DictReader(f))
    catalog = {r["event_id"]: r for r in catalog_rows}
    event_split = {r["event_id"]: r["split"] for r in catalog_rows}

    with (DATA_DIR / "listings.csv").open() as f:
        listings = list(csv.DictReader(f))
    with (DATA_DIR / "ground_truth.csv").open() as f:
        gt = {r["listing_id"]: r["true_event_id"] for r in csv.DictReader(f)}

    es = get_client()
    out_fields = [
        "listing_id", "raw_title", "raw_date", "raw_venue", "source",
        "candidate_event_id", "cand_home_team", "cand_away_team",
        "cand_venue_canonical", "cand_event_date", "cand_venue_lat", "cand_venue_lon",
        "blocking_score", "label", "split",
    ]

    n_pairs = 0
    n_listings_with_no_candidates = 0
    with (DATA_DIR / "candidate_pairs.csv").open("w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=out_fields)
        writer.writeheader()

        for i, listing in enumerate(listings):
            true_event_id = gt[listing["listing_id"]]
            split = listing_split(listing["listing_id"], true_event_id, event_split)
            candidates = retrieve_candidates(
                es, listing["raw_title"], listing["raw_date"], listing["raw_venue"], size=K
            )
            if not candidates:
                n_listings_with_no_candidates += 1
                continue

            for cand in candidates:
                cand_row = catalog[cand["event_id"]]
                writer.writerow({
                    "listing_id": listing["listing_id"],
                    "raw_title": listing["raw_title"],
                    "raw_date": listing["raw_date"],
                    "raw_venue": listing["raw_venue"],
                    "source": listing["source"],
                    "candidate_event_id": cand["event_id"],
                    "cand_home_team": cand_row["home_team"],
                    "cand_away_team": cand_row["away_team"],
                    "cand_venue_canonical": cand_row["venue_canonical"],
                    "cand_event_date": cand_row["event_date"],
                    "cand_venue_lat": cand_row["venue_lat"],
                    "cand_venue_lon": cand_row["venue_lon"],
                    "blocking_score": cand["score"],
                    "label": 1 if cand["event_id"] == true_event_id else 0,
                    "split": split,
                })
                n_pairs += 1

            if i % 1000 == 0:
                print(f"  {i}/{len(listings)} listings processed -> {n_pairs} pairs so far", flush=True)

    print(f"Wrote {n_pairs} candidate pairs from {len(listings)} listings")
    print(f"Listings with zero candidates retrieved (dropped): {n_listings_with_no_candidates}")


if __name__ == "__main__":
    main()
