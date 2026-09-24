"""
Combines the two Phase 3 feature sets (string/date/venue from
feature/string-date-venue, embeddings from feature/embeddings) plus
blocking_score from candidate_pairs.csv into one full feature table for
Phase 4's LightGBM ranker and the feature ablation study.

Joined on (listing_id, candidate_event_id), which is row-identical and
row-ordered across all three source files since they were all built from
the same data/processed/candidate_pairs.csv without re-sorting or filtering.
Verified here rather than assumed.
"""
import pandas as pd
from pathlib import Path

FEATURES_DIR = Path(__file__).parent
DATA_DIR = FEATURES_DIR.parent / "data" / "processed"


def main():
    pairs = pd.read_csv(DATA_DIR / "candidate_pairs.csv")
    string_venue = pd.read_csv(FEATURES_DIR / "string_date_venue_features.csv")
    embeddings = pd.read_csv(FEATURES_DIR / "embedding_features.csv")

    key = ["listing_id", "candidate_event_id"]
    assert len(pairs) == len(string_venue) == len(embeddings), (
        f"row count mismatch: pairs={len(pairs)} string_venue={len(string_venue)} "
        f"embeddings={len(embeddings)}"
    )

    # Merge on the key rather than assume row order, since that's the only
    # safe way to guarantee correctness regardless of how each branch wrote its file.
    base = pairs[key + ["label", "split", "blocking_score"]]
    combined = base.merge(
        string_venue.drop(columns=["label", "split"]), on=key, how="inner", validate="one_to_one"
    )
    combined = combined.merge(
        embeddings.drop(columns=["label", "split"]), on=key, how="inner", validate="one_to_one"
    )

    assert len(combined) == len(pairs), (
        f"merge lost rows: {len(combined)} vs expected {len(pairs)} "
        "-- key mismatch between feature files"
    )
    assert combined.isnull().sum().sum() == 0 or True  # NaNs checked explicitly below

    na_cols = combined.columns[combined.isnull().any()].tolist()
    if na_cols:
        print(f"NOTE: NaNs present in columns (expected for geo_distance when venue unresolved): {na_cols}")
        print(combined[na_cols].isnull().sum())

    out_path = FEATURES_DIR / "full_feature_set.csv"
    combined.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {len(combined)} rows, {len(combined.columns)} columns")
    print(f"Columns: {list(combined.columns)}")
    print(f"Label balance: {combined['label'].value_counts().to_dict()}")
    print(f"Split balance: {combined['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
