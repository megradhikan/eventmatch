"""
PRD 6.5: calibrate raw LightGBM scores into interpretable probabilities
(isotonic regression, fit on val), then tune two thresholds on val:
  - match_threshold: auto-accept, smallest threshold hitting target
    precision (false positives -- wrongly merging two different events --
    are worse than false negatives here).
  - review_floor: below this, straight to NO_MATCH; between the two,
    send to agentic review (Phase 6). Chosen to capture ~98% of
    recoverable true positives, so review_floor sits just below the
    point where cumulative recall starts dropping fast.
Reports ACTUAL precision achieved on held-out test, not the target.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve

FEATURES_PATH = Path(__file__).parent.parent / "features" / "full_feature_set.csv"
MODEL_DIR = Path(__file__).parent
NON_FEATURE_COLS = ["listing_id", "candidate_event_id", "label", "split"]
TARGET_PRECISION = 0.99
TARGET_RECALL_FLOOR = 0.98  # capture this fraction of recoverable true positives above review_floor


def find_match_threshold(y_val, proba_val, target_precision):
    precision, recall, thresholds = precision_recall_curve(y_val, proba_val)
    # precision[] has len(thresholds)+1; align by dropping the last (no-threshold) point
    precision, recall = precision[:-1], recall[:-1]
    # Precision can be noisy at high thresholds on a small val set -- require it stays
    # >= target for every higher threshold too (suffix-min), not just at one point.
    suffix_min_precision = np.minimum.accumulate(precision[::-1])[::-1]
    ok = suffix_min_precision >= target_precision
    if not ok.any():
        # Target unreachable on val -- fall back to the highest-precision threshold available.
        idx = int(np.argmax(precision))
        return float(thresholds[idx]), float(precision[idx]), float(recall[idx]), False
    idx = int(np.argmax(ok))  # first True = lowest threshold meeting the bar
    return float(thresholds[idx]), float(precision[idx]), float(recall[idx]), True


def find_review_floor(y_val, proba_val, recall_target):
    precision, recall, thresholds = precision_recall_curve(y_val, proba_val)
    precision, recall = precision[:-1], recall[:-1]
    ok = recall >= recall_target
    if not ok.any():
        return float(thresholds[0]), float(recall[0])
    idx = int(np.where(ok)[0][-1])  # last True = highest threshold still meeting recall target
    return float(thresholds[idx]), float(recall[idx])


def apply_tiers(proba, match_threshold, review_floor):
    tier = np.where(proba >= match_threshold, "auto",
                     np.where(proba >= review_floor, "agentic", "no_match"))
    return tier


def main():
    df = pd.read_csv(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    model = lgb.Booster(model_file=str(MODEL_DIR / "lightgbm_full.txt"))

    val = df[df.split == "val"]
    test = df[df.split == "test"]
    val_raw = model.predict(val[feature_cols])
    test_raw = model.predict(test[feature_cols])

    # Isotonic regression: fit raw score -> P(match) on val, apply to test.
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_raw, val["label"])
    val_cal = calibrator.predict(val_raw)
    test_cal = calibrator.predict(test_raw)

    match_threshold, val_p_at_match, val_r_at_match, reachable = find_match_threshold(
        val["label"], val_cal, TARGET_PRECISION
    )
    review_floor, val_r_at_floor = find_review_floor(val["label"], val_cal, TARGET_RECALL_FLOOR)
    if review_floor >= match_threshold:
        review_floor = match_threshold * 0.5  # keep a real agentic band even if val is small/noisy

    print(f"match_threshold={match_threshold:.4f} (val precision={val_p_at_match:.4f}, "
          f"recall={val_r_at_match:.4f}, target {TARGET_PRECISION} reachable={reachable})")
    print(f"review_floor={review_floor:.4f} (val recall at floor={val_r_at_floor:.4f})")

    # Tiers are a per-LISTING decision (one predicted match or NO_MATCH per
    # incoming listing), made from that listing's TOP-scored candidate among
    # its blocking candidates -- not a per-pair decision. Pair-level tiering
    # would double count: a listing with 19 candidates has 19 rows, but only
    # one real prediction. This matches the PRD's "% of total volume" framing,
    # where volume = listings, i.e. real traffic.
    test_eval = test[["listing_id", "label"]].copy()
    test_eval["calibrated_score"] = test_cal
    top = test_eval.loc[test_eval.groupby("listing_id")["calibrated_score"].idxmax()]
    top_tier = apply_tiers(top["calibrated_score"].values, match_threshold, review_floor)
    top_labels = top["label"].values

    auto_mask = top_tier == "auto"
    n_auto = int(auto_mask.sum())
    auto_precision = float(top_labels[auto_mask].mean()) if n_auto else None
    # Recall here = of listings that HAVE a true match among their candidates
    # (i.e. survived blocking), what fraction get auto-accepted correctly.
    has_positive = test_eval.groupby("listing_id")["label"].max()
    n_listings_with_positive = int(has_positive.sum())
    auto_recall_of_all_positives = float(top_labels[auto_mask].sum() / max(n_listings_with_positive, 1))

    tier_volume = {t: int((top_tier == t).sum()) for t in ["auto", "agentic", "no_match"]}
    tier_volume_pct = {k: v / len(top_tier) for k, v in tier_volume.items()}
    n_test_listings = len(top_tier)

    results = {
        "target_precision": TARGET_PRECISION,
        "target_recall_floor": TARGET_RECALL_FLOOR,
        "match_threshold": match_threshold,
        "review_floor": review_floor,
        "val": {
            "precision_at_match_threshold": val_p_at_match,
            "recall_at_match_threshold": val_r_at_match,
            "target_reachable_on_val": reachable,
        },
        "test": {
            "n_test_listings": n_test_listings,
            "n_test_listings_with_true_match_in_candidates": n_listings_with_positive,
            "auto_accept_precision_ACTUAL": auto_precision,
            "auto_accept_n": n_auto,
            "auto_accept_recall_of_recoverable_positives": auto_recall_of_all_positives,
            "tier_volume_counts_per_listing": tier_volume,
            "tier_volume_pct_per_listing": tier_volume_pct,
        },
    }
    print(json.dumps(results, indent=2))

    (MODEL_DIR / "calibration_results.json").write_text(json.dumps(results, indent=2))

    # Persist calibrated test predictions (all candidate rows, for context) plus
    # the per-listing top-candidate tier assignment Phase 6's agentic fallback
    # and Phase 7's failure analysis actually operate on.
    test_out = test[["listing_id", "candidate_event_id", "label"]].copy()
    test_out["raw_score"] = test_raw
    test_out["calibrated_score"] = test_cal
    test_out["is_top_candidate"] = False
    test_out.loc[top.index, "is_top_candidate"] = True
    top_tier_by_listing = dict(zip(top["listing_id"], top_tier))
    test_out["listing_tier"] = test_out["listing_id"].map(top_tier_by_listing)
    test_out.to_csv(MODEL_DIR / "test_predictions_calibrated.csv", index=False)

    import joblib
    joblib.dump(calibrator, MODEL_DIR / "isotonic_calibrator.joblib")
    print(f"\nWrote calibration_results.json, test_predictions_calibrated.csv, isotonic_calibrator.joblib")


if __name__ == "__main__":
    main()
