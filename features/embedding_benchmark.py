"""
Standalone benchmark of embedding-only matching (PRD 6.3 / phase-4 embedding
slice). Uses ONLY features/embedding_features.csv -- no string/date/venue
features, no LightGBM. Two framings, per the task spec:

1. Top-1 "would picking the highest-cosine candidate get it right" accuracy,
   on the TEST split, restricted to listings whose true match actually
   survived blocking (is present among that listing's candidates) -- listings
   where blocking itself dropped the true match can't be won by any feature,
   so folding them in would understate the embedding signal specifically.

2. Pairwise binary "is this (listing, candidate) pair a match" classification
   using embedding_cosine_sim as the sole score: sweep thresholds on VAL,
   pick the F1-optimal one, then report precision/recall/F1 at that fixed
   threshold on TEST. This is the framing comparable to the string-similarity
   baseline the other feature-extraction agent is building.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

FEATURES_PATH = Path(__file__).parent / "embedding_features.csv"


def top1_accuracy(df, split):
    sub = df[df["split"] == split]
    n_considered = 0
    n_correct = 0
    for listing_id, g in sub.groupby("listing_id"):
        if g["label"].sum() == 0:
            # true match did not survive blocking for this listing -- can't
            # be won on embeddings (or anything else), exclude per spec.
            continue
        n_considered += 1
        top = g.loc[g["embedding_cosine_sim"].idxmax()]
        if top["label"] == 1:
            n_correct += 1
    return n_correct, n_considered


def f1_optimal_threshold(df, split):
    sub = df[df["split"] == split]
    precision, recall, thresholds = precision_recall_curve(
        sub["label"].to_numpy(), sub["embedding_cosine_sim"].to_numpy()
    )
    # precision_recall_curve returns len(thresholds) == len(precision) - 1
    p, r = precision[:-1], recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
    best_i = int(np.argmax(f1))
    return thresholds[best_i], p[best_i], r[best_i], f1[best_i]


def precision_recall_f1_at_threshold(df, split, threshold):
    sub = df[df["split"] == split]
    pred = (sub["embedding_cosine_sim"] >= threshold).astype(int)
    y = sub["label"].to_numpy()
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, tp, fp, fn


def main():
    df = pd.read_csv(FEATURES_PATH)

    print("=== Top-1 embedding-only accuracy (TEST split) ===")
    n_correct, n_considered = top1_accuracy(df, "test")
    n_test_listings = df[df["split"] == "test"]["listing_id"].nunique()
    acc = n_correct / n_considered if n_considered else float("nan")
    print(f"  test listings total: {n_test_listings}")
    print(f"  test listings with true match surviving blocking: {n_considered}")
    print(f"  top-1 correct: {n_correct}")
    print(f"  top-1 accuracy: {acc:.4f}")

    print()
    print("=== F1-optimal threshold, tuned on VAL, reported on TEST ===")
    thr, val_p, val_r, val_f1 = f1_optimal_threshold(df, "val")
    print(f"  chosen threshold (val F1-optimal): {thr:.4f}")
    print(f"  val precision/recall/F1 at that threshold: {val_p:.4f} / {val_r:.4f} / {val_f1:.4f}")

    test_p, test_r, test_f1, tp, fp, fn = precision_recall_f1_at_threshold(df, "test", thr)
    print(f"  test precision: {test_p:.4f}")
    print(f"  test recall:    {test_r:.4f}")
    print(f"  test F1:        {test_f1:.4f}")
    print(f"  test tp={tp} fp={fp} fn={fn}")


if __name__ == "__main__":
    main()
