"""
PRD 10 phase 3: baseline logistic regression on string-similarity features
ALONE (no date/venue/source/embeddings), trained on the `train` split of
features/string_date_venue_features.csv and evaluated on `val` and `test`.

This is a per-PAIR classifier (does this candidate match this listing),
not a top-1-per-listing ranker -- kept deliberately simple per PRD 10
phase 3 ("train a simple baseline ... to get a first F1 number").

Run: .venv/bin/python3 model/baseline_lr.py
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support

FEATURES_PATH = Path(__file__).parent.parent / "features" / "string_date_venue_features.csv"
OUT_PATH = Path(__file__).parent / "baseline_lr_results.json"

STRING_FEATURES = [
    "str_lev_ratio_raw",
    "str_lev_ratio_norm",
    "str_jaro_winkler_raw",
    "str_jaro_winkler_norm",
    "str_token_jaccard_raw",
    "str_token_jaccard_norm",
    "str_token_sort_ratio_raw",
    "str_token_sort_ratio_norm",
]


def evaluate(model, X, y, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, preds, average="binary", zero_division=0
    )

    p_curve, r_curve, thresh_curve = precision_recall_curve(y, proba)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + 1e-12)
    best_idx = f1_curve[:-1].argmax() if len(f1_curve) > 1 else 0
    best_f1 = float(f1_curve[best_idx])
    best_precision = float(p_curve[best_idx])
    best_recall = float(r_curve[best_idx])
    best_threshold = float(thresh_curve[best_idx]) if len(thresh_curve) > best_idx else threshold

    return {
        "threshold_0.5": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        },
        "best_f1_on_pr_curve": {
            "precision": best_precision,
            "recall": best_recall,
            "f1": best_f1,
            "threshold": best_threshold,
        },
        "n_pairs": int(len(y)),
        "n_positive": int(y.sum()),
    }


def main():
    df = pd.read_csv(FEATURES_PATH)

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]

    X_train, y_train = train[STRING_FEATURES], train["label"]
    X_val, y_val = val[STRING_FEATURES], val["label"]
    X_test, y_test = test[STRING_FEATURES], test["label"]

    print(f"train: {len(X_train)} pairs ({y_train.sum()} positive)")
    print(f"val:   {len(X_val)} pairs ({y_val.sum()} positive)")
    print(f"test:  {len(X_test)} pairs ({y_test.sum()} positive)")
    print(f"features used ({len(STRING_FEATURES)}): {STRING_FEATURES}")

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    results = {
        "features": STRING_FEATURES,
        "n_features": len(STRING_FEATURES),
        "val": evaluate(model, X_val, y_val),
        "test": evaluate(model, X_test, y_test),
        "coefficients": dict(zip(STRING_FEATURES, model.coef_[0].tolist())),
        "intercept": float(model.intercept_[0]),
    }

    print("\n=== VAL ===")
    print(json.dumps(results["val"], indent=2))
    print("\n=== TEST ===")
    print(json.dumps(results["test"], indent=2))

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote results to {OUT_PATH}")


if __name__ == "__main__":
    main()
