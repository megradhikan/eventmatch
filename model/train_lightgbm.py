"""
PRD 6.4: pointwise binary LightGBM classifier (match/not-match per pair) on
features/full_feature_set.csv, trained on `train`, evaluated on `val`/`test`.
Split is by EVENT (inherited via listing -> split, verified in features/build_pairs.py),
never by listing, so no event's listings cross train/val/test.
"""
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
)

FEATURES_PATH = Path(__file__).parent.parent / "features" / "full_feature_set.csv"
MODEL_DIR = Path(__file__).parent
NON_FEATURE_COLS = ["listing_id", "candidate_event_id", "label", "split"]


def best_f1_point(y, proba):
    p, r, t = precision_recall_curve(y, proba)
    f1 = 2 * p * r / (p + r + 1e-12)
    idx = f1[:-1].argmax() if len(f1) > 1 else 0
    return {
        "threshold": float(t[idx]) if idx < len(t) else 0.5,
        "precision": float(p[idx]),
        "recall": float(r[idx]),
        "f1": float(f1[idx]),
    }


def evaluate(model, X, y):
    proba = model.predict(X, num_iteration=model.best_iteration)
    preds_05 = (proba >= 0.5).astype(int)
    p5, r5, f5, _ = precision_recall_fscore_support(y, preds_05, average="binary", zero_division=0)
    return {
        "threshold_0.5": {"precision": float(p5), "recall": float(r5), "f1": float(f5)},
        "best_f1": best_f1_point(y, proba),
        "average_precision": float(average_precision_score(y, proba)),
    }, proba


def top1_accuracy(df_split, proba):
    """Per-listing: does the highest-scored candidate match the true label?
    Restricted to listings that have a true positive among their candidates
    (blocking-recall-bounded), consistent with how Phase 3's embedding
    benchmark framed this."""
    d = df_split.copy()
    d["proba"] = proba
    has_positive = d.groupby("listing_id")["label"].transform("max") == 1
    d = d[has_positive]
    top1 = d.loc[d.groupby("listing_id")["proba"].idxmax()]
    return float((top1["label"] == 1).mean()), int(len(top1))


def train_and_eval(df, feature_cols, label="full"):
    train = df[df.split == "train"]
    val = df[df.split == "val"]
    test = df[df.split == "test"]

    train_set = lgb.Dataset(train[feature_cols], label=train["label"])
    val_set = lgb.Dataset(val[feature_cols], label=val["label"], reference=train_set)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": (train["label"] == 0).sum() / max((train["label"] == 1).sum(), 1),
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )

    val_metrics, val_proba = evaluate(model, val[feature_cols], val["label"])
    test_metrics, test_proba = evaluate(model, test[feature_cols], test["label"])

    val_top1, val_n = top1_accuracy(val, val_proba)
    test_top1, test_n = top1_accuracy(test, test_proba)
    val_metrics["top1_accuracy"] = {"accuracy": val_top1, "n_listings": val_n}
    test_metrics["top1_accuracy"] = {"accuracy": test_top1, "n_listings": test_n}

    importances = dict(zip(feature_cols, model.feature_importance(importance_type="gain").tolist()))

    return {
        "label": label,
        "n_features": len(feature_cols),
        "features": feature_cols,
        "best_iteration": model.best_iteration,
        "val": val_metrics,
        "test": test_metrics,
        "feature_importance_gain": importances,
    }, model, test_proba


def main():
    df = pd.read_csv(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    print(f"Training on {len(feature_cols)} features, {len(df)} rows "
          f"(train={sum(df.split=='train')}, val={sum(df.split=='val')}, test={sum(df.split=='test')})")

    results, model, test_proba = train_and_eval(df, feature_cols, label="full_feature_set")

    print(json.dumps({k: v for k, v in results.items() if k != "feature_importance_gain"}, indent=2))

    top_features = sorted(results["feature_importance_gain"].items(), key=lambda x: -x[1])[:10]
    print("\nTop 10 features by gain:")
    for f, g in top_features:
        print(f"  {f}: {g:.1f}")

    model.save_model(str(MODEL_DIR / "lightgbm_full.txt"))
    (MODEL_DIR / "lightgbm_full_results.json").write_text(json.dumps(results, indent=2))

    # Save test-set predictions for calibration (Phase 5) and failure analysis (Phase 7)
    test_df = df[df.split == "test"][["listing_id", "candidate_event_id", "label"]].copy()
    test_df["raw_score"] = test_proba
    test_df.to_csv(MODEL_DIR / "test_predictions.csv", index=False)

    print(f"\nWrote {MODEL_DIR / 'lightgbm_full.txt'}, lightgbm_full_results.json, test_predictions.csv")


if __name__ == "__main__":
    main()
