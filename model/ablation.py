"""
PRD 8: feature ablation -- F1 with (a) string similarity only, (b) +date/venue,
(c) +embeddings. Shows the marginal contribution of each feature group; the
single most interview-friendly artifact in the project per the PRD.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from train_lightgbm import train_and_eval

FEATURES_PATH = Path(__file__).parent.parent / "features" / "full_feature_set.csv"
MODEL_DIR = Path(__file__).parent
DOCS_DIR = Path(__file__).parent.parent / "docs"

STRING_ONLY = [
    "str_lev_ratio_raw", "str_lev_ratio_norm", "str_jaro_winkler_raw", "str_jaro_winkler_norm",
    "str_token_jaccard_raw", "str_token_jaccard_norm", "str_token_sort_ratio_raw", "str_token_sort_ratio_norm",
]
DATE_VENUE = [
    "date_missing", "date_year_inferred", "date_day_diff", "date_exact_match", "date_same_weekday",
    "venue_missing", "venue_exact_match", "venue_alias_match", "venue_fuzzy_score",
    "venue_geo_distance_km", "venue_geo_distance_missing",
]
EMBEDDING = ["embedding_cosine_sim"]
OTHER = ["blocking_score", "source_official_feed", "source_partner_feed_a",
         "source_scraped_feed_b", "source_scraped_feed_c"]


def main():
    df = pd.read_csv(FEATURES_PATH)

    stages = [
        ("(a) string similarity only", STRING_ONLY),
        ("(b) + date/venue", STRING_ONLY + DATE_VENUE),
        ("(b) + date/venue + blocking/source", STRING_ONLY + DATE_VENUE + OTHER),
        ("(c) + embeddings (full feature set)", STRING_ONLY + DATE_VENUE + OTHER + EMBEDDING),
    ]

    all_results = []
    for name, cols in stages:
        results, _, _ = train_and_eval(df, cols, label=name)
        all_results.append(results)
        print(f"{name}: test F1(best)={results['test']['best_f1']['f1']:.4f}  "
              f"test top1_acc={results['test']['top1_accuracy']['accuracy']:.4f}  "
              f"n_features={len(cols)}")

    (MODEL_DIR / "ablation_results.json").write_text(
        json.dumps([{k: v for k, v in r.items() if k != "feature_importance_gain"} for r in all_results], indent=2)
    )

    names = [r["label"] for r in all_results]
    f1s = [r["test"]["best_f1"]["f1"] for r in all_results]
    top1s = [r["test"]["top1_accuracy"]["accuracy"] for r in all_results]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = range(len(names))
    ax1.bar([i - 0.15 for i in x], f1s, width=0.3, label="Test F1 (best threshold)", color="#2b6cb0")
    ax1.bar([i + 0.15 for i in x], top1s, width=0.3, label="Test top-1 accuracy", color="#c05621")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Score")
    ax1.set_title("Feature ablation: marginal contribution per group (test split)")
    ax1.legend()
    ax1.grid(alpha=0.3, axis="y")
    for i, (f, t) in enumerate(zip(f1s, top1s)):
        ax1.annotate(f"{f:.3f}", (i - 0.15, f), ha="center", va="bottom", fontsize=8)
        ax1.annotate(f"{t:.3f}", (i + 0.15, t), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out_path = DOCS_DIR / "phase4_ablation.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
