"""Renders the recall@K vs K curve (PRD 8: headline blocking tradeoff)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

DOCS_DIR = Path(__file__).parent.parent / "docs"


def main():
    results = json.loads((DOCS_DIR / "phase2_blocking_results.json").read_text())
    recall = results["recall_at_k"]
    ks = sorted(int(k) for k in recall.keys())
    recalls = [recall[str(k)] for k in ks]
    latency = results["latency_by_k_ms"]
    p50s = [latency[str(k)]["p50_ms"] for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(ks, recalls, marker="o", color="#2b6cb0")
    ax1.set_xlabel("K (candidates retrieved)")
    ax1.set_ylabel("Recall@K")
    ax1.set_title("Blocking recall@K")
    ax1.set_ylim(0, 1.02)
    ax1.grid(alpha=0.3)
    for k, r in zip(ks, recalls):
        ax1.annotate(f"{r:.3f}", (k, r), textcoords="offset points", xytext=(0, 8), fontsize=8)

    ax2.plot(ks, p50s, marker="s", color="#c05621")
    ax2.set_xlabel("K (candidates retrieved)")
    ax2.set_ylabel("p50 query latency (ms)")
    ax2.set_title("ES retrieval latency vs K")
    ax2.grid(alpha=0.3)

    fig.suptitle(f"Candidate blocking tradeoff (n={results['n_positive_listings']} positive listings)")
    fig.tight_layout()
    out_path = DOCS_DIR / "phase2_recall_curve.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
