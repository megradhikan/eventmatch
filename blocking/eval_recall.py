"""
PRD 6.2 headline metric: what fraction of the time does the true match
survive into the top-K blocking candidates, as a function of K? Run
before any ML touches the pipeline -- this bounds the model's achievable
recall downstream.
"""
import csv
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from retrieve import get_client, retrieve_candidates  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
DOCS_DIR = Path(__file__).parent.parent / "docs"
K_VALUES = [1, 3, 5, 10, 20]
MAX_K = max(K_VALUES)


def load_positive_listings():
    with (DATA_DIR / "listings.csv").open() as f:
        listings = {r["listing_id"]: r for r in csv.DictReader(f)}
    with (DATA_DIR / "ground_truth.csv").open() as f:
        gt = list(csv.DictReader(f))
    positives = [(listings[g["listing_id"]], g["true_event_id"]) for g in gt if g["true_event_id"]]
    return positives


def main():
    es = get_client()
    positives = load_positive_listings()
    print(f"Evaluating blocking recall over {len(positives)} positive listings")

    hits_at_k = {k: 0 for k in K_VALUES}
    total = 0
    rank_not_found = 0
    per_query_latency_ms = []

    for listing, true_event_id in positives:
        t0 = time.perf_counter()
        candidates = retrieve_candidates(
            es, listing["raw_title"], listing["raw_date"], listing["raw_venue"], size=MAX_K
        )
        per_query_latency_ms.append((time.perf_counter() - t0) * 1000)

        cand_ids = [c["event_id"] for c in candidates]
        total += 1
        if true_event_id in cand_ids:
            rank = cand_ids.index(true_event_id) + 1
            for k in K_VALUES:
                if rank <= k:
                    hits_at_k[k] += 1
        else:
            rank_not_found += 1

    recall_at_k = {k: hits_at_k[k] / total for k in K_VALUES}

    print("\nRecall@K (retrieval-only latency measured at K=20 query size):")
    for k in K_VALUES:
        print(f"  recall@{k:>2} = {recall_at_k[k]:.4f}")
    print(f"  true match never retrieved (even at K={MAX_K}): {rank_not_found}/{total} "
          f"({rank_not_found/total:.4f})")

    latency_stats = {
        "p50_ms": statistics.median(per_query_latency_ms),
        "p95_ms": statistics.quantiles(per_query_latency_ms, n=20)[18],
        "p99_ms": statistics.quantiles(per_query_latency_ms, n=100)[98],
        "mean_ms": statistics.mean(per_query_latency_ms),
    }
    print(f"\nRetrieval latency (size={MAX_K} query, includes network round-trip):")
    for k, v in latency_stats.items():
        print(f"  {k} = {v:.2f}")

    # Separate benchmark: does query latency itself vary meaningfully with
    # requested candidate-set size K? (downstream feature-extraction cost
    # scales with K regardless; this isolates the ES-side cost.)
    print("\nBenchmarking ES query latency at each K (200-listing sample)...")
    sample = positives[:200] if len(positives) > 200 else positives
    latency_by_k = {}
    for k in K_VALUES:
        times = []
        for listing, _ in sample:
            t0 = time.perf_counter()
            retrieve_candidates(es, listing["raw_title"], listing["raw_date"], listing["raw_venue"], size=k)
            times.append((time.perf_counter() - t0) * 1000)
        latency_by_k[k] = {"p50_ms": statistics.median(times), "mean_ms": statistics.mean(times)}
        print(f"  K={k:>2}: p50={latency_by_k[k]['p50_ms']:.2f}ms mean={latency_by_k[k]['mean_ms']:.2f}ms")

    results = {
        "n_positive_listings": total,
        "recall_at_k": recall_at_k,
        "never_retrieved": rank_not_found,
        "overall_latency_ms": latency_stats,
        "latency_by_k_ms": latency_by_k,
    }
    out_path = DOCS_DIR / "phase2_blocking_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
