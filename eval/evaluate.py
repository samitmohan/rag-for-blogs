"""
Retrieval evaluation pipeline.

Loads the FAISS index, runs golden set queries through retrieve(),
and computes MRR@k, Recall@k, Precision@k.

Usage: uv run python -m eval.evaluate
"""

import json
import logging
import os
import sys

from config import FAISS_DIR
from embeddings.vector_store import VectorStore
from retrieval.retrieve import retrieve

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
GOLDEN_SET_PATH = os.path.join(EVAL_DIR, "golden_set.json")
RESULTS_PATH = os.path.join(EVAL_DIR, "results.json")


def load_golden_set():
    with open(GOLDEN_SET_PATH) as f:
        return json.load(f)


def is_relevant(chunk, entry):
    """Check if a retrieved chunk is relevant to the golden set entry."""
    post_title = chunk["metadata"].get("post_title", "")
    section = chunk["metadata"].get("section", "")

    # Match on post title
    title_match = any(
        rp.lower() in post_title.lower()
        for rp in entry["relevant_posts"]
    )
    if not title_match:
        return False

    # If sections are specified, also check section match
    if entry.get("relevant_sections"):
        section_match = any(
            rs.lower() in section.lower() or section.lower() in rs.lower()
            for rs in entry["relevant_sections"]
        )
        return section_match

    return True


def evaluate(k_values=(1, 3, 5), save_results=True):
    # Load index
    store = VectorStore.load(FAISS_DIR)
    golden = load_golden_set()

    max_k = max(k_values)
    per_query_results = []

    for entry in golden:
        query = entry["query"]
        results = retrieve(query, store, k=max_k)
        chunks = [r["chunk"] for r in results]

        # Check relevance of each result
        relevance = [is_relevant(c, entry) for c in chunks]

        # Find rank of first relevant result (1-indexed)
        first_relevant_rank = None
        for i, rel in enumerate(relevance):
            if rel:
                first_relevant_rank = i + 1
                break

        per_query_results.append({
            "query": query,
            "first_relevant_rank": first_relevant_rank,
            "relevance": relevance,
            "retrieved_titles": [c["metadata"].get("post_title") for c in chunks],
            "retrieved_sections": [c["metadata"].get("section") for c in chunks],
        })

    # Compute metrics for each k
    metrics = {}
    for k in k_values:
        mrr_sum = 0
        recall_hits = 0
        precision_sum = 0

        for result in per_query_results:
            rank = result["first_relevant_rank"]
            relevance_at_k = result["relevance"][:k]

            # MRR@k: reciprocal rank if first relevant is within top k
            if rank is not None and rank <= k:
                mrr_sum += 1.0 / rank
            # else: contributes 0

            # Recall@k: did we get at least one relevant result in top k?
            if any(relevance_at_k):
                recall_hits += 1

            # Precision@k: fraction of top k that are relevant
            precision_sum += sum(relevance_at_k) / k

        n = len(per_query_results)
        metrics[k] = {
            "MRR": mrr_sum / n,
            "Recall": recall_hits / n,
            "Precision": precision_sum / n,
        }

    # Print results
    print()
    print("=" * 60)
    print(f"  Retrieval Evaluation  |  {len(golden)} queries")
    print("=" * 60)
    print(f"{'Metric':<15} {'@1':>8} {'@3':>8} {'@5':>8}")
    print("-" * 41)
    for metric_name in ["MRR", "Recall", "Precision"]:
        vals = [metrics[k][metric_name] for k in k_values]
        print(f"{metric_name:<15} {vals[0]:>8.3f} {vals[1]:>8.3f} {vals[2]:>8.3f}")
    print("-" * 41)
    print()

    # Per-query breakdown
    print("Per-query breakdown:")
    print(f"{'#':<4} {'Hit?':>5}  Query")
    print("-" * 60)
    for i, result in enumerate(per_query_results):
        rank = result["first_relevant_rank"]
        hit = f"@{rank}" if rank else "MISS"
        query_short = result["query"][:50]
        print(f"{i+1:<4} {hit:>5}  {query_short}")
    print()

    # Save results
    if save_results:
        output = {
            "metrics": {str(k): v for k, v in metrics.items()},
            "per_query": per_query_results,
        }
        with open(RESULTS_PATH, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {RESULTS_PATH}")

    return metrics


if __name__ == "__main__":
    evaluate()
