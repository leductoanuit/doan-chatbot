"""Helper functions for building eval summary reports and computing latency stats."""

import importlib.util
import statistics
from datetime import datetime
from pathlib import Path

# Load retrieval-metrics module (hyphen in filename requires importlib)
_spec = importlib.util.spec_from_file_location(
    "retrieval_metrics", Path(__file__).parent / "retrieval-metrics.py"
)
_ret_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ret_mod)


def _token_set(text: str) -> set:
    """Lowercase word tokens from text."""
    return set(text.lower().split())


def _is_relevant(chunk_text: str, reference_contexts: list[str], recall_threshold: float = 0.4) -> bool:
    """True if chunk contains >= recall_threshold of reference tokens.

    Uses recall-only (not F1) because retrieved chunks are much longer than
    reference summaries — F1 would penalise large chunks unfairly.
    """
    chunk_tokens = _token_set(chunk_text)
    if not chunk_tokens:
        return False
    for ref in reference_contexts:
        ref_tokens = _token_set(ref)
        if not ref_tokens:
            continue
        recall = len(chunk_tokens & ref_tokens) / len(ref_tokens)
        if recall >= recall_threshold:
            return True
    return False


def compute_retrieval_rank_metrics(samples: list[dict], k: int = 10) -> dict:
    """Compute MAP@K, MRR, Hit@K for all samples using token-overlap relevance judgment.

    Each retrieved chunk is marked relevant if it contains >= 40% of reference
    context tokens (recall-based, not F1). Returns aggregate scores and per-category breakdown.
    """
    list_retrieved: list[list[str]] = []
    list_relevant: list[set] = []
    hit_scores: list[float] = []
    mrr_scores: list[float] = []

    for s in samples:
        contexts: list[str] = s.get("contexts", [])
        refs: list[str] = s.get("reference_contexts", [])

        # Build ordered list of chunk IDs (index strings) for rank metrics
        retrieved_ids = [str(i) for i in range(len(contexts))]
        relevant_ids = {
            str(i) for i, c in enumerate(contexts)
            if _is_relevant(c, refs)
        }

        list_retrieved.append(retrieved_ids)
        list_relevant.append(relevant_ids)
        hit_scores.append(_ret_mod.hit_at_k(retrieved_ids, relevant_ids, k))
        mrr_scores.append(_ret_mod.mrr(retrieved_ids, relevant_ids))

    map_score = _ret_mod.map_at_k(list_retrieved, list_relevant, k)
    mrr_avg = round(sum(mrr_scores) / len(mrr_scores), 4) if mrr_scores else 0.0
    hit_avg = round(sum(hit_scores) / len(hit_scores), 4) if hit_scores else 0.0

    # Per-category breakdown
    categories = sorted(set(s.get("category", "") for s in samples))
    by_category: dict = {}
    for cat in categories:
        cat_samples = [s for s in samples if s.get("category") == cat]
        cat_indices = [i for i, s in enumerate(samples) if s.get("category") == cat]
        if not cat_samples:
            continue
        cat_retrieved = [list_retrieved[i] for i in cat_indices]
        cat_relevant = [list_relevant[i] for i in cat_indices]
        by_category[cat] = {
            "map_at_k": round(_ret_mod.map_at_k(cat_retrieved, cat_relevant, k), 4),
            "mrr": round(sum(_ret_mod.mrr(r, rel) for r, rel in zip(cat_retrieved, cat_relevant)) / len(cat_samples), 4),
            "hit_at_k": round(sum(_ret_mod.hit_at_k(r, rel, k) for r, rel in zip(cat_retrieved, cat_relevant)) / len(cat_samples), 4),
        }

    return {
        f"map_at_{k}": round(map_score, 4),
        "mrr": mrr_avg,
        f"hit_at_{k}": hit_avg,
        "k": k,
        "by_category": by_category,
    }


def compute_latency_stats(samples: list[dict]) -> dict:
    """Compute p50/p90/p99 and per-category average latency from sample list."""
    latencies = [s["latency_ms"] for s in samples if s.get("latency_ms") is not None]
    if not latencies:
        return {}

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    def percentile(p: float) -> float:
        idx = int(p / 100 * n)
        return round(sorted_lat[min(idx, n - 1)], 2)

    categories = set(s.get("category", "") for s in samples)
    per_category = {}
    for cat in sorted(categories):
        cat_lats = [s["latency_ms"] for s in samples if s.get("category") == cat and s.get("latency_ms") is not None]
        per_category[cat] = round(statistics.mean(cat_lats), 2) if cat_lats else 0.0

    return {
        "p50_ms": percentile(50),
        "p90_ms": percentile(90),
        "p99_ms": percentile(99),
        "avg_ms": round(statistics.mean(latencies), 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "per_category_avg_ms": per_category,
    }


def build_report(eval_result: dict) -> dict:
    """Aggregate eval scores by category and difficulty into a summary dict."""
    samples = eval_result["samples"]
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    def avg_scores(subset):
        if not subset:
            return {}
        valid = {m: [s["scores"][m] for s in subset if s["scores"].get(m) is not None] for m in metrics}
        return {m: round(sum(v) / len(v), 4) for m, v in valid.items() if v}

    categories = set(s["category"] for s in samples)
    by_category = {cat: avg_scores([s for s in samples if s["category"] == cat]) for cat in sorted(categories)}

    difficulties = set(s["difficulty"] for s in samples)
    by_difficulty = {diff: avg_scores([s for s in samples if s["difficulty"] == diff]) for diff in sorted(difficulties)}

    sorted_by_recall = sorted(samples, key=lambda s: s["scores"].get("context_recall") or 1)
    worst_recall = [
        {
            "question": s["question"],
            "category": s["category"],
            "context_recall": s["scores"].get("context_recall"),
            "faithfulness": s["scores"].get("faithfulness"),
        }
        for s in sorted_by_recall[:5]
        if s["scores"].get("context_recall") is not None
    ]

    rank_metrics = compute_retrieval_rank_metrics(samples)

    return {
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(samples),
        "metrics": eval_result["aggregate"],
        "retrieval_rank_metrics": rank_metrics,
        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "worst_context_recall": worst_recall,
        "latency": eval_result.get("latency", {}),
        "judge_model": eval_result.get("judge_model", ""),
    }
