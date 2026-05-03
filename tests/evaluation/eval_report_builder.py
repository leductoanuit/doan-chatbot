"""Helper functions for building eval summary reports and computing latency stats."""

import statistics
from datetime import datetime
from pathlib import Path


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

    return {
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(samples),
        "metrics": eval_result["aggregate"],
        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "worst_context_recall": worst_recall,
        "latency": eval_result.get("latency", {}),
        "judge_model": eval_result.get("judge_model", ""),
    }
