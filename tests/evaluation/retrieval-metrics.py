"""Retrieval ranking metrics for RAG evaluation.

Implements IR metrics from Auepora framework (2405.07437v2, Section 3.3):
  - precision_at_k, recall_at_k, hit_at_k
  - mrr, average_precision_at_k, map_at_k

All functions return 0.0 on empty inputs (documented per-function).
"""

from typing import List, Set


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Fraction of top-k retrieved items that are relevant. Returns 0.0 if k<=0 or retrieved empty."""
    if not retrieved or k <= 0:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for doc in top_k if doc in relevant) / k


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Fraction of relevant items found in top-k. Returns 0.0 if relevant empty."""
    if not relevant or not retrieved or k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / len(relevant)


def hit_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """1.0 if any relevant item appears in top-k, else 0.0."""
    if not retrieved or not relevant or k <= 0:
        return 0.0
    return 1.0 if any(doc in relevant for doc in retrieved[:k]) else 0.0


def mrr(retrieved: List[str], relevant: Set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant item, 0.0 if no hit."""
    if not retrieved or not relevant:
        return 0.0
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Average Precision@K for a single query. 0.0 if relevant empty."""
    if not relevant or not retrieved or k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = 0
    precision_sum = 0.0
    for i, doc in enumerate(top_k, start=1):
        if doc in relevant:
            hits += 1
            precision_sum += hits / i
    if hits == 0:
        return 0.0
    return precision_sum / min(len(relevant), k)


def map_at_k(
    list_of_retrieved: List[List[str]],
    list_of_relevant: List[Set[str]],
    k: int,
) -> float:
    """Mean Average Precision@K over multiple queries. 0.0 if no queries."""
    if not list_of_retrieved:
        return 0.0
    ap_scores = [
        average_precision_at_k(ret, rel, k)
        for ret, rel in zip(list_of_retrieved, list_of_relevant)
    ]
    return sum(ap_scores) / len(ap_scores)
