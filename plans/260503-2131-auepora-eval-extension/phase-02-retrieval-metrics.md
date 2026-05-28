# Phase 02 — Retrieval Ranking Metrics

## Context Links
- Paper 2405.07437v2.pdf Section 3.3 — accuracy metrics: MAP@K, MRR@K, Hit@K
- No existing retrieval-only metrics in repo

## Overview
- **Priority**: P2
- **Status**: complete
- **Description**: Pure-Python ranking metrics for retriever evaluation. No LLM, no network.

## Key Insights
- Functions take `retrieved_ids: List[str]` (ranked) + `relevant_ids: Set[str]` (ground truth) + `k: int`.
- Stateless, deterministic — easy to unit test.
- Must agree with standard IR definitions (sanity-check against textbook formulas).

## Requirements
- `precision_at_k(retrieved, relevant, k) -> float`
- `recall_at_k(retrieved, relevant, k) -> float`
- `hit_at_k(retrieved, relevant, k) -> float` — 1.0 if any hit in top-k else 0.0
- `mrr(retrieved, relevant) -> float` — 1/rank of first hit, 0 if none
- `average_precision_at_k(retrieved, relevant, k) -> float` — AP for one query
- `map_at_k(list_of_retrieved, list_of_relevant, k) -> float` — mean over queries
- Edge cases: empty retrieved, empty relevant, k > len(retrieved)

## Architecture
```
tests/evaluation/retrieval-metrics.py   (single file, < 100 lines)
├── precision_at_k
├── recall_at_k
├── hit_at_k
├── mrr
├── average_precision_at_k
└── map_at_k
```

## Related Code Files
**Create:** `tests/evaluation/retrieval-metrics.py`

**Future (not this phase):** integration into `run-evaluation.py` requires retriever to expose chunk IDs and dataset to provide `relevant_chunk_ids` field — flagged as unresolved Q in plan.md.

## Implementation Steps
1. Create `retrieval-metrics.py` with module docstring citing Auepora.
2. Implement each metric using set operations and rank scanning.
3. All functions handle empty inputs by returning 0.0 (document this).
4. Add type hints; use `List[str]` and `Set[str]`.
5. No external deps beyond stdlib.

## Todo List
- [x] Create retrieval-metrics.py
- [x] precision_at_k + recall_at_k
- [x] hit_at_k + mrr
- [x] average_precision_at_k + map_at_k
- [x] Edge case handling (empty inputs)

## Success Criteria
- All 6 functions importable
- File < 100 lines
- No imports beyond `typing` and stdlib
- Phase 5 unit tests pass

## Risk Assessment
- Off-by-one in MAP formula — mitigated by Phase 5 unit tests with known reference values.

## Next Steps
- Phase 5 unit tests verify correctness.
- Future integration: requires `relevant_chunk_ids` field in eval dataset (out of scope).
