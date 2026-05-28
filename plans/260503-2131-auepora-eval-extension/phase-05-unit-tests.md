# Phase 05 — Unit Tests for Retrieval Metrics

## Context Links
- Depends on: Phase 02 (`tests/evaluation/retrieval-metrics.py`)
- Standard IR formulas (Manning et al., textbook MAP/MRR definitions)

## Overview
- **Priority**: P2
- **Status**: complete
- **Description**: Pure-Python unit tests asserting correctness of all 6 ranking metrics with known reference values.

## Key Insights
- No LLM, no network, no fixtures — fast deterministic tests.
- Use stdlib `unittest` (consistent with repo) or `pytest` if already used. Check repo first.

## Requirements
- File: `tests/evaluation/test-metrics-unit.py`
- ≥ 6 test cases per metric (happy path + edge cases)
- Reference values hand-computed from textbook examples
- Edge cases: empty retrieved, empty relevant, no overlap, all overlap, k > len

## Architecture
```
test-metrics-unit.py
├── TestPrecisionAtK
├── TestRecallAtK
├── TestHitAtK
├── TestMRR
├── TestAveragePrecision
└── TestMAP
```

## Related Code Files
**Create:** `tests/evaluation/test-metrics-unit.py`

## Implementation Steps
1. Detect framework: check repo for `pytest` config; default to `unittest` if none.
2. For each metric, write:
   - 1 happy-path case (mid-range value)
   - All-hits case → expected = 1.0
   - No-hits case → expected = 0.0
   - Empty inputs → 0.0
   - k > len(retrieved) → graceful behavior
   - 1-2 textbook reference cases
3. Hand-compute MAP example: retrieved `[a,b,c,d]`, relevant `{a,c}`, k=4 → AP = (1/1 + 2/3)/2 = 0.833.
4. Hand-compute MRR example: retrieved `[x,y,a]`, relevant `{a}` → 1/3.
5. Run with `.claude/skills/.venv/bin/python3 -m pytest tests/evaluation/test-metrics-unit.py` (or `python -m unittest`).

## Todo List
- [x] Pick test framework (unittest vs pytest)
- [x] Test class per metric
- [x] Hand-computed reference values documented in comments
- [x] Edge cases covered
- [x] All tests pass locally

## Success Criteria
- ≥ 36 assertions total (6 metrics × 6 cases)
- All pass on first commit
- File runnable via `python -m pytest` or `python -m unittest`
- Runs in < 1 second (pure math)

## Risk Assessment
- Wrong reference value → test masks bug. Mitigation: cross-check 2 textbook examples + symmetry properties (MAP with all relevant = 1.0).

## Next Steps
- After passing, future phase can integrate retrieval-metrics into `run-evaluation.py` (requires `relevant_chunk_ids` in dataset — out of scope here).
