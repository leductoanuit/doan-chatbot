# Phase 04 — Latency Measurement

## Context Links
- Paper Section 4 — operational metric: latency
- Existing entry: `tests/evaluation/run-evaluation.py`

## Overview
- **Priority**: P3
- **Status**: complete
- **Description**: Measure per-query RAG latency, report p50/p90/p99 + per-category averages.

## Key Insights
- Pure timing — no LLM judge. `time.perf_counter()` around pipeline call.
- Use Python's `statistics.quantiles` for percentiles (stdlib, no deps).

## Requirements
- For each sample: capture `latency_ms` (float)
- Aggregate: `{p50, p90, p99, avg, min, max}` overall + `per_category: {tuyen_sinh: avg_ms, ...}`
- Include in JSON report and Markdown report

## Architecture
```
run-evaluation.py
├── for sample in dataset:
│   ├── t0 = perf_counter()
│   ├── result = pipeline.query(sample.question)
│   ├── sample.latency_ms = (perf_counter() - t0) * 1000
└── compute_latency_stats(samples) → dict
```

## Related Code Files
**Modify:**
- `tests/evaluation/run-evaluation.py` — add timing + stats fn
- `tests/evaluation/generate-eval-report.py` — render latency table

## Implementation Steps
1. In `run-evaluation.py` query loop, wrap pipeline call with `perf_counter()`.
2. Store `latency_ms` per sample in result dict.
3. Add `compute_latency_stats(samples)` helper using `statistics.quantiles(data, n=100)` for percentiles.
4. Group by `category` for per-category avg.
5. Add `"latency": {...}` block to JSON report.
6. Update `generate-eval-report.py` to render latency Markdown table.

## Todo List
- [x] Add per-query timing in run-evaluation.py
- [x] compute_latency_stats helper
- [x] Per-category aggregation
- [x] JSON report includes latency block
- [x] Markdown report includes latency table

## Success Criteria
- Report shows p50/p90/p99 in ms
- Per-category averages present
- No measurable overhead added to pipeline (timing only)

## Risk Assessment
- Cold-start first query inflates p99 — acceptable (real-world); optionally warm up with one discarded query (YAGNI: skip unless metric is unstable).

## Next Steps
- None; independent phase.
