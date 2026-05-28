---
title: "Auepora Evaluation Extension"
description: "Extend RAG eval suite with correctness, retrieval ranking metrics, robustness, latency, and unit tests per Auepora framework"
status: complete
progress: 100%
priority: P2
effort: 8h
branch: main
tags: [evaluation, rag, ragas, auepora, testing]
created: 2026-05-03
completed: 2026-05-03
---

# Auepora Evaluation Extension Plan

Extends the existing RAG eval suite (`tests/evaluation/`) to cover the full Auepora framework from paper 2405.07437v2:
- Generation: add **correctness**, **noise robustness**, **negative rejection**
- Retrieval: add ranking metrics (**MAP@K, MRR, Hit@K, Recall@K, Precision@K**)
- Operational: add **latency** measurement
- Robustness: extra dataset (out-of-domain + noisy contexts)
- Quality: unit tests for pure-math metric functions

## Reference
- Paper: `/Users/cps/do an chatbot/2405.07437v2.pdf` (Auepora — Section 3)
- Existing code: `tests/evaluation/ragas_evaluator.py`, `run-evaluation.py`, `eval-dataset.json`

## Phases

| # | Phase | File | Effort | Status |
|---|-------|------|--------|--------|
| 1 | Generation metrics (correctness, noise robustness, negative rejection) | [phase-01-generation-metrics.md](phase-01-generation-metrics.md) | 2h | complete |
| 2 | Retrieval ranking metrics (pure math, no LLM) | [phase-02-retrieval-metrics.md](phase-02-retrieval-metrics.md) | 1.5h | complete |
| 3 | Robustness dataset (noise + out-of-domain) | [phase-03-robustness-dataset.md](phase-03-robustness-dataset.md) | 1.5h | complete |
| 4 | Latency measurement in `run-evaluation.py` | [phase-04-latency.md](phase-04-latency.md) | 1h | complete |
| 5 | Unit tests for retrieval metrics | [phase-05-unit-tests.md](phase-05-unit-tests.md) | 1h | complete |

## Dependencies
- Phase 1 → independent (extends `ragas_evaluator.py`)
- Phase 2 → independent (new file)
- Phase 3 → blocks Phase 1 robustness scoring runs
- Phase 4 → independent (`run-evaluation.py`)
- Phase 5 → depends on Phase 2

Execute order: 2 + 3 in parallel, then 1, then 5, then 4 (or 4 anytime).

## File Size Watch
`ragas_evaluator.py` currently 191 lines. Phase 1 adds ~3 score functions + judge prompts (~80 lines). If exceeds 200 lines, split into:
- `ragas-generation-metrics.py` (faithfulness, answer_relevancy, correctness, noise_robustness, negative_rejection)
- `ragas-retrieval-metrics.py` (context_precision, context_recall)
- `ragas_evaluator.py` keeps `RagasEvaluator` orchestrator class only.

## Success Criteria
- All new metric fns importable, return floats 0-1
- Retrieval metrics validated by unit tests (>= 6 test cases each)
- `run-evaluation.py` reports latency p50/p90/p99
- Robustness dataset has >= 5 noise + >= 5 negative-rejection samples
- No file exceeds 200 lines

## Unresolved Questions
- Does retriever expose stable chunk IDs for ranking metrics? (Phase 2 needs this — verify before implementation)
- For noise robustness, do we inject noisy chunks at retrieval time or pre-bake into dataset? (Plan assumes pre-baked in `eval-dataset-robustness.json`)
- Ground-truth relevant chunk IDs for MAP/MRR — must be added to existing `eval-dataset.json` or new field `relevant_chunk_ids`?
