# Phase 01 — Generation Metrics (Correctness, Noise Robustness, Negative Rejection)

## Context Links
- Paper: `2405.07437v2.pdf` Section 3.4 (generation targets)
- Existing: `tests/evaluation/ragas_evaluator.py`

## Overview
- **Priority**: P2
- **Status**: complete
- **Description**: Add 3 LLM-as-judge metrics to existing evaluator. Update `RagasEvaluator.evaluate_sample()` and aggregate.

## Key Insights
- Correctness ≠ Faithfulness: correctness compares answer ↔ ground truth (semantic), faithfulness compares answer ↔ contexts.
- Negative rejection: ground truth in dataset is `"__REJECT__"` or `null` — system should reply with refusal phrase. Score 1.0 if refusal detected.
- Noise robustness needs sample with `noisy_contexts` field (Phase 3 builds dataset).

## Requirements
### Functional
- `score_correctness(client, question, answer, ground_truth) -> float`
- `score_noise_robustness(client, question, answer, ground_truth, noisy_contexts) -> float` — judges if answer remained correct despite noise
- `score_negative_rejection(client, question, answer) -> float` — 1.0 if answer is a refusal ("không có thông tin", "tôi không biết", etc.); 0.0 if it hallucinates an answer
- `RagasEvaluator.evaluate_sample()` adds `correctness` to output dict (always)
- Robustness/rejection scored only when sample has flag (`sample.get("eval_type")` in `{"noise","negative"}`)

### Non-functional
- Vietnamese prompts where appropriate (system answers in Vietnamese)
- Reuse `_score_prompt()` infra
- Keep file under 200 lines; split if needed (see plan.md)

## Architecture
```
ragas_evaluator.py
├── _score_prompt() (existing)
├── score_faithfulness/answer_relevancy/context_* (existing)
├── score_correctness()           NEW
├── score_noise_robustness()      NEW
├── score_negative_rejection()    NEW (heuristic + LLM hybrid)
└── RagasEvaluator
    └── evaluate_sample()         UPDATED — branch on eval_type
```

If split needed:
- `ragas-generation-metrics.py` ← faithfulness, answer_relevancy, correctness, noise_robustness, negative_rejection
- `ragas-retrieval-metrics.py` ← context_precision, context_recall
- `ragas_evaluator.py` ← `_score_prompt`, `_make_client`, `RagasEvaluator` orchestrator only

## Related Code Files
**Modify:**
- `tests/evaluation/ragas_evaluator.py`
- `tests/evaluation/run-evaluation.py` (pass new fields through)
- `tests/evaluation/generate-eval-report.py` (render new metrics)

**Maybe create (if size > 200 lines):**
- `tests/evaluation/ragas-generation-metrics.py`
- `tests/evaluation/ragas-retrieval-metrics.py`

## Implementation Steps
1. Add `score_correctness()` — prompt: compare ANSWER vs GROUND TRUTH semantically, score 0-1.
2. Add `score_negative_rejection()` — quick regex check for Vietnamese refusal phrases (`không có thông tin|tôi không biết|không đủ dữ liệu`); fallback to LLM judge: "Did the assistant correctly refuse to answer because info is unavailable? 1.0 yes / 0.0 no/hallucinated".
3. Add `score_noise_robustness()` — prompt includes noisy_contexts + ground_truth, asks LLM whether answer still matches GT despite noise.
4. Update `evaluate_sample()`: always compute correctness; if `eval_type == "noise"` compute noise score; if `eval_type == "negative"` compute rejection score (skip faithfulness/correctness).
5. Update aggregation in `evaluate()` to include new metrics (handle `None` for samples that didn't run a metric).
6. Update `generate-eval-report.py` to display new aggregate rows.
7. Check line count — if `ragas_evaluator.py` > 200, split per Architecture.

## Todo List
- [x] Add score_correctness with Vietnamese-aware prompt
- [x] Add score_negative_rejection (regex + LLM fallback)
- [x] Add score_noise_robustness
- [x] Update evaluate_sample branching by eval_type
- [x] Update evaluate() aggregation (skip None)
- [x] Update generate-eval-report.py rendering
- [x] Verify file < 200 lines, split if needed

## Success Criteria
- 3 new functions return float in [0,1] for valid inputs
- evaluate_sample() handles all 3 sample types: standard, noise, negative
- Aggregate report includes correctness for all standard samples; noise + rejection only when applicable
- Line counts < 200 per file

## Risk Assessment
- **LLM cost**: 3 extra judge calls per sample → ~75% more API spend. Mitigation: only correctness runs on every sample; noise/rejection only on tagged samples (~10 total).
- **Vietnamese refusal regex** may miss phrasings → LLM fallback covers it.

## Security
- No new secrets; reuses GEMINI_API_KEY.

## Next Steps
- Phase 3 (dataset) must land before phase-1 noise/rejection scoring usable.
