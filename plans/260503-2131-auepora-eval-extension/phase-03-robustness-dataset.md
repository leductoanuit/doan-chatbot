# Phase 03 — Robustness Dataset

## Context Links
- Paper Section 4 — Noise Robustness, Negative Rejection, Counterfactual Robustness
- Existing dataset: `tests/evaluation/eval-dataset.json`

## Overview
- **Priority**: P2
- **Status**: complete
- **Description**: Build supplementary dataset to test robustness behaviors not covered by main Q&A set.

## Key Insights
- Domain = university admissions/training regulations (CITD). Out-of-domain questions: cooking, weather, celebrity gossip, unrelated tech.
- Noise samples reuse legitimate questions but inject 1-2 irrelevant chunks among the relevant ones.
- Use existing chunks from `data/processed/all_documents_final.json` as noise sources for realism.

## Requirements
- File: `tests/evaluation/eval-dataset-robustness.json`
- ≥ 5 noise samples: schema `{question, ground_truth, reference_contexts, noisy_contexts, eval_type: "noise", category, difficulty}`
- ≥ 5 negative-rejection samples: `{question, ground_truth: "__REJECT__", eval_type: "negative", category: "out_of_domain", difficulty}`
- Vietnamese for question text (matching production language)

## Architecture
Single JSON array file, mirroring `eval-dataset.json` schema with extra `eval_type` discriminator.

## Related Code Files
**Create:** `tests/evaluation/eval-dataset-robustness.json`

**Modify (consumers):**
- `tests/evaluation/run-evaluation.py` — accept `--dataset` flag or auto-merge robustness file
- `tests/evaluation/ragas_evaluator.py` — branch on `eval_type` (Phase 1)

## Implementation Steps
1. Pick 5 existing questions from `eval-dataset.json` as bases for noise samples.
2. For each, pick 1-2 unrelated chunks from `all_documents_final.json` (different category) and append to `reference_contexts` as `noisy_contexts`.
3. Author 5+ out-of-domain questions in Vietnamese (e.g., "Cách nấu phở bò?", "Thời tiết Hà Nội hôm nay?", "Đội tuyển Việt Nam đá với đội nào?").
4. Set `ground_truth = "__REJECT__"` and `eval_type = "negative"`.
5. Validate JSON parses; ensure no duplicate questions vs main dataset.

## Todo List
- [x] Identify 5 base questions for noise samples
- [x] Source noisy chunks from corpus
- [x] Author 5 out-of-domain Vietnamese questions
- [x] Compose JSON file with consistent schema
- [x] Validate with `python -m json.tool`

## Success Criteria
- File parses as valid JSON list
- ≥ 10 samples total
- Schema fields match consumer code expectations
- Vietnamese text in questions

## Risk Assessment
- "Noisy" chunk too topically close → may not actually distract retriever. Mitigation: pick chunks from clearly different categories (e.g., regulations chunks injected into admission questions).

## Next Steps
- Phase 1 reads this dataset for noise/rejection scoring.
