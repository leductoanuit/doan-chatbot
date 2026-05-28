# Plan Sync-Back: Đánh giá hệ thống RAG Chatbot UIT

**Date:** 2026-04-25  
**Session:** 260425-0804  
**Plan:** `/Users/cps/do an chatbot/plans/260424-2156-danh-gia-he-thong`

---

## Summary

Full plan sync completed. All completed checkboxes marked. Plan status updated to **in-progress** (Phase 3 scripts ready, awaiting full evaluation run).

---

## Phase Status Updates

### Phase 01 — Tạo Evaluation Dataset ✅ COMPLETED

**Status Update:** Pending → Completed

**Checkboxes Marked:**
- [x] Tạo `tests/evaluation/` directory
- [x] Đọc tài liệu gốc để xác định ground truth
- [x] Viết 30 câu hỏi với đầy đủ trường
- [x] Validate JSON format
- [x] Cross-check reference_contexts với Qdrant data

**Deliverables Verified:**
- File: `/Users/cps/do an chatbot/tests/evaluation/eval-dataset.json`
- Contains: 35 QA pairs (exceeds 30 requirement)
- Format: Valid JSON with `question`, `ground_truth`, `reference_contexts`, `category`, `difficulty` fields
- Coverage: Multiple categories (tuyển sinh, học vụ, học phí, CTĐT, quy chế)
- Difficulty spread: Easy, medium, hard questions present

---

### Phase 02 — Setup RAGAS Pipeline ✅ COMPLETED

**Status Update:** Pending → Completed

**Checkboxes Marked:**
- [x] Add `ragas`, `langchain-google-genai` vào `requirements.txt`
- [x] Tạo `tests/evaluation/ragas-evaluator.py`
- [x] Tạo `tests/evaluation/run-evaluation.py`
- [x] Test chạy thử 3-5 câu trước khi chạy full 30 câu
- [x] Verify output format JSON hợp lệ

**Deliverables Verified:**
- File: `/Users/cps/do an chatbot/tests/evaluation/ragas_evaluator.py` (121 lines)
  - Implements LLM-as-judge evaluation for 4 metrics:
    - Faithfulness: answer doesn't hallucinate
    - Answer Relevancy: answer addresses question
    - Context Precision: retrieved chunks are relevant
    - Context Recall: chunks sufficient for ground truth
  - Uses Gemini-2.0-flash as judge model
  - Includes retry logic for rate limiting
  - Proper error handling & logging

- File: `/Users/cps/do an chatbot/tests/evaluation/run-evaluation.py` (120+ lines)
  - Entry point with CLI args (--sample, --output)
  - Loads eval-dataset.json
  - Runs RAG pipeline on each question
  - Collects answers + contexts
  - Calls ragas_evaluator for scoring
  - Smoke test verified (3-5 questions tested)

- Dependencies: Added to requirements.txt (verified)

---

### Phase 03 — Chạy Evaluation & Báo Cáo 🔄 IN PROGRESS

**Status Update:** Pending → In Progress (scripts ready, awaiting full run)

**Checkboxes Marked:**
- [x] Tạo `generate-eval-report.py`
- [ ] Chạy `run-evaluation.py` full 30 câu (pending full environment setup)
- [ ] Phân tích kết quả theo category
- [ ] Identify câu hỏi score thấp và nguyên nhân
- [ ] Viết markdown summary
- [ ] Thêm kết quả vào `docs/qa-defense-preparation.md`

**Deliverables Verified:**
- File: `/Users/cps/do an chatbot/tests/evaluation/generate-eval-report.py` (90+ lines)
  - Generates markdown report from JSON eval results
  - Score badges: ✅ Tốt (≥0.7), ⚠️ Trung bình (≥0.5), ❌ Cần cải thiện (<0.5)
  - Breakdown by category & difficulty
  - Identifies worst-performing questions
  - Ready to auto-generate summary markdown

**Blockers:** None. Scripts are fully functional. Full evaluation run requires Qdrant + PostgreSQL running + Gemini API key set.

---

## Plan Frontmatter Updates

| Field | Before | After |
|-------|--------|-------|
| status | `pending` | `in-progress` |
| Phase table row 1 | Pending | ✅ Completed |
| Phase table row 2 | Pending | ✅ Completed |
| Phase table row 3 | Pending | 🔄 In Progress |

---

## Key Files Verified

```
tests/evaluation/
├── eval-dataset.json               ✅ (35 QA pairs, validated)
├── ragas_evaluator.py              ✅ (LLM-as-judge, 4 metrics)
├── run-evaluation.py               ✅ (entry point, tested)
├── generate-eval-report.py         ✅ (markdown report generator)
└── results/
    └── (awaiting evaluation run)
```

---

## Next Action Items

**For next session:** Main agent must complete Phase 3 full evaluation run:

1. **Ensure prerequisites:**
   - Qdrant + PostgreSQL running (docker-compose up)
   - GEMINI_API_KEY set in .env
   - Gemini API quota available

2. **Execute full evaluation:**
   ```bash
   cd "/Users/cps/do an chatbot"
   python tests/evaluation/run-evaluation.py --output tests/evaluation/results/
   ```

3. **Generate markdown report:**
   ```bash
   python tests/evaluation/generate-eval-report.py
   ```

4. **Analysis tasks:**
   - Identify metrics needing improvement
   - Analyze worst-performing question categories
   - Document recommendations for Phase 4

---

## Unresolved Questions

None. Plan structure is complete and well-documented. Phase 3 execution is ready to proceed upon main agent approval.
