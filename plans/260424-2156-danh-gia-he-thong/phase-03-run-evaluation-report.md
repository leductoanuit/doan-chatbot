# Phase 03 — Chạy Evaluation & Báo Cáo

## Overview
- **Priority**: P1
- **Status**: Pending
- **Effort**: 2h
- **Blocked by**: Phase 02

Chạy full evaluation 30 câu, phân tích kết quả, tạo báo cáo markdown cho đồ án.

## Key Insights

- RAGAS scores từ 0→1, ngưỡng tốt: ≥ 0.7 cho production RAG system
- Phân tích breakdown theo category để tìm điểm yếu cụ thể
- Báo cáo cần cả số liệu tổng + phân tích định tính (tại sao score thấp ở đâu)

## Requirements

- Chạy full 30 câu hỏi, không skip
- Output: JSON raw + markdown summary cho đồ án
- Phân tích theo category (tuyển sinh, học vụ, học phí, CTĐT, quy chế)
- So sánh trước/sau (nếu có baseline)

## Expected Output Format

### `results/eval-report-{timestamp}.json`
```json
{
  "timestamp": "2026-04-25T...",
  "total_questions": 30,
  "metrics": {
    "faithfulness": 0.82,
    "answer_relevancy": 0.79,
    "context_precision": 0.74,
    "context_recall": 0.71
  },
  "by_category": {
    "tuyen_sinh": { "faithfulness": 0.88, ... },
    "hoc_vu": { ... }
  },
  "per_question": [ ... ]
}
```

### `results/eval-summary-{timestamp}.md`
```markdown
# Kết quả đánh giá RAG Chatbot UIT

## Tổng quan
| Metric | Score | Đánh giá |
|--------|-------|----------|
| Faithfulness | 0.82 | ✅ Tốt |
| Answer Relevancy | 0.79 | ✅ Tốt |
| Context Precision | 0.74 | ⚠️ Trung bình |
| Context Recall | 0.71 | ⚠️ Trung bình |

## Phân tích theo category
...

## Câu hỏi có score thấp nhất
...

## Kết luận & Hướng cải thiện
...
```

## Related Code Files

- **Create**: `tests/evaluation/generate-eval-report.py` — tạo markdown summary từ JSON
- **Read**: `tests/evaluation/results/eval-report-*.json`
- **Create**: `tests/evaluation/results/eval-summary-*.md`

## Implementation Steps

1. **Chạy full evaluation**:
   ```bash
   cd "do an chatbot"
   python tests/evaluation/run-evaluation.py --output results/
   ```

2. **Tạo `tests/evaluation/generate-eval-report.py`**:
   - Load JSON result
   - Tính average per metric và per category
   - Identify top 5 câu hỏi score thấp nhất (context_recall thấp = retrieval miss)
   - Output markdown report

3. **Phân tích kết quả**:
   - **Faithfulness thấp** → Gemini hallucinate → cần tăng constraint trong system prompt
   - **Context Precision thấp** → retrieval lấy nhiều chunk không liên quan → tăng MIN_SCORE hoặc giảm top_k
   - **Context Recall thấp** → retrieval bỏ sót chunk quan trọng → kiểm tra chunking strategy
   - **Answer Relevancy thấp** → câu trả lời không focus → cải thiện system prompt

4. **Viết báo cáo đồ án** dựa trên số liệu thực tế

## Todo List

- [ ] Chạy `run-evaluation.py` full 30 câu
- [ ] Tạo `generate-eval-report.py`
- [ ] Phân tích kết quả theo category
- [ ] Identify câu hỏi score thấp và nguyên nhân
- [ ] Viết markdown summary
- [ ] Thêm kết quả vào `docs/qa-defense-preparation.md`

## Success Criteria

- Có file `results/eval-report-*.json` với đủ 30 câu
- Có file `results/eval-summary-*.md` với phân tích rõ ràng
- Tổng score ≥ 0.7 trên ít nhất 3/4 metrics

## Risk Assessment

- **Score thấp bất ngờ**: không sao — báo cáo cả điểm yếu và hướng cải thiện là đủ cho đồ án
- **Gemini judge bias**: LLM judge có thể không nhất quán → chạy 2 lần để verify nếu cần
- **Thời gian**: 30 câu × Gemini calls ≈ 5-10 phút — có thể dùng `--sample 10` để test nhanh

## Next Steps

- Cập nhật `docs/qa-defense-preparation.md` với số liệu thực tế
- Có thể dùng kết quả để justify các design decision trong đồ án
