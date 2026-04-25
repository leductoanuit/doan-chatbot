# Phase 02 — Setup RAGAS Pipeline

## Overview
- **Priority**: P1
- **Status**: Pending
- **Effort**: 2h
- **Blocked by**: Phase 01

Implement RAGAS evaluation pipeline: load dataset → chạy RAG pipeline → score từng metric.

## Key Insights

**4 RAGAS metrics cần đo:**

| Metric | Đo gì | Cần gì |
|--------|-------|--------|
| **Faithfulness** | Answer có bịa thông tin ngoài context không? | question, answer, contexts |
| **Answer Relevancy** | Answer có trả lời đúng câu hỏi không? | question, answer |
| **Context Precision** | Các chunks retrieved có liên quan không? | question, contexts, ground_truth |
| **Context Recall** | Chunks retrieved có đủ để trả lời không? | contexts, ground_truth |

**RAGAS dùng LLM judge** → tận dụng Gemini API đã có sẵn (không tốn thêm)

**Thư viện:** `ragas` PyPI package — cần add vào `requirements.txt`

## Requirements

- `ragas>=0.2` compatible với `langchain` hoặc standalone
- Dùng Gemini làm LLM judge (đã có `GEMINI_API_KEY`)
- Output: JSON + markdown report
- Chạy được offline (chỉ cần Gemini API, không cần internet khác)

## Architecture

```
eval-dataset.json
      │
      ▼
run-evaluation.py
      │
      ├─→ RAGPipeline.query(question) → {answer, contexts}
      │
      ├─→ ragas-evaluator.py
      │       ├── assemble Dataset (question, answer, contexts, ground_truth)
      │       └── ragas.evaluate() → scores
      │
      └─→ results/eval-report-{timestamp}.json
          results/eval-summary-{timestamp}.md
```

## Related Code Files

- **Create**: `tests/evaluation/ragas-evaluator.py` — RAGAS wrapper
- **Create**: `tests/evaluation/run-evaluation.py` — entry point
- **Modify**: `requirements.txt` — add `ragas`, `langchain-google-genai`
- **Read**: `src/rag/pipeline.py` — import RAGPipeline
- **Read**: `tests/evaluation/eval-dataset.json` — input dataset

## Implementation Steps

1. **Add dependencies** vào `requirements.txt`:
   ```
   ragas>=0.2.0
   langchain-google-genai>=2.0.0
   ```

2. **Tạo `tests/evaluation/ragas-evaluator.py`**:
   ```python
   from ragas import evaluate
   from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
   from ragas.llms import LangchainLLMWrapper
   from langchain_google_genai import ChatGoogleGenerativeAI

   class RagasEvaluator:
       def __init__(self):
           llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=...)
           self.llm = LangchainLLMWrapper(llm)
           self.metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

       def evaluate(self, samples: list[dict]) -> dict:
           # samples: [{question, answer, contexts, ground_truth}]
           dataset = Dataset.from_list(samples)
           result = evaluate(dataset, metrics=self.metrics, llm=self.llm)
           return result
   ```

3. **Tạo `tests/evaluation/run-evaluation.py`**:
   - Load `eval-dataset.json`
   - Khởi tạo `RAGPipeline` (cần Qdrant + PostgreSQL running)
   - Với mỗi câu hỏi: gọi `pipeline.query(q)` → lấy `answer` và `contexts`
   - Assemble RAGAS sample: `{question, answer, contexts, ground_truth}`
   - Gọi `RagasEvaluator.evaluate(samples)`
   - Lưu kết quả vào `results/`

4. **Xử lý edge case**:
   - Timeout handling khi gọi Gemini judge
   - Fallback nếu RAGAS fails trên 1 sample → skip và log
   - Progress bar (tqdm) vì 30 câu × 2 LLM calls = chậm

## Todo List

- [ ] Add `ragas`, `langchain-google-genai` vào `requirements.txt`
- [ ] Tạo `tests/evaluation/ragas-evaluator.py`
- [ ] Tạo `tests/evaluation/run-evaluation.py`
- [ ] Test chạy thử 3-5 câu trước khi chạy full 30 câu
- [ ] Verify output format JSON hợp lệ

## Success Criteria

- `run-evaluation.py` chạy không lỗi với 5 câu test
- Output `results/eval-report-*.json` có đủ 4 metric scores
- Không crash khi Gemini judge timeout (có retry logic)

## Risk Assessment

- **RAGAS API thay đổi**: RAGAS v0.1 vs v0.2 có breaking changes → pin version, đọc changelog
- **Rate limit Gemini**: 30 câu × 4 metrics = nhiều LLM calls → thêm `time.sleep(1)` giữa calls
- **Qdrant not running**: pipeline.query() sẽ fail → document trong README cách start services

## Security Considerations

- Không hardcode `GEMINI_API_KEY` — đọc từ `.env`
- Không commit `results/` nếu chứa data nhạy cảm

## Next Steps

→ Phase 03: chạy full evaluation + phân tích kết quả + viết báo cáo
