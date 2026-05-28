# Phase 04 — Testing & Evaluation

**Priority:** Medium  
**Status:** ⬜ Todo  
**Effort:** S (Small)  
**Depends on:** Phase 01, 02, 03

## Context Links

- Pipeline: `src/rag/pipeline.py`
- Query Processor: `src/rag/query_processor.py` (new)
- Test queries: tự định nghĩa dưới đây

## Mục tiêu

Xác nhận các cải tiến hoạt động đúng và không gây regression. Đây là đánh giá **định tính thủ công** (không yêu cầu RAGAS framework).

## Test Cases

### Phase 01 — Conversation-Aware Rewriting

| Turn | Query | Expected behavior |
|------|-------|-------------------|
| 1 | "Điều kiện tuyển sinh hệ từ xa là gì?" | Normal retrieval |
| 2 | "Học phí của nó bao nhiêu?" | Rewrite → "Học phí hệ đào tạo từ xa UIT bao nhiêu?" |
| 3 | "Còn thời hạn nộp hồ sơ?" | Rewrite → "Thời hạn nộp hồ sơ tuyển sinh hệ đào tạo từ xa UIT?" |

**Kiểm tra `detect_context_dependency()`:**
```python
assert detect_context_dependency("học phí của nó") == True
assert detect_context_dependency("còn điều kiện tốt nghiệp?") == True
assert detect_context_dependency("điều kiện tuyển sinh hệ từ xa") == False
assert detect_context_dependency("học phí hệ từ xa bao nhiêu?") == False
```

### Phase 02 — Multi-Query Generation

Kiểm tra thủ công:
- Query: `"học phí hệ từ xa"` với `use_multi_query=True`
- Verify: 2 variants được generate, có nghĩa, khác cách diễn đạt
- Verify: RRF merge không có duplicate chunks
- So sánh: top-5 results với và không có multi-query

### Phase 03 — Metadata Routing

**Kiểm tra `classify_query_intent()`:**
```python
assert classify_query_intent("điều kiện tuyển sinh") == "tuyển sinh"
assert classify_query_intent("học phí hệ từ xa") == "tuyển sinh"
assert classify_query_intent("quy chế học vụ") == "đào tạo"
assert classify_query_intent("chứng chỉ ứng dụng cntt") == "chứng chỉ"
assert classify_query_intent("giới thiệu về UIT") is None      # ambiguous → no filter
assert classify_query_intent("so sánh chương trình TTNT và CNTT") is None  # multi-intent
```

### Regression Tests

Các câu hỏi hiện tại đang hoạt động tốt — xác nhận vẫn đúng sau khi thêm phases:

| Query | Expected source |
|-------|----------------|
| "lúc nào tuyển sinh hệ từ xa?" | FAQ tuyển sinh hoặc thông báo khai giảng |
| "507-QĐ quy định gì về bảo lưu?" | 507-QĐ-2024 |
| "học phí mỗi tín chỉ bao nhiêu?" | FAQ hoặc thông báo học phí |
| "chương trình TTNT gồm những môn gì?" | CTĐT TTNT |

## Implementation Steps

### 1. Script test thủ công

Tạo `scripts/test-query-improvements.py` để chạy nhanh các test cases:

```python
"""Manual test script for query improvement phases."""
import sys
sys.path.insert(0, ".")

from src.rag.query_processor import detect_context_dependency, classify_query_intent

# Unit tests
def test_context_dependency():
    assert detect_context_dependency("học phí của nó") == True
    assert detect_context_dependency("còn thời hạn nộp?") == True
    assert detect_context_dependency("điều kiện tuyển sinh hệ từ xa") == False
    print("✓ detect_context_dependency")

def test_intent_classifier():
    assert classify_query_intent("điều kiện tuyển sinh") == "tuyển sinh"
    assert classify_query_intent("quy chế học vụ") == "đào tạo"
    assert classify_query_intent("giới thiệu UIT") is None
    print("✓ classify_query_intent")

if __name__ == "__main__":
    test_context_dependency()
    test_intent_classifier()
    print("\nAll unit tests passed.")
```

### 2. End-to-end smoke test

Sau khi chạy toàn bộ pipeline, test 3 câu hỏi đại diện và log ra:
- `retrieval_query` (sau rewrite)
- `expanded_query` (sau expand)
- `system_type_filter` (metadata routing result)
- `sources` top-3 (file name + score)

## Todo

- [ ] Chạy unit tests cho `detect_context_dependency()` và `classify_query_intent()`
- [ ] Tạo `scripts/test-query-improvements.py`
- [ ] Chạy end-to-end với 5 test queries, log kết quả
- [ ] Xác nhận không có regression trên 4 câu hỏi baseline
- [ ] Ghi chú kết quả quan sát vào section dưới

## Kết quả quan sát (điền sau khi test)

```
Phase 01 rewrite:
- "học phí của nó" → [điền kết quả]

Phase 03 routing:
- "điều kiện tuyển sinh" → filter: [điền], results count: [điền]
- "giới thiệu UIT" → filter: None, results count: [điền]

Regression:
- "507-QĐ bảo lưu" → top source: [điền], score: [điền]
```

## Success Criteria

- Tất cả unit tests pass
- Không có regression trên 4 baseline queries
- Phase 01: ít nhất 2/3 test conversations rewrite đúng
- Phase 03: filter đúng cho các query rõ ràng, None cho query mơ hồ
