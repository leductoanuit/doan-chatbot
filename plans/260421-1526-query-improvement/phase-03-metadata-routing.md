# Phase 03 — Metadata-Based Query Routing

**Priority:** Medium  
**Status:** ⬜ Todo  
**Effort:** S (Small)

## Context Links

- Pipeline: `src/rag/pipeline.py` — `query()`
- Retriever: `src/rag/retriever.py` — `hybrid_search()` (đã có `doc_type`, `system_type` params)
- Qdrant store: `src/storage/qdrant_vector_store.py`
- Query Processor: `src/rag/query_processor.py`

## Vấn đề

`hybrid_search()` đã hỗ trợ `doc_type` và `system_type` filter, nhưng `pipeline.py` không bao giờ truyền chúng → mọi query đều search toàn bộ corpus (247 chunks).

Ví dụ query "học phí hệ từ xa" sẽ retrieve cả tài liệu về chứng chỉ CNTT, thông báo khai giảng, v.v. — noise không cần thiết.

**Metadata hiện có trong Qdrant payload:**

| Field | Giá trị ví dụ |
|-------|--------------|
| `system_type` | `"tuyển sinh"`, `"đào tạo"`, `"chứng chỉ"` |
| `document_type` | `"quyết định"`, `"thông tư"`, `"faq"` |

## Giải pháp: Rule-Based Query Classifier

Classify query → ánh xạ sang `system_type` filter → thu hẹp search space.

**Không dùng LLM** cho bước này (không đáng tốn API call, rule-based đủ chính xác với domain hẹp).

```
Query: "điều kiện xét tuyển hệ từ xa?"
  → classify → system_type = "tuyển sinh"
  → hybrid_search(..., system_type="tuyển sinh")
  → chỉ search trong ~60 chunks tuyển sinh thay vì 247
```

## Related Code Files

**Cập nhật:**
- `src/rag/query_processor.py` — thêm `classify_query_intent()`
- `src/rag/pipeline.py` — thêm routing logic trong `query()`

## Implementation Steps

### 1. Thêm `classify_query_intent()` vào `src/rag/query_processor.py`

```python
# Mapping: keyword signals → system_type filter value
_INTENT_RULES: List[tuple[List[str], str]] = [
    (
        ["tuyển sinh", "xét tuyển", "điều kiện vào", "hồ sơ đăng ký",
         "nộp hồ sơ", "thời hạn đăng ký", "kết quả trúng tuyển", "học bổng"],
        "tuyển sinh",
    ),
    (
        ["học phí", "chi phí", "tiền học", "đóng tiền", "học bổng học phí",
         "miễn giảm học phí", "hoàn học phí"],
        "tuyển sinh",  # học phí thường nằm trong docs tuyển sinh
    ),
    (
        ["chứng chỉ", "cntt", "ứng dụng cntt", "chứng chỉ tin học",
         "thi chứng chỉ", "cấp chứng chỉ"],
        "chứng chỉ",
    ),
    (
        ["quy chế", "điều khoản", "quy định", "kỷ luật", "học vụ",
         "bảo lưu", "thôi học", "cảnh báo học vụ", "tốt nghiệp"],
        "đào tạo",
    ),
]


def classify_query_intent(query: str) -> Optional[str]:
    """Return system_type filter string or None if intent is ambiguous.
    
    Returns None when query spans multiple intents or is unclear,
    to avoid over-filtering.
    """
    q = query.lower()
    matched: set[str] = set()

    for keywords, system_type in _INTENT_RULES:
        if any(kw in q for kw in keywords):
            matched.add(system_type)

    # Only filter when exactly one intent matched — avoid false precision
    return matched.pop() if len(matched) == 1 else None
```

### 2. Cập nhật `src/rag/pipeline.py`

```python
from src.rag.query_processor import classify_query_intent

def query(self, question: str, history=None, top_k: int = 10, ...) -> Dict:
    # ... (existing rewrite + expand) ...

    # Metadata routing
    system_type_filter = classify_query_intent(expanded_query)

    results = self.retriever.hybrid_search(
        expanded_query,
        k=top_k,
        reranker=self.reranker,
        system_type=system_type_filter,   # None = no filter (unchanged behavior)
    )
    ...
```

**Fallback quan trọng:** Nếu filter trả về ít hơn 3 results → retry không có filter:

```python
results = self.retriever.hybrid_search(
    expanded_query, k=top_k, reranker=self.reranker,
    system_type=system_type_filter,
)
if len(results) < 3 and system_type_filter:
    # Filtered search too narrow — fall back to full corpus
    results = self.retriever.hybrid_search(
        expanded_query, k=top_k, reranker=self.reranker,
    )
```

## Todo

- [ ] Kiểm tra giá trị `system_type` thực tế trong Qdrant (query payload để xác nhận exact string values)
- [ ] Thêm `classify_query_intent()` vào `src/rag/query_processor.py`
- [ ] Cập nhật `src/rag/pipeline.py` với routing + fallback logic
- [ ] Test: query "điều kiện tuyển sinh" → confirm chỉ search docs tuyển sinh
- [ ] Test: query chung "giới thiệu UIT" → confirm `None` filter (no restriction)

## Kiểm tra system_type values thực tế

Trước khi implement, chạy:

```python
# Kiểm tra unique system_type values trong Qdrant
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)
points, _ = client.scroll("documents", limit=300, with_payload=True)
types = set(p.payload.get("system_type") for p in points)
print(types)
```

## Success Criteria

- Query về tuyển sinh → `system_type="tuyển sinh"` filter applied
- Query mơ hồ → `None` filter (full corpus search, safe fallback)
- Số candidates giảm → reranker nhanh hơn
- Retrieval quality không giảm (hoặc tăng) so với unfiltered

## Risk Assessment

- **Risk:** `system_type` string trong Qdrant không khớp với rule trong code
  - Mitigation: kiểm tra actual values trước khi implement (bước đầu tiên trong Todo)
- **Risk:** Over-filtering → bỏ sót chunks quan trọng
  - Mitigation: fallback khi `len(results) < 3`; chỉ filter khi single intent matched
