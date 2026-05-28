# Phase 02 — Multi-Query Generation (RAG Fusion)

**Priority:** High  
**Status:** ⬜ Todo  
**Effort:** M (Medium)  
**Depends on:** Phase 01 (`src/rag/query_processor.py` đã tồn tại)

## Context Links

- Retriever: `src/rag/retriever.py` — `hybrid_search()`
- Pipeline: `src/rag/pipeline.py` — `query()`
- Query Processor: `src/rag/query_processor.py` ← thêm function vào đây

## Vấn đề

Single-query retrieval có coverage thấp — cùng một câu hỏi có thể được diễn đạt nhiều cách, và các chunks liên quan có thể không match với một cách diễn đạt nhất định:

```
Query gốc:   "học phí hệ từ xa bao nhiêu?"
Variant 1:   "mức học phí chương trình đào tạo từ xa UIT"
Variant 2:   "chi phí học tập hệ từ xa mỗi tín chỉ"
```

Mỗi variant có thể retrieve chunks khác nhau → merge lại tăng coverage.

## Giải pháp: RAG Fusion

```
Original query
     ↓
generate_query_variants(n=2)  ← Gemini tạo 2 variant
     ↓
[query_0, query_1, query_2]   ← original + 2 variants
     ↓
hybrid_search × 3 (parallel)
     ↓
Reciprocal Rank Fusion (RRF) merge + deduplicate
     ↓
reranker (top k)
```

**Tại sao RRF thay vì score merge đơn giản?**
- Scores từ các queries khác nhau không so sánh được trực tiếp
- RRF chỉ dùng rank (vị trí), không dùng score → ổn định hơn
- Formula: `RRF_score(d) = Σ 1/(k + rank_i(d))` với k=60

## Related Code Files

**Cập nhật:**
- `src/rag/query_processor.py` — thêm `generate_query_variants()` và `reciprocal_rank_fusion()`
- `src/rag/pipeline.py` — thêm multi-query path trong `query()`

## Implementation Steps

### 1. Thêm vào `src/rag/query_processor.py`

```python
def generate_query_variants(
    query: str,
    llm_client,
    n: int = 2,
) -> List[str]:
    """Generate n alternative phrasings of the query via LLM.
    
    Returns list of variants (not including original).
    Falls back to empty list on error.
    """
    prompt = (
        f"Viết {n} cách diễn đạt khác nhau cho câu hỏi sau, "
        f"giữ nguyên ý nghĩa, mỗi câu một dòng, không đánh số:\n\n{query}"
    )
    try:
        raw = llm_client.generate(query=prompt, temperature=0.5, max_tokens=200)
        variants = [line.strip() for line in raw.strip().splitlines() if line.strip()]
        return variants[:n]
    except Exception:
        return []


def reciprocal_rank_fusion(
    results_lists: List[List[Dict]],
    k: int = 60,
) -> List[Dict]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.
    
    Args:
        results_lists: Each inner list is a ranked result from one query.
        k: RRF constant (default 60, standard value from literature).
    
    Returns:
        Deduplicated list sorted by RRF score descending.
    """
    scores: Dict[int, float] = {}
    chunks: Dict[int, Dict] = {}

    for results in results_lists:
        for rank, chunk in enumerate(results):
            key = hash(chunk["content"][:100])
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in chunks:
                chunks[key] = chunk

    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    merged = []
    for key in sorted_keys:
        chunk = chunks[key].copy()
        chunk["rrf_score"] = scores[key]
        chunk["final_score"] = scores[key]  # override for downstream compatibility
        merged.append(chunk)

    return merged
```

### 2. Cập nhật `src/rag/pipeline.py`

Thêm feature flag `use_multi_query` (mặc định `False` để không phá vỡ hiện tại):

```python
from src.rag.query_processor import (
    rewrite_standalone_query,
    generate_query_variants,
    reciprocal_rank_fusion,
)

def query(
    self,
    question: str,
    history=None,
    top_k: int = 10,
    use_multi_query: bool = False,   # NEW flag
) -> Dict:
    # 1. Rewrite (Phase 01)
    retrieval_query = rewrite_standalone_query(question, history or [], self.llm)
    expanded_query = self._expand_query(retrieval_query)

    # 2. Retrieval
    if use_multi_query:
        variants = generate_query_variants(expanded_query, self.llm, n=2)
        all_queries = [expanded_query] + variants

        # Retrieve for each query (reduced k to control total candidates)
        per_query_k = max(top_k, 10)
        results_lists = [
            self.retriever.hybrid_search(q, k=per_query_k)
            for q in all_queries
        ]
        candidates = reciprocal_rank_fusion(results_lists)
    else:
        candidates = self.retriever.hybrid_search(
            expanded_query, k=top_k, reranker=None
        )

    # 3. Rerank merged candidates
    results = self.reranker.rerank(question, candidates, top_k=top_k)

    # 4–5. Build context + generate (unchanged)
    ...
```

**Lưu ý:** Reranker nhận `question` gốc (không expanded) để cross-encoder score sát thực hơn với ý định user.

## Todo

- [ ] Thêm `generate_query_variants()` vào `src/rag/query_processor.py`
- [ ] Thêm `reciprocal_rank_fusion()` vào `src/rag/query_processor.py`
- [ ] Cập nhật `src/rag/pipeline.py` với `use_multi_query` flag
- [ ] Test thủ công: câu hỏi về học phí, tuyển sinh → kiểm tra variants có hợp lý không
- [ ] So sánh kết quả retrieval với/không có multi-query

## Success Criteria

- `use_multi_query=True` → retrieval trả về chunks đa dạng hơn so với single-query
- Variants được generate là các cách diễn đạt khác nhau nhưng cùng nghĩa
- Không có duplicate chunks trong kết quả cuối (RRF dedup hoạt động đúng)
- Latency tăng chấp nhận được (2 extra Gemini calls nhỏ + 2 extra searches)

## Risk Assessment

- **Risk:** Variants lạc đề → retrieve chunks không liên quan
  - Mitigation: reranker cuối lọc lại, chunks lạc đề sẽ bị drop
- **Risk:** Tốn API call mỗi query
  - Mitigation: `use_multi_query=False` mặc định; chỉ bật khi demo/đánh giá
- **Risk:** Tổng candidates quá lớn → reranker chậm
  - Mitigation: giới hạn `per_query_k` và RRF tự nhiên đẩy noise xuống thấp
