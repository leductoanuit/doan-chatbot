# Phase 01 — Conversation-Aware Query Rewriting

**Priority:** High  
**Status:** ⬜ Todo  
**Effort:** S (Small)

## Context Links

- Pipeline: `src/rag/pipeline.py`
- Retriever: `src/rag/retriever.py`
- LLM Client: `src/rag/llm_client.py`

## Vấn đề

Khi user hỏi tiếp theo trong hội thoại, query thường chứa đại từ hoặc tham chiếu mơ hồ:

```
Turn 1: "Điều kiện tuyển sinh hệ từ xa là gì?"
Turn 2: "Học phí của nó bao nhiêu?"  ← "nó" = hệ từ xa, nhưng retriever không biết
Turn 3: "Còn thời hạn nộp hồ sơ?"   ← thiếu context hoàn toàn
```

Hiện tại `history` chỉ được gửi vào Gemini để generate, **không dùng để cải thiện query retrieval**.

## Giải pháp

Thêm bước **standalone query rewriting** trước khi embed:
- Dùng Gemini (light call) để rewrite query thành câu hỏi độc lập (standalone)
- Chỉ activate khi: `len(history) > 0 AND query có dấu hiệu phụ thuộc context`

## Architecture

```
User query + history
       ↓
_rewrite_query_with_history()   ← NEW
       ↓
standalone_query
       ↓
_expand_query() [existing]
       ↓
hybrid_search + reranker
```

## Related Code Files

**Tạo mới:**
- `src/rag/query_processor.py` — chứa `rewrite_standalone_query()` và `detect_context_dependency()`

**Cập nhật:**
- `src/rag/pipeline.py` — thêm bước rewrite vào `query()` method

## Implementation Steps

### 1. Tạo `src/rag/query_processor.py`

```python
"""Query processing utilities — context rewriting and expansion."""

import re
from typing import List, Dict, Optional

# Từ/pattern báo hiệu query phụ thuộc vào context trước
_CONTEXT_DEPENDENCY_SIGNALS = [
    r"\b(nó|họ|đó|này|kia|vậy|thế)\b",       # đại từ chỉ định
    r"\b(còn|thêm|nữa|khác|tiếp theo)\b",     # continuity signals
    r"^(vậy|thế thì|thế còn|còn)\b",          # sentence starters
]

def detect_context_dependency(query: str) -> bool:
    """Return True if query likely depends on conversation context."""
    q = query.lower().strip()
    for pattern in _CONTEXT_DEPENDENCY_SIGNALS:
        if re.search(pattern, q):
            return True
    # Very short queries without domain keywords likely need context
    if len(q.split()) <= 4:
        domain_signals = ["học phí", "tuyển sinh", "quy chế", "môn học", "tín chỉ", "ngành"]
        if not any(kw in q for kw in domain_signals):
            return True
    return False


def rewrite_standalone_query(
    query: str,
    history: List[Dict],
    llm_client,
) -> str:
    """Rewrite query into a standalone version using conversation history.
    
    Only calls LLM when history suggests context dependency.
    Returns original query unchanged if no rewrite needed.
    """
    if not history or not detect_context_dependency(query):
        return query

    # Build minimal context from last 3 turns
    recent = history[-3:]
    history_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Bot'}: {m['content'][:200]}"
        for m in recent
    )

    prompt = (
        f"Lịch sử hội thoại:\n{history_text}\n\n"
        f"Câu hỏi mới: {query}\n\n"
        "Hãy viết lại câu hỏi trên thành một câu hỏi độc lập, đầy đủ context, "
        "không cần đọc lịch sử. Chỉ trả về câu hỏi đã viết lại, không giải thích."
    )

    try:
        rewritten = llm_client.generate(query=prompt, temperature=0.1, max_tokens=128)
        return rewritten.strip() if rewritten else query
    except Exception:
        return query
```

### 2. Cập nhật `src/rag/pipeline.py` — method `query()`

Thêm bước rewrite **trước** `_expand_query()`:

```python
from src.rag.query_processor import rewrite_standalone_query

def query(self, question: str, history=None, top_k: int = 10) -> Dict:
    # 1. Rewrite query if context-dependent
    retrieval_query = rewrite_standalone_query(question, history or [], self.llm)
    
    # 2. Expand (existing logic)
    expanded_query = self._expand_query(retrieval_query)
    
    # 3. Retrieve + rerank (unchanged)
    results = self.retriever.hybrid_search(expanded_query, k=top_k, reranker=self.reranker)
    ...
    # Generate still uses original `question` for natural response
    answer = self.llm.generate(query=question, context=context, history=history)
```

**Lưu ý quan trọng:** Generate vẫn dùng `question` gốc (không rewritten) để response tự nhiên hơn. Chỉ retrieval dùng rewritten query.

## Todo

- [ ] Tạo `src/rag/query_processor.py` với `detect_context_dependency()` và `rewrite_standalone_query()`
- [ ] Cập nhật `src/rag/pipeline.py` để gọi rewrite trước expand
- [ ] Kiểm tra không phá vỡ flow hiện tại khi `history=None`
- [ ] Test thủ công: hỏi tiếp theo về cùng chủ đề và kiểm tra retrieval có đúng hơn không

## Success Criteria

- Query có đại từ/tham chiếu được rewrite thành câu hỏi standalone rõ ràng
- Khi `history=None` hoặc query không phụ thuộc context → bỏ qua rewrite (không tốn API call)
- Latency tăng không quá 1s cho trường hợp cần rewrite

## Risk Assessment

- **Risk:** Gemini rewrite không chính xác → query bị sai hướng
  - Mitigation: fallback về query gốc nếu rewrite rỗng/lỗi
- **Risk:** Tốn thêm API call mỗi turn
  - Mitigation: chỉ gọi khi `detect_context_dependency() = True`
