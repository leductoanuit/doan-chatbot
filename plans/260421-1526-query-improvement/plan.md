# Plan: Cải thiện truy vấn cho chatbot UIT

**Created:** 2026-04-21  
**Status:** 🟢 Done  
**Branch:** main

## Mục tiêu

Cải thiện chất lượng retrieval của RAG pipeline bằng cách nâng cấp xử lý query — từ `_expand_query()` đơn giản hiện tại lên các kỹ thuật nâng cao hơn.

## Hiện trạng

```
User query → _expand_query() [rule-based] → hybrid_search(k=10) → reranker → build_context() → Gemini
```

**Vấn đề:**
1. Query expansion chỉ là rule-based normalization, không hiểu ngữ nghĩa
2. Không dùng history để cải thiện query retrieval (chỉ gửi vào LLM)
3. Single-query: 1 truy vấn → 1 tập kết quả (thiếu coverage)
4. Không có metadata routing (doc_type/system_type không được dùng trong pipeline.py)

## Các Phase

| Phase | Tên | Trạng thái | Effort |
|-------|-----|-----------|--------|
| 01 | [Conversation-Aware Query Rewriting](phase-01-conversation-query-rewriting.md) | ✅ Done | S |
| 02 | [Multi-Query Generation (RAG Fusion)](phase-02-multi-query-generation.md) | ✅ Done | M |
| 03 | [Metadata-Based Query Routing](phase-03-metadata-routing.md) | ✅ Done | S |
| 04 | [Testing & Evaluation](phase-04-testing-evaluation.md) | ⬜ Todo | S |

## Phụ thuộc

- Phase 01 độc lập (thêm module mới `query_processor.py`)
- Phase 02 phụ thuộc Phase 01 (dùng chung `query_processor.py`)
- Phase 03 độc lập (thêm routing logic vào `pipeline.py`)
- Phase 04 sau tất cả phases trên

## Files chính bị ảnh hưởng

- `src/rag/pipeline.py` — orchestration (cập nhật `query()`)
- `src/rag/retriever.py` — hybrid_search (thêm metadata filter path)
- `src/rag/query_processor.py` — **[NEW]** module xử lý query
