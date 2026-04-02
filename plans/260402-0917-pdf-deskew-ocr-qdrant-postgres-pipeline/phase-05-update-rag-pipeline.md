# Phase 5: Update RAG Pipeline

## Context
- Current: `src/rag/retriever.py` uses MongoDB `$vectorSearch` + keyword regex
- Need: Replace with Qdrant vector search + PostgreSQL metadata filtering

## Overview
- **Priority**: High
- **Status**: Pending
- **Description**: Rewrite HybridRetriever to use Qdrant for vectors and PostgreSQL for metadata-aware filtering

## Requirements
### Functional
- Vector search via Qdrant client
- Keyword fallback search via PostgreSQL full-text or content field
- Filter by date range and document type (using PostgreSQL metadata + Qdrant payload filters)
- Maintain same `hybrid_search()` and `build_context()` API for RAG pipeline compatibility

### Non-functional
- Drop MongoDB dependency from retriever
- Keep `RAGPipeline` interface unchanged — only internal retriever changes

## Related Code Files
- **Modify**: `src/rag/retriever.py` — rewrite to use Qdrant + PostgreSQL
- **Verify**: `src/rag/pipeline.py` — ensure no breaking changes
- **Verify**: `src/api/routes.py` — ensure API still works

## Implementation Steps

1. Rewrite `src/rag/retriever.py`:
   ```python
   class HybridRetriever:
       def __init__(self):
           self.qdrant = get_client()  # from src.storage.qdrant_vector_store
           self.embedder = SentenceTransformer(model_name)
           self.pg_conn_str = os.getenv("DATABASE_URL")

       def vector_search(self, query, k=10, doc_type=None):
           query_embedding = self.embedder.encode(query).tolist()
           results = search_vectors(self.qdrant, query_embedding, k, doc_type)
           return [{"content": r.payload["content"],
                     "metadata": r.payload,
                     "score": r.score} for r in results.points]

       def keyword_search(self, query, k=10):
           # Search Qdrant payload content via scroll + filter
           # Or use PostgreSQL content_preview for keyword matching
           ...

       def hybrid_search(self, query, k=5, ...):
           # Same merge logic, different backends
           ...

       def build_context(self, results, max_tokens=1500):
           # Unchanged logic
           ...

       def is_healthy(self):
           # Check Qdrant + PostgreSQL connectivity
           ...
   ```

2. Update imports in `src/rag/pipeline.py` if needed (should be transparent)

3. Update `src/api/routes.py` if it directly references MongoDB

## Todo
- [ ] Rewrite `HybridRetriever` with Qdrant backend
- [ ] Keep `hybrid_search()` and `build_context()` API compatible
- [ ] Add date/type filtering parameters to search methods
- [ ] Update health check to verify Qdrant + PostgreSQL
- [ ] Verify RAGPipeline still works end-to-end

## Success Criteria
- `RAGPipeline.query()` works without changes
- Vector search returns relevant Vietnamese document chunks
- Metadata filtering by date range and type works
- Health check reports Qdrant + PostgreSQL status

## Risk Assessment
- **API breakage**: Keep same method signatures → RAGPipeline and API routes unchanged
- **Keyword search**: Qdrant payload search is less flexible than MongoDB regex → consider PostgreSQL `ILIKE` on content_preview as fallback
