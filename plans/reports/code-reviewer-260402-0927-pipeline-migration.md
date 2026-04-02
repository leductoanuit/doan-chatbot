# Code Review: PDF Deskew + OCR + Qdrant + PostgreSQL Pipeline Migration

**Date:** 2026-04-02  
**Reviewer:** code-reviewer  
**Score: 7.5 / 10**

---

## Scope

- **Files reviewed:** 10 (7 Python, 1 Pydantic schema, 1 routes, 1 Colab notebook)
- **LOC:** ~650 new/modified
- **Focus:** Qdrant migration, PostgreSQL metadata, OCR preprocessing, API compatibility

## Overall Assessment

Solid migration from MongoDB to Qdrant. Clean separation of concerns, good error handling in most paths, parameterized SQL queries (no injection risk). A few issues need attention before merge.

---

## Critical Issues

### C1. File naming violation: `postgres-metadata.py` uses hyphen

- **File:** `src/storage/postgres-metadata.py`
- **Problem:** Python cannot import modules with hyphens. `from src.storage.postgres-metadata import ...` is a syntax error.
- **Fix:** Rename to `src/storage/postgres_metadata.py` (underscore).
- **Impact:** Currently the file is never imported anywhere (orphaned code). Once someone tries to import it, it will fail.

### C2. `postgres-metadata.py` is never integrated

- **File:** `src/storage/postgres-metadata.py`
- **Problem:** No file imports from it. `ingest_pipeline.py` stores vectors in Qdrant but never writes document metadata to PostgreSQL. The pipeline is incomplete.
- **Fix:** Add `insert_document()` + `insert_chunks_batch()` calls in `ingest_pipeline.py` after chunking, before/after Qdrant upsert.

### C3. Connection leak in `postgres-metadata.py`

- **File:** `src/storage/postgres-metadata.py`, all functions
- **Problem:** `psycopg2.connect()` returns a connection. Using `with conn` only calls `conn.__exit__` which commits/rollbacks but does NOT close the connection. Each function call leaks a connection.
- **Fix:** Either use `conn.close()` explicitly in a `finally` block, or wrap in a helper:
  ```python
  from contextlib import contextmanager

  @contextmanager
  def get_connection():
      conn = psycopg2.connect(DATABASE_URL)
      try:
          yield conn
      finally:
          conn.close()
  ```

---

## High Priority

### H1. MD5 hash collision risk in `_make_point_id`

- **File:** `src/storage/qdrant_vector_store.py:74-78`
- **Problem:** `int(md5(...).hexdigest(), 16) % (2**63)` maps 128-bit hash to 63-bit int. With large document sets, birthday paradox makes collisions plausible (~50% at ~3 billion points). Collision = silent data overwrite.
- **Fix:** Use UUID directly — Qdrant supports string IDs:
  ```python
  import uuid
  def _make_point_id(document_id, page, chunk_idx):
      return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}_{page}_{chunk_idx}"))
  ```

### H2. Duplicate embedding model instantiation in retriever

- **File:** `src/rag/retriever.py:26`
- **Problem:** `HybridRetriever.__init__` loads `SentenceTransformer(embedding_model)`. But `ingest_pipeline.py` uses `VietnameseEmbedder` (same underlying model). If both are used in the same process, the model is loaded twice (~500MB RAM wasted).
- **Suggestion:** Consider sharing a single model instance or using `VietnameseEmbedder` in both places for consistency.

### H3. Keyword search swallows all exceptions silently

- **File:** `src/rag/retriever.py:97-98`
- **Problem:** `except Exception: continue` silently swallows errors including connection failures, timeouts, and programming bugs. If Qdrant is unreachable, keyword_search returns `[]` with no logging.
- **Fix:** Add `logging.warning(...)` in the except block at minimum.

### H4. `requirements.txt` still has MongoDB deps

- **File:** `requirements.txt:20-21, 28-29`
- **Problem:** `pymongo`, `motor`, `llama-index-vector-stores-mongodb` are still listed. If migration is complete, these should be removed to avoid confusion and reduce install size.
- **Note:** `src/embedding/mongo_setup.py` also still exists. Decide: keep for rollback or remove.

---

## Medium Priority

### M1. No `content` text index in Qdrant for keyword search

- **File:** `src/storage/qdrant_vector_store.py:61-66`
- **Problem:** `init_collection` creates keyword indexes on `document_type`, `system_type`, `document_id`, but not a text index on `content`. The retriever's `keyword_search` uses `MatchText` on `content`, which requires a full-text index for performance.
- **Fix:** Add text index:
  ```python
  client.create_payload_index(
      collection_name=QDRANT_COLLECTION,
      field_name="content",
      field_schema=PayloadSchemaType.TEXT,
  )
  ```

### M2. Deduplication in `hybrid_search` uses content hash — fragile

- **File:** `src/rag/retriever.py:124`
- **Problem:** `hash(r["content"][:100])` depends on Python's built-in `hash()` which is not stable across processes (randomized by default via `PYTHONHASHSEED`). Two runs could deduplicate differently. Also, first-100-chars may collide for documents with identical headers.
- **Fix:** Use a deterministic hash or compare on `(source, page, chunk_idx)` tuple if available.

### M3. OCR timeout of 120s per page may be too aggressive for large pages

- **File:** `src/scraper/pdf_extractor.py:81`
- **Problem:** DeepSeek VL2 on T4 GPU takes 2-5s/page per notebook docs, but complex pages with tables/dense text could exceed this. A 50-page PDF would take up to 100 minutes serially.
- **Suggestion:** Consider async/parallel page processing or at least a configurable timeout via env var.

### M4. `preprocess_for_ocr` uses `print()` for logging

- **Files:** `src/scraper/image_preprocessor.py:109`, `src/scraper/pdf_extractor.py:85`, `src/storage/qdrant_vector_store.py` (multiple)
- **Problem:** Inconsistent with `routes.py` which uses `logging.getLogger()`. Mixed print/logging makes production log management harder.
- **Fix:** Use `logger = logging.getLogger(__name__)` throughout.

---

## Low Priority

### L1. `cv2` imported inside functions (lazy import)

- **File:** `src/scraper/image_preprocessor.py:18, 76`
- **Acceptable:** This is intentional to avoid import errors when opencv is not installed (e.g., in test env). Fine as-is.

### L2. Notebook has hardcoded `NGROK_TOKEN = "YOUR_NGROK_TOKEN"`

- **File:** `notebooks/deepseek-ocr-service.ipynb`, Cell 2
- **Risk:** Low. Placeholder value. But worth adding a check:
  ```python
  assert NGROK_TOKEN != "YOUR_NGROK_TOKEN", "Replace with your ngrok token"
  ```

### L3. `query_documents` builds WHERE clause with f-string

- **File:** `src/storage/postgres-metadata.py:170`
- **Assessment:** Safe. The f-string only interpolates column names and operators that are hardcoded, not user input. Parameters use `%(name)s` placeholders. No SQL injection risk.

---

## API Compatibility

**RAGPipeline interface is preserved.** Verified:
- `pipeline.py` calls `retriever.hybrid_search(query, k=top_k)` — method exists with same signature
- `retriever.build_context(results)` — method exists with same signature
- `HealthResponse` schema changed `mongodb: bool` to `qdrant: bool` — this IS a breaking change for any frontend/client parsing the health response JSON. Ensure frontend is updated.

---

## Positive Observations

- Clean fallback logic in `process_pdf`: text_native -> OCR when < 100 chars
- Deskew implementation is well-engineered: skips trivial angles, handles empty images
- Qdrant integration follows best practices (batch upsert, payload indexes, cosine distance)
- Parameterized SQL throughout PostgreSQL module
- Good use of `ON CONFLICT DO NOTHING` for idempotent inserts
- `preprocess_for_ocr` wraps everything in try/catch with fallback to raw bytes

---

## Recommended Actions (Priority Order)

1. **[CRITICAL]** Rename `postgres-metadata.py` to `postgres_metadata.py`
2. **[CRITICAL]** Fix connection leak — make `get_connection()` a context manager that closes
3. **[CRITICAL]** Integrate PostgreSQL metadata writes into ingest pipeline
4. **[HIGH]** Switch `_make_point_id` to UUID5 to avoid hash collisions
5. **[HIGH]** Add text index on `content` field in Qdrant collection init
6. **[HIGH]** Add logging to keyword_search exception handler
7. **[MEDIUM]** Remove MongoDB deps from requirements.txt (or document why kept)
8. **[MEDIUM]** Standardize logging (replace print with logging module)
9. **[LOW]** Update frontend for `HealthResponse.qdrant` field rename

---

## Unresolved Questions

- Is `mongo_setup.py` being kept intentionally for rollback, or should it be removed?
- Should `postgres_metadata.py` use connection pooling (e.g., `psycopg2.pool`) for production?
- Is there a plan to add async support to the PostgreSQL layer for FastAPI compatibility?
- What is the expected document volume? Hash collision risk depends on scale.
