# Phase 3: PostgreSQL Metadata Storage

## Context
- Current: Document metadata stored inside MongoDB documents alongside embeddings
- Need: Structured metadata in PostgreSQL for date-based filtering, document type queries
- Vietnamese legal documents have: issue date, effective date, document number, issuing body

## Overview
- **Priority**: High
- **Status**: Pending
- **Description**: Setup PostgreSQL schema for document metadata, create Python CRUD module

## Requirements
### Functional
- Store document metadata: title, issue_date, effective_date, document_number, source, doc_type
- Link metadata to Qdrant vectors via shared `document_id`
- Support queries: filter by date range, document type, issuing body
- Bulk insert from PDF processing pipeline

### Non-functional
- Use `psycopg2` (sync) for simplicity
- Indexes on date columns and document_type for fast filtering

## Architecture
```
PostgreSQL Schema:
┌─────────────────────────────────────┐
│ documents                           │
├─────────────────────────────────────┤
│ id           UUID PK (= doc_id)     │
│ title        TEXT                    │
│ document_number TEXT                 │
│ issue_date   DATE                   │
│ effective_date DATE                 │
│ issuing_body TEXT                   │
│ document_type TEXT                  │
│ source_file  TEXT                   │
│ source_url   TEXT                   │
│ system_type  TEXT (chinh-quy/tu-xa) │
│ created_at   TIMESTAMPTZ           │
│ updated_at   TIMESTAMPTZ           │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ chunks                              │
├─────────────────────────────────────┤
│ id           TEXT PK (chunk_id)     │
│ document_id  UUID FK → documents.id │
│ page_number  INT                    │
│ chunk_index  INT                    │
│ content_preview TEXT (first 200ch)  │
│ token_count  INT                    │
│ created_at   TIMESTAMPTZ           │
└─────────────────────────────────────┘

Shared ID: chunk_id = "{doc_id}_{page}_{chunk_idx}"
→ Same ID used as Qdrant point ID
```

## Related Code Files
- **Create**: `src/storage/postgres-metadata.py` — PostgreSQL CRUD module
- **Modify**: `src/embedding/ingest_pipeline.py` — insert metadata during ingestion
- **Modify**: `requirements.txt` — add `psycopg2-binary`

## Implementation Steps

1. Add `psycopg2-binary` to `requirements.txt`

2. Create `src/storage/__init__.py`

3. Create `src/storage/postgres-metadata.py`:
   ```python
   import os, uuid
   from datetime import date
   import psycopg2
   from psycopg2.extras import execute_values

   DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/chatbot")

   def get_connection():
       return psycopg2.connect(DATABASE_URL)

   def init_schema():
       """Create tables if not exist."""
       with get_connection() as conn:
           with conn.cursor() as cur:
               cur.execute("""
                   CREATE TABLE IF NOT EXISTS documents (
                       id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                       title TEXT,
                       document_number TEXT,
                       issue_date DATE,
                       effective_date DATE,
                       issuing_body TEXT,
                       document_type TEXT,
                       source_file TEXT,
                       source_url TEXT,
                       system_type TEXT,
                       created_at TIMESTAMPTZ DEFAULT NOW(),
                       updated_at TIMESTAMPTZ DEFAULT NOW()
                   );
                   CREATE TABLE IF NOT EXISTS chunks (
                       id TEXT PRIMARY KEY,
                       document_id UUID REFERENCES documents(id),
                       page_number INT,
                       chunk_index INT,
                       content_preview TEXT,
                       token_count INT,
                       created_at TIMESTAMPTZ DEFAULT NOW()
                   );
                   CREATE INDEX IF NOT EXISTS idx_docs_issue_date ON documents(issue_date);
                   CREATE INDEX IF NOT EXISTS idx_docs_type ON documents(document_type);
                   CREATE INDEX IF NOT EXISTS idx_docs_system ON documents(system_type);
                   CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(document_id);
               """)

   def insert_document(metadata: dict) -> str:
       """Insert document metadata, return document_id."""
       doc_id = str(uuid.uuid4())
       with get_connection() as conn:
           with conn.cursor() as cur:
               cur.execute("""
                   INSERT INTO documents (id, title, document_number, issue_date,
                       effective_date, issuing_body, document_type, source_file,
                       source_url, system_type)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               """, (doc_id, metadata.get('title'), ...))
       return doc_id

   def insert_chunks_batch(chunks: list[dict]):
       """Batch insert chunk records."""
       ...

   def query_documents(doc_type=None, date_from=None, date_to=None):
       """Filter documents by type and date range."""
       ...
   ```

4. Update `src/embedding/ingest_pipeline.py` to call `insert_document()` and `insert_chunks_batch()` during ingestion

## Todo
- [ ] Add psycopg2-binary to requirements.txt
- [ ] Create `src/storage/__init__.py`
- [ ] Create `src/storage/postgres-metadata.py` with schema + CRUD
- [ ] Update ingest pipeline to store metadata in PostgreSQL
- [ ] Test with sample documents

## Success Criteria
- Tables created successfully on PostgreSQL
- Document metadata inserted during ingestion
- Date range and type queries return correct results
- Chunk IDs consistent between PostgreSQL and Qdrant

## Risk Assessment
- **Date parsing**: Vietnamese documents have varied date formats → reuse `extract_date()` from `filter-latest-regulations.py`
- **Connection management**: Use connection pooling if needed later, `psycopg2` connections fine for now
