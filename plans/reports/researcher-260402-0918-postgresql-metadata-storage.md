# PostgreSQL Document Metadata Storage Research Report

**Date:** 2026-04-02 | **Research Focus:** PostgreSQL schema, CRUD patterns, Qdrant integration for Vietnamese legal document chatbot

---

## Executive Summary

PostgreSQL is ideal for metadata + Qdrant vectors pattern. Use B-tree indexes on dates + document_type, composite keys (document_id, chunk_id) for linking, and asyncpg for concurrent operations. Staging table approach fastest for bulk upserts.

---

## 1. Schema Design

### Recommended Structure

```sql
CREATE TABLE documents (
  document_id BIGSERIAL PRIMARY KEY,
  title TEXT NOT NULL,
  document_number VARCHAR(50) UNIQUE NOT NULL,
  document_type VARCHAR(50) NOT NULL,  -- e.g., 'law', 'decree', 'regulation'
  issue_date DATE NOT NULL,
  effective_date DATE NOT NULL,
  source_url TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE document_chunks (
  chunk_id BIGSERIAL PRIMARY KEY,
  document_id BIGINT REFERENCES documents(document_id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  content TEXT NOT NULL,
  content_hash CHAR(32),  -- MD5 for change detection
  qdrant_point_id BIGINT,  -- Link to Qdrant vector ID
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(document_id, chunk_index)
);
```

**Key Design Choices:**
- document_number as unique natural key (for external references)
- document_type as string (filterable, easier than enums for future types)
- chunk_index + document_id composite uniqueness
- qdrant_point_id bridges PostgreSQL metadata ↔ Qdrant vectors
- content_hash enables efficient change detection (don't re-embed unchanged chunks)

---

## 2. Linking PostgreSQL to Qdrant

### Pattern: UUID Management in PostgreSQL

**Critical:** Manage point IDs externally (PostgreSQL), not in Qdrant. This eliminates lookup steps.

**Flow:**
1. Insert document → Get document_id from PostgreSQL
2. Split into chunks → Store chunk_id in PostgreSQL
3. Generate embeddings → Use chunk_id as qdrant_point_id
4. For updates: Query PostgreSQL for chunk_id → Use to update Qdrant vector

**Reconciliation:** Background job checks for orphaned vectors (in Qdrant but not PostgreSQL) and missing embeddings (in PostgreSQL but not Qdrant).

---

## 3. Index Strategy

### Essential Indexes

```sql
-- Date range filtering (primary use case)
CREATE INDEX idx_documents_date_range 
  ON documents(issue_date, effective_date);

-- Document type filtering
CREATE INDEX idx_documents_type 
  ON documents(document_type);

-- Composite for common queries: type + date range
CREATE INDEX idx_documents_type_dates 
  ON documents(document_type, issue_date DESC, effective_date);

-- Chunk lookups
CREATE INDEX idx_chunks_document_id 
  ON document_chunks(document_id);

-- Qdrant linking
CREATE INDEX idx_chunks_qdrant_point 
  ON document_chunks(qdrant_point_id);
```

**Index Performance:**
- B-tree indexes fastest for BETWEEN/date range queries
- Composite index on (type, date) covers most filter queries without separate lookups
- BRIN indexes alternative if documents sorted by date (storage cost very low)

---

## 4. CRUD Patterns

### async/await with asyncpg (Recommended)

**Why asyncpg over psycopg2:**
- Supports async/await (non-blocking)
- 3-10x faster for concurrent operations
- Better for RAG pipelines with many parallel embedding calls

### Basic CRUD

```python
import asyncpg

async def create_document(pool, title, doc_num, doc_type, issue_date, effective_date, source_url):
    async with pool.acquire() as conn:
        result = await conn.fetchrow('''
            INSERT INTO documents 
            (title, document_number, document_type, issue_date, effective_date, source_url)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING document_id
        ''', title, doc_num, doc_type, issue_date, effective_date, source_url)
        return result['document_id']

async def filter_documents(pool, doc_type=None, start_date=None, end_date=None):
    async with pool.acquire() as conn:
        query = 'SELECT * FROM documents WHERE 1=1'
        params = []
        
        if doc_type:
            params.append(doc_type)
            query += f' AND document_type = ${len(params)}'
        if start_date:
            params.append(start_date)
            query += f' AND issue_date >= ${len(params)}'
        if end_date:
            params.append(end_date)
            query += f' AND issue_date <= ${len(params)}'
            
        return await conn.fetch(query + ' ORDER BY issue_date DESC', *params)
```

### Bulk Upsert (Staging Table Pattern - Fastest)

```python
async def bulk_upsert_documents(pool, documents):
    """
    documents: List of dicts with keys: title, document_number, document_type, issue_date, effective_date, source_url
    Returns: count of inserted/updated rows
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Create temp table
            await conn.execute('''
                CREATE TEMP TABLE temp_docs AS TABLE documents WITH NO DATA
            ''')
            
            # Bulk copy to staging
            await conn.copy_records_to_table(
                'temp_docs',
                records=[(d['title'], d['document_number'], d['document_type'], 
                         d['issue_date'], d['effective_date'], d['source_url']) 
                        for d in documents],
                columns=['title', 'document_number', 'document_type', 'issue_date', 'effective_date', 'source_url']
            )
            
            # Upsert from staging
            result = await conn.execute('''
                INSERT INTO documents 
                (title, document_number, document_type, issue_date, effective_date, source_url)
                SELECT title, document_number, document_type, issue_date, effective_date, source_url 
                FROM temp_docs
                ON CONFLICT (document_number) DO UPDATE SET
                  title = EXCLUDED.title,
                  issue_date = EXCLUDED.issue_date,
                  effective_date = EXCLUDED.effective_date,
                  updated_at = CURRENT_TIMESTAMP
            ''')
            
            return int(result.split()[-1])  # Extract row count from result string
```

### Chunk Insert with Qdrant Linking

```python
async def insert_chunks_batch(pool, document_id, chunks_with_embeddings):
    """
    chunks_with_embeddings: List of {content, qdrant_point_id, content_hash}
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            for idx, chunk in enumerate(chunks_with_embeddings):
                await conn.execute('''
                    INSERT INTO document_chunks 
                    (document_id, chunk_index, content, qdrant_point_id, content_hash)
                    VALUES ($1, $2, $3, $4, $5)
                ''', document_id, idx, chunk['content'], chunk['qdrant_point_id'], chunk['content_hash'])
```

---

## 5. Date Range & Document Type Filtering

### Typical Query Pattern

```python
async def search_documents_by_filters(pool, doc_types=None, issue_date_from=None, issue_date_to=None):
    """Filter documents for Vietnamese legal chatbot"""
    query_parts = ['SELECT * FROM documents WHERE 1=1']
    params = []
    
    if doc_types and len(doc_types) > 0:
        placeholders = ','.join([f'${i+1}' for i in range(len(doc_types))])
        query_parts.append(f'AND document_type IN ({placeholders})')
        params.extend(doc_types)
    
    if issue_date_from:
        params.append(issue_date_from)
        query_parts.append(f'AND issue_date >= ${len(params)}')
    
    if issue_date_to:
        params.append(issue_date_to)
        query_parts.append(f'AND issue_date <= ${len(params)}')
    
    query_parts.append('ORDER BY issue_date DESC')
    query = ' '.join(query_parts)
    
    async with pool.acquire() as conn:
        return await conn.fetch(query, *params)
```

---

## Key Decisions for Legal Document Chatbot

| Decision | Rationale |
|----------|-----------|
| **asyncpg** | Concurrent embedding calls, non-blocking I/O |
| **document_type VARCHAR** | Flexibility for law/decree/regulation/circular types |
| **Staging table for bulk ops** | 2-3x faster than executemany() |
| **B-tree composite index (type, date)** | Covers 90% of filter queries |
| **content_hash column** | Skip re-embedding unchanged content |
| **qdrant_point_id in PostgreSQL** | Single source of truth for ID mapping |
| **document_number UNIQUE** | Natural key for external API calls, idempotent upserts |

---

## Unresolved Questions

1. What's the expected document volume (100s, 1000s, 100k+)? Affects partitioning strategy.
2. Do you need full-text search on document content, or only metadata filters? (Might justify pgvector extension)
3. How often are documents updated vs. created? (Affects reconciliation job frequency)
4. Should chunk_id be UUID or BIGINT for Qdrant compatibility?

---

## Sources

- [PostgreSQL Index Types Documentation](https://www.postgresql.org/docs/current/indexes-types.html)
- [asyncpg Official Documentation](https://magicstack.github.io/asyncpg/current/usage.html)
- [Asyncpg Batch Upsert Patterns](https://medium.com/@santhanu/batch-upsert-pyspark-dataframe-into-postgres-tables-with-error-handling-using-psycopg2-and-asyncpg-59f08aa020b0)
- [PostgreSQL Staging Table for Bulk Upserts](https://overflow.no/blog/2025/1/5/using-staging-tables-for-faster-bulk-upserts-with-python-and-postgresql/)
- [Qdrant + PostgreSQL UUID Integration](https://medium.com/razroo/how-to-update-an-existing-vector-in-qdrant-using-an-external-database-for-uuid-management-ec99cf5a50b1)
- [Document Chunking with PostgreSQL](https://medium.com/@changtimwu/efficient-document-change-detection-in-postgresql-a-deep-dive-into-chunking-and-hashing-a455195331f2)
- [Range Types & GiST Indexes for Date Filtering](https://www.postgresql.org/docs/current/rangetypes.html)
