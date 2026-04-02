# Phase 4: Qdrant Vector Store Migration

## Context
- Current: MongoDB Atlas `$vectorSearch` with 768-dim cosine similarity — `src/embedding/mongo_setup.py`
- Need: Replace with Qdrant for better vector search, native payload filtering
- Embedding model unchanged: `dangvantuan/vietnamese-embedding` (768-dim)

## Overview
- **Priority**: High
- **Status**: Pending
- **Description**: Replace MongoDB Atlas vector storage with Qdrant, update setup and ingestion

## Requirements
### Functional
- Create Qdrant collection with 768-dim cosine distance
- Upsert vectors with payload (content, metadata, chunk references)
- Support filtered search by document_type, system_type
- Store chunk content in payload for retrieval without DB roundtrip

### Non-functional
- Use `qdrant-client` Python SDK
- Run Qdrant via Docker locally or Qdrant Cloud
- Batch upsert for ingestion performance

## Architecture
```
Qdrant Collection: "documents"
├── Vector: 768-dim (cosine)
├── Point ID: chunk_id string
└── Payload:
    ├── content: str (full chunk text)
    ├── document_id: str (FK to PostgreSQL)
    ├── source: str
    ├── page: int
    ├── chunk_idx: int
    ├── document_type: str
    ├── system_type: str
    └── method: str (pymupdf/deepseek_ocr)
```

## Related Code Files
- **Create**: `src/storage/qdrant-vector-store.py` — Qdrant setup + CRUD
- **Modify**: `src/embedding/ingest_pipeline.py` — replace MongoDB insert with Qdrant upsert
- **Delete logic in**: `src/embedding/mongo_setup.py` — no longer needed (keep file, mark deprecated)
- **Modify**: `requirements.txt` — add `qdrant-client`

## Implementation Steps

1. Add `qdrant-client` to `requirements.txt`

2. Create `src/storage/qdrant-vector-store.py`:
   ```python
   import os
   from qdrant_client import QdrantClient
   from qdrant_client.models import (
       Distance, VectorParams, PointStruct,
       Filter, FieldCondition, MatchValue
   )

   QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
   QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
   COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "documents")
   EMBEDDING_DIM = 768

   def get_client() -> QdrantClient:
       return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

   def init_collection():
       """Create collection if not exists."""
       client = get_client()
       collections = [c.name for c in client.get_collections().collections]
       if COLLECTION_NAME not in collections:
           client.create_collection(
               collection_name=COLLECTION_NAME,
               vectors_config=VectorParams(
                   size=EMBEDDING_DIM,
                   distance=Distance.COSINE
               )
           )
           # Create payload indexes for filtering
           client.create_payload_index(COLLECTION_NAME, "document_type", "keyword")
           client.create_payload_index(COLLECTION_NAME, "system_type", "keyword")
           client.create_payload_index(COLLECTION_NAME, "document_id", "keyword")
       return client

   def upsert_vectors(client, chunks: list[dict], embeddings: list[list[float]]):
       """Batch upsert vectors with payload."""
       points = []
       for chunk, embedding in zip(chunks, embeddings):
           chunk_id = f"{chunk['metadata'].get('document_id', 'unknown')}_{chunk['metadata']['page']}_{chunk['metadata']['chunk_idx']}"
           points.append(PointStruct(
               id=chunk_id,
               vector=embedding,
               payload={
                   "content": chunk["content"],
                   "document_id": chunk["metadata"].get("document_id", ""),
                   "source": chunk["metadata"]["source"],
                   "page": chunk["metadata"]["page"],
                   "chunk_idx": chunk["metadata"]["chunk_idx"],
                   "document_type": chunk["metadata"].get("document_type", ""),
                   "system_type": chunk["metadata"].get("system_type", ""),
                   "method": chunk["metadata"].get("method", ""),
               }
           ))

       # Batch upsert (100 at a time)
       BATCH = 100
       for i in range(0, len(points), BATCH):
           client.upsert(COLLECTION_NAME, points[i:i+BATCH])

   def search_vectors(client, query_vector, k=10, doc_type=None):
       """Search with optional payload filtering."""
       query_filter = None
       if doc_type:
           query_filter = Filter(must=[
               FieldCondition(key="document_type", match=MatchValue(value=doc_type))
           ])
       return client.query_points(
           collection_name=COLLECTION_NAME,
           query=query_vector,
           limit=k,
           query_filter=query_filter,
           with_payload=True,
       )
   ```

3. Update `src/embedding/ingest_pipeline.py`:
   - Replace `setup_database()` → `init_collection()`
   - Replace `collection.insert_many()` → `upsert_vectors()`
   - Remove MongoDB imports

4. Add Docker Compose config for local Qdrant:
   ```yaml
   # docker-compose.yml (or add to existing)
   services:
     qdrant:
       image: qdrant/qdrant
       ports:
         - "6333:6333"
       volumes:
         - qdrant_data:/qdrant/storage
   volumes:
     qdrant_data:
   ```

## Todo
- [ ] Add qdrant-client to requirements.txt
- [ ] Create `src/storage/qdrant-vector-store.py`
- [ ] Update ingest pipeline to use Qdrant
- [ ] Add Qdrant to docker-compose.yml
- [ ] Update `.env.example` with Qdrant config
- [ ] Test upsert + search with sample data

## Success Criteria
- Qdrant collection created with 768-dim cosine vectors
- All chunks upserted with correct payload
- Vector search returns relevant results
- Filtered search by document_type works

## Risk Assessment
- **Qdrant Cloud vs local**: Cloud is easier but costs money → Docker for dev, Cloud for prod
- **Point ID format**: Qdrant accepts string IDs — use `{doc_id}_{page}_{chunk_idx}` consistently
- **Migration**: No live migration needed — fresh ingest into Qdrant from processed documents
