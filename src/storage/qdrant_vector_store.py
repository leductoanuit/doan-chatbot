"""Qdrant vector store — collection management, upsert, and search.

Config via environment variables:
    QDRANT_URL        — default http://localhost:6333
    QDRANT_API_KEY    — optional (for Qdrant Cloud)
    QDRANT_COLLECTION — default "documents"
"""

import os
import hashlib
from typing import Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")

# Must match the embedding model output dimension (BAAI/bge-m3 = 1024)
EMBEDDING_DIM = 1024

BATCH_SIZE = 100


def get_client() -> QdrantClient:
    """Return a QdrantClient using env-configured URL and optional API key."""
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def init_collection(client: Optional[QdrantClient] = None) -> QdrantClient:
    """Create collection if not exists, with payload indexes. Return client.

    Creates:
      - 768-dim cosine vector collection
      - Keyword indexes: document_type, system_type, document_id
    """
    if client is None:
        client = get_client()

    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"[qdrant] Created collection '{QDRANT_COLLECTION}'")

        # Create payload indexes for fast filtered search
        for field in ("document_type", "system_type", "document_id"):
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
            print(f"[qdrant] Created keyword index on '{field}'")
    else:
        print(f"[qdrant] Collection '{QDRANT_COLLECTION}' already exists")

    return client


def _make_point_id(document_id: str, page: int, chunk_idx: int) -> int:
    """Build a deterministic unsigned int ID from document metadata.
    Qdrant requires unsigned int (u64) or UUID — take mod 2^63 to stay safe."""
    raw = f"{document_id}_{page}_{chunk_idx}"
    return int(hashlib.md5(raw.encode()).hexdigest(), 16) % (2**63)


def upsert_vectors(
    client: QdrantClient,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> None:
    """Batch upsert chunk vectors into Qdrant.

    Point ID = stable hash of {document_id}_{page}_{chunk_idx}.
    Payload stores all metadata fields for retrieval and filtering.
    """
    points = []
    for chunk, embedding in zip(chunks, embeddings):
        meta = chunk.get("metadata", {})
        doc_id = str(meta.get("document_id", meta.get("source", "unknown")))
        page = int(meta.get("page", 0))
        chunk_idx = int(meta.get("chunk_idx", 0))

        points.append(
            PointStruct(
                id=_make_point_id(doc_id, page, chunk_idx),
                vector=embedding,
                payload={
                    "content": chunk["content"],
                    "document_id": doc_id,
                    "source": meta.get("source", ""),
                    "page": page,
                    "chunk_idx": chunk_idx,
                    "document_type": meta.get("document_type", ""),
                    "system_type": meta.get("system_type", ""),
                    "method": meta.get("method", ""),
                },
            )
        )

    # Upload in batches of BATCH_SIZE
    total = len(points)
    for i in range(0, total, BATCH_SIZE):
        batch = points[i : i + BATCH_SIZE]
        client.upsert(collection_name=QDRANT_COLLECTION, points=batch)
        print(f"[qdrant] Upserted {i + len(batch)}/{total} points")


def search_vectors(
    client: QdrantClient,
    query_vector: list[float],
    k: int = 10,
    doc_type: Optional[str] = None,
    system_type: Optional[str] = None,
) -> list[dict]:
    """Search collection with optional payload filters.

    Returns list of dicts with 'score' and payload fields.
    """
    conditions = []
    if doc_type:
        conditions.append(FieldCondition(key="document_type", match=MatchValue(value=doc_type)))
    if system_type:
        conditions.append(FieldCondition(key="system_type", match=MatchValue(value=system_type)))

    query_filter = Filter(must=conditions) if conditions else None

    hits = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        query_filter=query_filter,
        limit=k,
        with_payload=True,
    ).points

    return [{"score": h.score, **h.payload} for h in hits]


def delete_collection(client: QdrantClient) -> None:
    """Delete the entire collection (use before re-ingest from scratch)."""
    client.delete_collection(collection_name=QDRANT_COLLECTION)
    print(f"[qdrant] Deleted collection '{QDRANT_COLLECTION}'")
