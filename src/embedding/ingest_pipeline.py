"""Full ingestion pipeline — chunk → embed → store in Qdrant vector store.

Usage:
    python src/embedding/ingest-pipeline.py
    python src/embedding/ingest-pipeline.py --data data/processed/all_documents.json
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.embedding.chunker import chunk_documents
from src.embedding.embedder import VietnameseEmbedder
from src.storage.postgres_metadata import init_schema, insert_document, insert_chunks_batch
from src.storage.qdrant_vector_store import init_collection, search_vectors, upsert_vectors

load_dotenv()


METADATA_PATH = "data/processed/document-metadata.json"
DEFAULT_DATA_PATH = "data/processed/all_documents_final.json"


def _load_metadata_map() -> dict[str, dict]:
    """Load document metadata keyed by source_file for fast lookup.

    Supports both formats:
    - Dict (new): {source_file: {doc_id, title, ...}}
    - Array (legacy): [{source_file, title, ...}, ...]
    """
    if not os.path.exists(METADATA_PATH):
        print(f"[ingest] WARNING: {METADATA_PATH} not found, metadata will be sparse")
        return {}
    with open(METADATA_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return {e["source_file"]: e for e in data}
    return data  # Already dict keyed by source_file


def ingest(data_path: str = DEFAULT_DATA_PATH) -> None:
    """Load documents, chunk, embed, and upsert into Qdrant."""

    # ------------------------------------------------------------------
    # 1. Load processed documents
    # ------------------------------------------------------------------
    print(f"[ingest] Loading {data_path} …")
    with open(data_path, "r", encoding="utf-8") as fh:
        documents = json.load(fh)
    print(f"[ingest] Loaded {len(documents)} documents")

    # Load rich metadata for each source file
    meta_map = _load_metadata_map()
    print(f"[ingest] Loaded metadata for {len(meta_map)} source files")

    # ------------------------------------------------------------------
    # 2. Chunk
    # ------------------------------------------------------------------
    print("[ingest] Chunking …")
    chunks = chunk_documents(documents)
    print(f"[ingest] {len(chunks)} chunks created")

    # ------------------------------------------------------------------
    # 3. Initialise storage (Qdrant vectors + PostgreSQL metadata)
    # ------------------------------------------------------------------
    print("[ingest] Setting up Qdrant …")
    qdrant_client = init_collection()

    print("[ingest] Setting up PostgreSQL schema …")
    init_schema()

    # Insert document-level metadata into PostgreSQL.
    # Chunks from all_documents_final.json already carry metadata fields —
    # fall back to meta_map lookup for legacy chunk files without pre-merged metadata.
    sources_seen: dict[str, str] = {}  # source_file -> pg document_id
    for chunk in chunks:
        src = chunk["metadata"].get("source", "unknown")
        if src not in sources_seen:
            # Prefer pre-merged metadata on the chunk; fall back to meta_map
            chunk_meta = chunk.get("metadata", {})
            rich_meta = meta_map.get(src, {})
            pg_doc_id = insert_document({
                "source_file": src,
                "title": chunk_meta.get("title") or rich_meta.get("title"),
                "document_number": chunk_meta.get("document_number") or rich_meta.get("document_number"),
                "issue_date": chunk_meta.get("issue_date") or rich_meta.get("issue_date"),
                "issuing_body": chunk_meta.get("issuing_body") or rich_meta.get("issuing_body"),
                "document_type": chunk_meta.get("document_type") or rich_meta.get("document_type", ""),
                "system_type": chunk_meta.get("system_type") or rich_meta.get("system_type", ""),
            })
            sources_seen[src] = pg_doc_id
        chunk["metadata"]["document_id"] = sources_seen[src]

    # Insert chunk records into PostgreSQL
    pg_chunks = [
        {
            "id": f"{c['metadata']['document_id']}_{c['metadata']['page']}_{c['metadata']['chunk_idx']}",
            "document_id": c["metadata"]["document_id"],
            "page_number": c["metadata"]["page"],
            "chunk_index": c["metadata"]["chunk_idx"],
            "content_preview": c["content"][:200],
            "token_count": len(c["content"].split()),
        }
        for c in chunks
    ]
    insert_chunks_batch(pg_chunks)
    print(f"[ingest] {len(pg_chunks)} chunk records stored in PostgreSQL")

    # ------------------------------------------------------------------
    # 4. Embed
    # ------------------------------------------------------------------
    print("[ingest] Embedding …")
    embedder = VietnameseEmbedder()
    texts = [c["content"] for c in chunks]
    embeddings = embedder.embed_texts(texts)

    # ------------------------------------------------------------------
    # 5. Upsert vectors in batches (idempotent — same IDs overwrite)
    # ------------------------------------------------------------------
    print("[ingest] Upserting into Qdrant …")
    upsert_vectors(qdrant_client, chunks, embeddings)

    print(f"\n[ingest] Complete — {len(chunks)} chunks stored")

    # ------------------------------------------------------------------
    # 6. Smoke-test vector search
    # ------------------------------------------------------------------
    _smoke_test(qdrant_client, embedder)


def _smoke_test(qdrant_client, embedder: VietnameseEmbedder) -> None:
    """Run a quick vector search to verify Qdrant collection is working."""
    test_query = "Thông tin tuyển sinh đại học"
    print(f"\n[ingest] Smoke test: '{test_query}'")

    query_emb = embedder.embed_query(test_query)
    try:
        results = search_vectors(qdrant_client, query_emb, k=3)

        if results:
            for r in results:
                score = r.get("score", "N/A")
                preview = r.get("content", "")[:100].replace("\n", " ")
                print(f"  score={score:.3f} | {preview}…")
        else:
            print("  No results returned from Qdrant")
    except Exception as exc:
        print(f"  Search error: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant")
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help="Path to processed JSON file (default: all_documents_final.json)",
    )
    args = parser.parse_args()
    ingest(data_path=args.data)
