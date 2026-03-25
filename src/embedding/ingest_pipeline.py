"""Full ingestion pipeline — chunk → embed → store in MongoDB Atlas.

Usage:
    python src/embedding/ingest-pipeline.py
    python src/embedding/ingest-pipeline.py --data data/processed/all_documents.json
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
from pymongo import MongoClient

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.embedding.chunker import chunk_documents
from src.embedding.embedder import VietnameseEmbedder
from src.embedding.mongo_setup import setup_database

load_dotenv()

BATCH_INSERT_SIZE = 100


def ingest(data_path: str = "data/processed/all_documents.json") -> None:
    """Load documents, chunk, embed, and bulk-insert into MongoDB."""

    # ------------------------------------------------------------------
    # 1. Load processed documents
    # ------------------------------------------------------------------
    print(f"[ingest] Loading {data_path} …")
    with open(data_path, "r", encoding="utf-8") as fh:
        documents = json.load(fh)
    print(f"[ingest] Loaded {len(documents)} documents")

    # ------------------------------------------------------------------
    # 2. Chunk
    # ------------------------------------------------------------------
    print("[ingest] Chunking …")
    chunks = chunk_documents(documents)
    print(f"[ingest] {len(chunks)} chunks created")

    # ------------------------------------------------------------------
    # 3. Initialise DB (creates collection + index if not present)
    # ------------------------------------------------------------------
    print("[ingest] Setting up MongoDB …")
    collection = setup_database()

    # Skip already-ingested documents (idempotent re-run)
    existing = collection.count_documents({})
    if existing > 0:
        print(f"[ingest] {existing} documents already in collection — clearing for fresh ingest")
        collection.delete_many({})

    # ------------------------------------------------------------------
    # 4. Embed
    # ------------------------------------------------------------------
    print("[ingest] Embedding …")
    embedder = VietnameseEmbedder()
    texts = [c["content"] for c in chunks]
    embeddings = embedder.embed_texts(texts)

    # ------------------------------------------------------------------
    # 5. Insert in batches
    # ------------------------------------------------------------------
    print("[ingest] Inserting into MongoDB …")
    docs_to_insert = [
        {
            "content": chunk["content"],
            "embedding": embedding,
            "metadata": chunk["metadata"],
            "token_count": len(chunk["content"].split()),
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]

    for i in range(0, len(docs_to_insert), BATCH_INSERT_SIZE):
        batch = docs_to_insert[i : i + BATCH_INSERT_SIZE]
        collection.insert_many(batch)
        print(f"[ingest] {i + len(batch)}/{len(docs_to_insert)} inserted")

    print(f"\n[ingest] Complete — {len(docs_to_insert)} chunks stored")

    # ------------------------------------------------------------------
    # 6. Smoke-test vector search
    # ------------------------------------------------------------------
    _smoke_test(collection, embedder)


def _smoke_test(collection, embedder: VietnameseEmbedder) -> None:
    """Run a quick vector search to verify the index is working."""
    test_query = "Thông tin tuyển sinh đại học"
    print(f"\n[ingest] Smoke test: '{test_query}'")

    query_emb = embedder.embed_query(test_query)
    try:
        results = list(collection.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_search_idx",
                    "path": "embedding",
                    "queryVector": query_emb,
                    "k": 3,
                    "numCandidates": 50,
                }
            },
            {
                "$project": {
                    "content": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"},
                    "_id": 0,
                }
            },
        ]))

        if results:
            for r in results:
                score = r.get("score", "N/A")
                preview = r["content"][:100].replace("\n", " ")
                print(f"  score={score:.3f} | {preview}…")
        else:
            print("  No results — vector index may still be building (wait ~1 min)")
    except Exception as exc:
        print(f"  Search error: {exc}")
        print("  Vector index may still be activating on Atlas")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into MongoDB")
    parser.add_argument(
        "--data",
        default="data/processed/all_documents.json",
        help="Path to processed JSON file",
    )
    args = parser.parse_args()
    ingest(data_path=args.data)
