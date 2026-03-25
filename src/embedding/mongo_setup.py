"""MongoDB Atlas setup — creates collection and 768-dim vector search index.

Run once before ingestion:
    python src/embedding/mongo-setup.py
"""

import os

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGODB_DB", "chatbot")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION", "documents")
INDEX_NAME = "vector_search_idx"
EMBEDDING_DIM = 768


def setup_database() -> object:
    """Initialise MongoDB collection + vector search index.

    Returns:
        pymongo Collection object ready for document insertion.
    """
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    # Create collection if absent
    if COLLECTION_NAME not in db.list_collection_names():
        db.create_collection(COLLECTION_NAME)
        print(f"[mongo-setup] Created collection '{COLLECTION_NAME}'")

    collection = db[COLLECTION_NAME]

    # Regular indexes for metadata filtering
    collection.create_index("metadata.source", background=True)
    collection.create_index("metadata.document_type", background=True)
    print("[mongo-setup] Metadata indexes ensured")

    # Vector search index (Atlas Search — requires Atlas Free Tier or M10+)
    vector_index = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": EMBEDDING_DIM,
                    "similarity": "cosine",
                },
                {
                    "type": "filter",
                    "path": "metadata.source",
                },
                {
                    "type": "filter",
                    "path": "metadata.document_type",
                },
            ]
        },
        name=INDEX_NAME,
        type="vectorSearch",
    )

    try:
        collection.create_search_indexes([vector_index])
        print(f"[mongo-setup] Vector search index '{INDEX_NAME}' created (may take ~1 min to activate)")
    except Exception as exc:
        # Index may already exist or Atlas tier may not support programmatic creation
        print(f"[mongo-setup] Index note: {exc}")
        print("[mongo-setup] If using Atlas, create the index manually via the UI:")
        print(f"  Database: {DB_NAME}, Collection: {COLLECTION_NAME}")
        print(f"  Index name: {INDEX_NAME}, Dimensions: {EMBEDDING_DIM}, Similarity: cosine")

    return collection


if __name__ == "__main__":
    col = setup_database()
    count = col.count_documents({})
    print(f"[mongo-setup] Done — collection has {count} documents")
