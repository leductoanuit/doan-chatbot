"""Hybrid retriever — MongoDB vector search + keyword fallback for Vietnamese queries."""

import os
import re
from typing import List, Dict

from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

load_dotenv()

_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "dangvantuan/vietnamese-embedding")
_MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
_DB_NAME = os.getenv("MONGODB_DB", "chatbot")
_COLLECTION = os.getenv("MONGODB_COLLECTION", "documents")
_VECTOR_INDEX = "vector_search_idx"


class HybridRetriever:
    """Combines MongoDB $vectorSearch with regex keyword fallback.

    Vector search handles semantic similarity; keyword search catches
    exact-term queries that may score low in embedding space.
    """

    def __init__(
        self,
        mongo_uri: str = _MONGO_URI,
        db_name: str = _DB_NAME,
        collection_name: str = _COLLECTION,
        embedding_model: str = _EMBEDDING_MODEL,
    ):
        self.mongo_client = MongoClient(mongo_uri)
        self.collection = self.mongo_client[db_name][collection_name]
        self.embedder = SentenceTransformer(embedding_model)

    # ------------------------------------------------------------------
    # Individual search strategies
    # ------------------------------------------------------------------

    def vector_search(self, query: str, k: int = 10) -> List[Dict]:
        """Semantic search via MongoDB Atlas $vectorSearch."""
        query_embedding = self.embedder.encode(query).tolist()
        pipeline = [
            {
                "$vectorSearch": {
                    "index": _VECTOR_INDEX,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "k": k,
                    "numCandidates": k * 10,
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
        ]
        return list(self.collection.aggregate(pipeline))

    def keyword_search(self, query: str, k: int = 10) -> List[Dict]:
        """Regex keyword search — fallback for exact-term matching."""
        keywords = [w for w in query.lower().split() if len(w) > 2]
        if not keywords:
            return []

        pattern = "|".join(re.escape(kw) for kw in keywords)
        results = list(
            self.collection.find(
                {"content": {"$regex": pattern, "$options": "i"}},
                {"content": 1, "metadata": 1, "_id": 0},
            ).limit(k)
        )
        for r in results:
            r["score"] = 0.5  # Uniform score for keyword hits
            r["search_type"] = "keyword"
        return results

    # ------------------------------------------------------------------
    # Hybrid merge
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> List[Dict]:
        """Merge vector + keyword results, deduplicate, rank by weighted score."""
        vector_results = self.vector_search(query, k=k * 2)
        keyword_results = self.keyword_search(query, k=k)

        seen: set[int] = set()
        merged: List[Dict] = []

        for r in vector_results:
            key = hash(r["content"][:100])
            if key not in seen:
                seen.add(key)
                r["final_score"] = r.get("score", 0.0) * vector_weight
                r["search_type"] = "vector"
                merged.append(r)

        for r in keyword_results:
            key = hash(r["content"][:100])
            if key not in seen:
                seen.add(key)
                r["final_score"] = r.get("score", 0.0) * keyword_weight
                merged.append(r)

        merged.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        return merged[:k]

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def build_context(self, results: List[Dict], max_tokens: int = 1500) -> str:
        """Assemble retrieved chunks into a context string, capped at max_tokens words."""
        parts: List[str] = []
        token_count = 0

        for r in results:
            content = r["content"]
            approx_tokens = len(content.split())
            if token_count + approx_tokens > max_tokens:
                break
            source = r.get("metadata", {}).get("source", "N/A")
            parts.append(f"[Nguồn: {source}]\n{content}")
            token_count += approx_tokens

        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_healthy(self) -> bool:
        """Return True if MongoDB is reachable."""
        try:
            self.mongo_client.admin.command("ping")
            return True
        except Exception:
            return False
