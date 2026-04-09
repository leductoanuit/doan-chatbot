"""Hybrid retriever — Qdrant vector search + keyword fallback for Vietnamese queries."""

import logging
import os
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.storage.qdrant_vector_store import get_client, search_vectors

load_dotenv()

_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")


class HybridRetriever:
    """Combines Qdrant vector search with keyword fallback.

    Vector search handles semantic similarity; keyword search catches
    exact-term queries that may score low in embedding space.
    """

    def __init__(self, embedding_model: str = _EMBEDDING_MODEL):
        self.qdrant_client = get_client()
        self.embedder = SentenceTransformer(embedding_model)

    # ------------------------------------------------------------------
    # Individual search strategies
    # ------------------------------------------------------------------

    def vector_search(
        self,
        query: str,
        k: int = 10,
        doc_type: Optional[str] = None,
        system_type: Optional[str] = None,
    ) -> List[Dict]:
        """Semantic search via Qdrant."""
        query_embedding = self.embedder.encode(query, normalize_embeddings=True).tolist()
        results = search_vectors(
            self.qdrant_client, query_embedding, k=k,
            doc_type=doc_type, system_type=system_type,
        )
        # Normalize to standard format expected by pipeline
        return [
            {
                "content": r.get("content", ""),
                "metadata": {
                    "source": r.get("source", ""),
                    "page": r.get("page", 0),
                    "document_type": r.get("document_type", ""),
                    "system_type": r.get("system_type", ""),
                },
                "score": r.get("score", 0.0),
                "search_type": "vector",
            }
            for r in results
        ]

    def keyword_search(self, query: str, k: int = 10) -> List[Dict]:
        """Keyword fallback — scroll Qdrant payload for content matches.

        Uses Qdrant scroll with payload substring matching. Less powerful
        than MongoDB regex but sufficient for exact-term fallback.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchText

        words = [w for w in query.lower().split() if len(w) > 2]
        if not words:
            return []

        # Prefer bigrams (compound phrases) over single words for precision.
        # e.g. "học phí" matches Q&A chunks better than "học" or "phí" alone.
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        keywords = (bigrams + words)[:4]  # bigrams first, then single words, cap at 4

        results: List[Dict] = []
        for keyword in keywords:  # Limit to 4 keywords for performance
            try:
                hits, _ = self.qdrant_client.scroll(
                    collection_name=os.getenv("QDRANT_COLLECTION", "documents"),
                    scroll_filter=Filter(
                        must=[FieldCondition(key="content", match=MatchText(text=keyword))]
                    ),
                    limit=k,
                    with_payload=True,
                )
                for hit in hits:
                    payload = hit.payload or {}
                    results.append({
                        "content": payload.get("content", ""),
                        "metadata": {
                            "source": payload.get("source", ""),
                            "page": payload.get("page", 0),
                            "document_type": payload.get("document_type", ""),
                            "system_type": payload.get("system_type", ""),
                        },
                        "score": 0.5,
                        "search_type": "keyword",
                    })
            except Exception as exc:
                logger.warning("Keyword search error for '%s': %s", keyword, exc)
                continue

        return results[:k]

    # ------------------------------------------------------------------
    # Hybrid merge
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        doc_type: Optional[str] = None,
        system_type: Optional[str] = None,
    ) -> List[Dict]:
        """Merge vector + keyword results, deduplicate, rank by weighted score."""
        vector_results = self.vector_search(query, k=k * 4, doc_type=doc_type, system_type=system_type)
        keyword_results = self.keyword_search(query, k=k)

        seen: set[int] = set()
        merged: List[Dict] = []

        for r in vector_results:
            key = hash(r["content"][:100])
            if key not in seen:
                seen.add(key)
                r["final_score"] = r.get("score", 0.0) * vector_weight
                merged.append(r)

        for r in keyword_results:
            key = hash(r["content"][:100])
            if key not in seen:
                seen.add(key)
                r["final_score"] = r.get("score", 0.0) * keyword_weight
                merged.append(r)

        merged.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

        # Chỉ giữ kết quả đủ liên quan (vector score >= 0.25)
        # Nếu không có kết quả nào đạt ngưỡng, trả về top-k để Gemini tự đánh giá
        MIN_SCORE = 0.25
        filtered = [r for r in merged if r.get("final_score", 0.0) >= MIN_SCORE * vector_weight]
        return (filtered if filtered else merged)[:k]

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
        """Return True if Qdrant is reachable."""
        try:
            self.qdrant_client.get_collections()
            return True
        except Exception:
            return False
