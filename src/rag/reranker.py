"""Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

Reranks retrieved chunks by scoring each (query, chunk) pair together —
more accurate than bi-encoder cosine similarity for final selection.
"""

from typing import List, Dict

from sentence_transformers import CrossEncoder


class BGEReranker:
    """Wraps BAAI/bge-reranker-v2-m3 for re-ranking retrieved chunks."""

    MODEL_NAME = "BAAI/bge-reranker-v2-m3"

    def __init__(self, model_name: str = MODEL_NAME):
        print(f"[reranker] Loading {model_name} …")
        self.model = CrossEncoder(model_name, max_length=512)
        print("[reranker] Model ready")

    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """Score each (query, chunk) pair and return top_k sorted by score.

        Args:
            query: User query string.
            chunks: List of chunk dicts with 'content' key.
            top_k: Number of top chunks to return.

        Returns:
            Reranked list of chunk dicts with added 'rerank_score' field.
        """
        if not chunks:
            return []

        pairs = [(query, c["content"]) for c in chunks]
        scores = self.model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
