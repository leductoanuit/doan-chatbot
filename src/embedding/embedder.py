"""Vietnamese text embedding using BAAI/bge-m3 (1024-dim, dense only).

BGE-M3 supports multilingual semantic understanding — handles synonyms,
paraphrases, and colloquial Vietnamese much better than PhoBERT-based models.
"""

from typing import List

from sentence_transformers import SentenceTransformer


class VietnameseEmbedder:
    """Wraps BAAI/bge-m3 for batch and single-query embedding."""

    MODEL_NAME = "BAAI/bge-m3"
    DIMENSION = 1024

    def __init__(self, model_name: str = MODEL_NAME):
        print(f"[embedder] Loading {model_name} …")
        self.model = SentenceTransformer(model_name)
        # BGE-M3 supports up to 8192 tokens but 512 is optimal for retrieval
        self.model.max_seq_length = 512
        print("[embedder] Model ready")

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 16,
    ) -> List[List[float]]:
        """Embed a list of texts in batches.

        Returns:
            List of float lists (one per input text).
        """
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self.model.encode(
                batch,
                show_progress_bar=False,
                normalize_embeddings=True,  # BGE-M3 requires normalized embeddings
            )
            all_embeddings.extend(embeddings.tolist())

            processed = i + len(batch)
            if processed % (batch_size * 10) == 0 or processed == len(texts):
                print(f"[embedder] {processed}/{len(texts)} embedded")

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        return self.model.encode(query, normalize_embeddings=True).tolist()
