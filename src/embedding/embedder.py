"""Vietnamese text embedding using dangvantuan/vietnamese-embedding (768-dim)."""

from typing import List

from sentence_transformers import SentenceTransformer


class VietnameseEmbedder:
    """Wraps dangvantuan/vietnamese-embedding for batch and single-query use."""

    MODEL_NAME = "dangvantuan/vietnamese-embedding"
    DIMENSION = 768

    def __init__(self, model_name: str = MODEL_NAME):
        print(f"[embedder] Loading {model_name} …")
        self.model = SentenceTransformer(model_name)
        print("[embedder] Model ready")

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[List[float]]:
        """Embed a list of texts in batches.

        Returns:
            List of float lists (one per input text).
        """
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self.model.encode(batch, show_progress_bar=False)
            all_embeddings.extend(embeddings.tolist())

            processed = i + len(batch)
            if processed % (batch_size * 10) == 0 or processed == len(texts):
                print(f"[embedder] {processed}/{len(texts)} embedded")

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        return self.model.encode(query).tolist()
