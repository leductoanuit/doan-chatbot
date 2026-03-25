"""RAG pipeline — orchestrates retrieval and LLM generation."""

from typing import Dict, List, Optional

from src.rag.retriever import HybridRetriever
from src.rag.llm_client import LLMClient


class RAGPipeline:
    """Full RAG query: embed query → hybrid search → build context → generate answer."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.retriever = retriever or HybridRetriever()
        self.llm = llm_client or LLMClient()

    def query(
        self,
        question: str,
        history: Optional[List[Dict]] = None,
        top_k: int = 5,
    ) -> Dict:
        """Run RAG: retrieve relevant docs then generate an answer with context.

        Args:
            question: User question string.
            history: Prior conversation turns (list of role/content dicts).
            top_k: Number of document chunks to retrieve.

        Returns:
            Dict with keys: answer, sources, context_used.
        """
        # 1. Retrieve
        results = self.retriever.hybrid_search(question, k=top_k)

        # 2. Build context string (capped at 1500 words)
        context = self.retriever.build_context(results, max_tokens=1500)

        # 3. Generate
        answer = self.llm.generate(
            query=question,
            context=context,
            history=history,
        )

        # 4. Collect source metadata for citation
        sources = [
            {
                "source": r.get("metadata", {}).get("source", ""),
                "page": r.get("metadata", {}).get("page", 0),
                "score": round(r.get("final_score", 0.0), 3),
                "preview": r["content"][:150] + "…",
            }
            for r in results
        ]

        return {
            "answer": answer,
            "sources": sources,
            "context_used": len(results),
        }
