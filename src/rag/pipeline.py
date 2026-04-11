"""RAG pipeline — orchestrates retrieval and LLM generation."""

from typing import Dict, List, Optional

from src.rag.retriever import HybridRetriever
from src.rag.llm_client import LLMClient
from src.rag.reranker import BGEReranker


class RAGPipeline:
    """Full RAG query: embed query → hybrid search → rerank → build context → generate answer."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.retriever = retriever or HybridRetriever()
        self.llm = llm_client or LLMClient()
        self.reranker = BGEReranker()

    @staticmethod
    def _expand_query(question: str) -> str:
        """Normalize and expand query for better embedding match.

        Two problems addressed:
        1. Short queries missing domain context → append "(hệ đào tạo từ xa UIT)"
        2. "hệ từ xa" / "chương trình từ xa" shorthand doesn't match corpus phrasing
           "đào tạo từ xa" → normalize to full form before embedding.
        """
        q = question

        # Normalize shorthand variants to full corpus phrasing
        normalizations = [
            ("hệ từ xa", "hệ đào tạo từ xa"),
            ("chương trình từ xa", "chương trình đào tạo từ xa"),
        ]
        q_lower = q.lower()
        for short, full in normalizations:
            if short in q_lower:
                idx = q_lower.index(short)
                q = q[:idx] + full + q[idx + len(short):]
                q_lower = q.lower()

        # Append domain context only for very short/generic queries lacking any domain signal
        domain_signals = [
            "đào tạo từ xa", "uit", "đại học công nghệ thông tin",
            "quy chế", "học vụ", "tuyển sinh", "học phí", "tín chỉ",
            "sinh viên", "môn học", "chương trình", "ngành", "bằng",
        ]
        if not any(kw in q_lower for kw in domain_signals):
            q = f"{q} (hệ đào tạo từ xa UIT)"
            q_lower = q.lower()

        return q

    def query(
        self,
        question: str,
        history: Optional[List[Dict]] = None,
        top_k: int = 10,
    ) -> Dict:
        """Run RAG: retrieve relevant docs then generate an answer with context.

        Args:
            question: User question string.
            history: Prior conversation turns (list of role/content dicts).
            top_k: Number of document chunks to retrieve.

        Returns:
            Dict with keys: answer, sources, context_used.
        """
        # 1. Expand query with domain context for better retrieval
        expanded_query = self._expand_query(question)

        # 2. Retrieve + rerank using expanded query
        results = self.retriever.hybrid_search(expanded_query, k=top_k, reranker=self.reranker)

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
