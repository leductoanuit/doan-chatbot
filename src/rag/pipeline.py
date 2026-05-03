"""RAG pipeline — orchestrates retrieval and LLM generation."""

import re
from typing import Dict, List, Optional

from src.rag.retriever import HybridRetriever
from src.rag.llm_client import LLMClient
from src.rag.reranker import BGEReranker
from src.rag.query_processor import (
    rewrite_standalone_query,
    generate_query_variants,
    reciprocal_rank_fusion,
    classify_query_intent,
    extract_comparison_subqueries,
)

# Chitchat patterns — skip RAG, answer directly with LLM
_CHITCHAT_PATTERNS = [
    r"^(hi|hello|hey|xin chào|chào|alo|ơi)\b",
    r"^(cảm ơn|cám ơn|thanks|thank you|ok|okay|ổn rồi|được rồi)\b",
    r"^(bạn là ai|bạn tên gì|ai tạo ra bạn|em là ai|mày là ai)\b",
    r"^(bạn có thể (làm|giúp|hỗ trợ)|bạn giỏi không|bạn biết gì)\b",
]


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

    def query(
        self,
        question: str,
        history: Optional[List[Dict]] = None,
        top_k: int = 10,
        use_multi_query: bool = False,
    ) -> Dict:
        """Run RAG: retrieve relevant docs then generate an answer with context.

        Args:
            question: User question string.
            history: Prior conversation turns (list of role/content dicts).
            top_k: Number of document chunks to retrieve.
            use_multi_query: Enable RAG Fusion with multi-query generation.

        Returns:
            Dict with keys: answer, sources, context_used.
        """
        # 0. Chitchat fast path — skip RAG entirely for greetings/small talk
        q_lower = question.lower().strip()
        if any(re.search(p, q_lower) for p in _CHITCHAT_PATTERNS):
            answer = self.llm.generate(query=question, context="", history=history)
            return {"answer": answer, "sources": [], "context_used": 0}

        # 1. Rewrite if query depends on conversation context (Phase 01)
        retrieval_query = rewrite_standalone_query(question, history or [], self.llm)

        # 2. Classify intent for metadata routing
        system_type_filter = classify_query_intent(retrieval_query)

        # 3. Boost top_k and force multi-query for comparison queries
        _COMPARISON_SIGNALS = ["so sánh", "khác nhau", "giống nhau", " vs ", "khác gì", "điểm khác", "điểm giống"]
        is_comparison = any(kw in q_lower for kw in _COMPARISON_SIGNALS)
        if is_comparison:
            top_k = top_k * 2
            use_multi_query = True  # Ensure both entities get dedicated retrieval

        # 4. Retrieve candidates
        if use_multi_query:
            # RAG Fusion: merge via RRF across multiple queries
            # For comparison queries, use entity-specific sub-queries (no LLM needed)
            # Otherwise fall back to LLM-generated variants
            if is_comparison:
                entity_queries = extract_comparison_subqueries(retrieval_query)
                variants = entity_queries if entity_queries else generate_query_variants(retrieval_query, self.llm, n=2)
            else:
                variants = generate_query_variants(retrieval_query, self.llm, n=2)
            all_queries = [retrieval_query] + variants
            per_query_k = max(top_k, 10)
            results_lists = [
                self.retriever.hybrid_search(q, k=per_query_k, system_type=system_type_filter)
                for q in all_queries
            ]
            candidates = reciprocal_rank_fusion(results_lists)
            # Fallback: if filter yielded too few candidates, retry without filter
            if len(candidates) < 3 and system_type_filter:
                results_lists = [
                    self.retriever.hybrid_search(q, k=per_query_k)
                    for q in all_queries
                ]
                candidates = reciprocal_rank_fusion(results_lists)
            results = self.reranker.rerank(question, candidates, top_k=top_k)
            # Use rerank_score as final_score so source display is consistent with single-query path
            for r in results:
                if "rerank_score" in r:
                    r["final_score"] = r["rerank_score"]
        else:
            # Single-query path with metadata routing
            results = self.retriever.hybrid_search(
                retrieval_query, k=top_k, reranker=self.reranker,
                system_type=system_type_filter,
            )
            # Fallback: if filter too narrow, retry against full corpus
            if len(results) < 3 and system_type_filter:
                results = self.retriever.hybrid_search(
                    retrieval_query, k=top_k, reranker=self.reranker,
                )

        # 5. Build context string — larger budget for comparison queries (more entities to cover)
        context_budget = 3500 if is_comparison else 2000
        context = self.retriever.build_context(results, max_tokens=context_budget)

        # 6. Generate — always use original question for natural response
        answer = self.llm.generate(
            query=question,
            context=context,
            history=history,
        )

        # 7. Collect source metadata for citation
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
