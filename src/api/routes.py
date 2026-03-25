"""FastAPI route handlers — /chat, /search, /export, /health."""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.api.schemas import (
    ChatRequest, ChatResponse,
    ExportRequest,
    HealthResponse,
    SearchResult,
    SourceInfo,
)
from src.api.word_exporter import export_chat_report, export_technical_report
from src.rag.llm_client import LLMClient
from src.rag.pipeline import RAGPipeline
from src.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialise shared singletons (loaded once at startup)
_retriever = HybridRetriever()
_llm_client = LLMClient()
_rag = RAGPipeline(retriever=_retriever, llm_client=_llm_client)


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Main RAG chat endpoint — retrieve relevant docs then generate answer."""
    try:
        result = _rag.query(question=request.message, history=request.history)
        sources = [SourceInfo(**s) for s in result["sources"]]
        return ChatResponse(
            answer=result["answer"],
            sources=sources,
            context_used=result["context_used"],
        )
    except Exception as exc:
        logger.error("Chat error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Search (direct retrieval, no LLM generation)
# ---------------------------------------------------------------------------

@router.get("/search", response_model=list[SearchResult])
async def search(query: str, top_k: int = 5) -> list[SearchResult]:
    """Direct hybrid vector+keyword search, no LLM generation."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    if not (1 <= top_k <= 20):
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")
    try:
        results = _retriever.hybrid_search(query, k=top_k)
        return [
            SearchResult(
                content=r["content"][:500],
                source=r.get("metadata", {}).get("source", ""),
                score=round(r.get("final_score", 0.0), 3),
            )
            for r in results
        ]
    except Exception as exc:
        logger.error("Search error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Word export
# ---------------------------------------------------------------------------

@router.post("/export")
async def export_word(request: ExportRequest) -> FileResponse:
    """Export conversation history or technical report as .docx."""
    if not request.history:
        raise HTTPException(status_code=400, detail="history is empty")

    try:
        if request.report_type == "technical":
            filepath = export_technical_report(request.history)
            filename = "bao-cao-ky-thuat-uit-chatbot.docx"
        else:
            filepath = export_chat_report(request.history)
            filename = "bao-cao-tu-van-dao-tao.docx"

        return FileResponse(
            path=filepath,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=filename,
        )
    except Exception as exc:
        logger.error("Export error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Service health check — MongoDB and LLM availability."""
    mongo_ok = _retriever.is_healthy()
    llm_ok = _llm_client.health_check()
    status = "ok" if (mongo_ok and llm_ok) else "degraded"
    return HealthResponse(status=status, mongodb=mongo_ok, llm=llm_ok)
