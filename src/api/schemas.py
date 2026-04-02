"""Pydantic request/response schemas for the chatbot API."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: Optional[List[Dict]] = None


class SourceInfo(BaseModel):
    source: str
    page: int
    score: float
    preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    context_used: int


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    content: str
    source: str
    score: float


class ExportRequest(BaseModel):
    history: List[Dict]
    report_type: str = Field(default="chat", pattern="^(chat|technical)$")


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    qdrant: bool
    llm: bool
