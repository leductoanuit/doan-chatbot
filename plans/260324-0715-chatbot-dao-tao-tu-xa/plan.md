# Chatbot Tư Vấn Đào Tạo Từ Xa — Implementation Plan

## Overview
Refactor existing UIT chatbot from `ctsv.uit.edu.vn` → `daa.uit.edu.vn` (đào tạo từ xa). Switch LLM from Ollama to Gemini API, OCR from PaddleOCR to DeepSeek (Colab), frontend from Gradio to Streamlit.

## Current State
- **Scraper**: BFS crawl targeting `ctsv.uit.edu.vn` — needs retargeting to `daa.uit.edu.vn`
- **PDF extractor**: PyMuPDF + PaddleOCR — needs DeepSeek OCR via Colab API
- **LLM**: Ollama (OpenAI-compatible) — needs Gemini API
- **Embedding**: `dangvantuan/vietnamese-embedding` (768-dim) — keep as-is
- **MongoDB**: Atlas vector search — keep as-is
- **Frontend**: Gradio — needs Streamlit
- **RAG pipeline**: hybrid vector+keyword search — keep as-is

## Target URLs (Seed Pages)
1. `https://daa.uit.edu.vn/09-quyet-dinh-ve-viec-ban-hanh-quy-che-dao-tao-cho-sinh-vien-he-dao-tao-tu-xa-trinh-do-dai-hoc`
2. `https://daa.uit.edu.vn/tu-xa/ctdt-khoa-2024`
3. `https://daa.uit.edu.vn/33-quy-che-tuyen-sinh-hinh-thuc-dao-tao-tu-xa-trinh-do-dai-hoc`
4. `https://daa.uit.edu.vn/34-quy-trinh-chuyen-sinh-vien-tu-hinh-thuc-dao-tao-chinh-quy-sang-hinh-thuc-dao-tao-tu-xa`
5. `https://daa.uit.edu.vn` (main page — follow internal links)

## Phases

| # | Phase | Status | Priority | Effort | File |
|---|-------|--------|----------|--------|------|
| 1 | Update Web Scraper | ✅ | High | Medium | [phase-01](phase-01-update-web-scraper.md) |
| 2 | DeepSeek OCR on Colab | ✅ | High | Medium | [phase-02](phase-02-deepseek-ocr-colab.md) |
| 3 | Switch LLM to Gemini API | ✅ | High | Low | [phase-03](phase-03-gemini-api.md) |
| 4 | Update Text Cleaner | ✅ | Medium | Low | [phase-04](phase-04-update-text-cleaner.md) |
| 5 | Streamlit Frontend | ✅ | High | Medium | [phase-05](phase-05-streamlit-frontend.md) |
| 6 | Integration & Pipeline Test | ✅ | High | Medium | [phase-06](phase-06-integration-testing.md) |

## Dependencies
```
Phase 1 (scraper) ──┐
Phase 2 (OCR)    ──┤──→ Phase 4 (cleaner) ──→ Phase 6 (integration)
Phase 3 (Gemini) ──┘
Phase 5 (Streamlit) ─────────────────────────→ Phase 6 (integration)
```

Phases 1, 2, 3, 5 can run in parallel. Phase 4 depends on 1+2. Phase 6 depends on all.

## Key Decisions
- **Keep**: MongoDB Atlas, Vietnamese embedder, chunker, hybrid retriever, RAG pipeline structure
- **Change**: Scraper target, OCR method, LLM provider, frontend framework
- **Add**: Colab notebook for DeepSeek OCR, Streamlit app
- **Remove**: Gradio dependency, PaddleOCR dependency, Ollama dependency
