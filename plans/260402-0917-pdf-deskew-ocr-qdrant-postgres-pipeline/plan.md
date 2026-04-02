# PDF Deskew + OCR + Qdrant + PostgreSQL Pipeline

## Overview
Upgrade PDF processing pipeline: add Warp Affine deskewing, improve DeepSeek OCR on Colab, migrate vector storage from MongoDB to Qdrant, add PostgreSQL for document metadata.

## Current State
- **PDF extraction**: PyMuPDF (text) + DeepSeek OCR API (image-based) — `src/scraper/pdf_extractor.py`
- **Embeddings**: `dangvantuan/vietnamese-embedding` (768-dim) — `src/embedding/embedder.py`
- **Vector store**: MongoDB Atlas `$vectorSearch` — `src/embedding/mongo_setup.py`, `src/rag/retriever.py`
- **RAG**: HybridRetriever (vector + keyword) → Gemini LLM — `src/rag/pipeline.py`
- **Chunking**: recursive paragraph/sentence split — `src/embedding/chunker.py`

## Target State
- **PDF preprocessing**: OpenCV Warp Affine deskew → denoise → binarize → DeepSeek OCR
- **Vector store**: Qdrant (replaces MongoDB Atlas vector search)
- **Metadata store**: PostgreSQL (document dates, types, numbers)
- **OCR service**: Google Colab notebook with GPU for DeepSeek OCR
- **RAG**: Updated retriever using Qdrant + PostgreSQL hybrid

## Phases

| # | Phase | Status | Priority | Effort |
|---|-------|--------|----------|--------|
| 1 | [PDF Image Preprocessing (Warp Affine)](phase-01-pdf-image-preprocessing.md) | Complete | High | Medium |
| 2 | [DeepSeek OCR Colab Notebook](phase-02-deepseek-ocr-colab.md) | Complete | High | Medium |
| 3 | [PostgreSQL Metadata Storage](phase-03-postgresql-metadata.md) | Complete | High | Medium |
| 4 | [Qdrant Vector Store Migration](phase-04-qdrant-vector-store.md) | Complete | High | High |
| 5 | [Update RAG Pipeline](phase-05-update-rag-pipeline.md) | Complete | High | Medium |
| 6 | [Integration & Testing](phase-06-integration-testing.md) | Complete | Medium | Medium |

## Dependencies
```
Phase 1 → Phase 2 (deskewed images fed to OCR)
Phase 3 → Phase 5 (metadata store ready for retriever)
Phase 4 → Phase 5 (vector store ready for retriever)
Phase 5 → Phase 6 (pipeline complete for testing)
```

## Key Decisions
- **Qdrant over MongoDB**: Better vector search performance, native filtering, simpler setup
- **PostgreSQL for metadata**: Structured queries on dates/types, relational integrity
- **Shared ID strategy**: `chunk_id = "{doc_id}_{page}_{chunk_idx}"` used in both Qdrant payload and PostgreSQL FK
- **Colab for OCR**: Free GPU for DeepSeek model inference, ngrok tunnel for API access
