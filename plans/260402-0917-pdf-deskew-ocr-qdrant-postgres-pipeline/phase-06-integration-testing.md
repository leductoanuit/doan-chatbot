# Phase 6: Integration & Testing

## Context
- After all components migrated, need end-to-end validation
- Full pipeline: PDF → deskew → OCR → chunk → embed → Qdrant + PostgreSQL → RAG query

## Overview
- **Priority**: Medium
- **Status**: Pending
- **Description**: End-to-end testing of the complete pipeline

## Requirements
### Functional
- Test full pipeline with real Vietnamese regulation PDFs (skewed + straight)
- Verify metadata stored correctly in PostgreSQL
- Verify vectors searchable in Qdrant
- Verify RAG answers use correct context

### Non-functional
- Document Colab setup steps
- Update `.env.example` with all new config vars

## Implementation Steps

1. **Environment setup test**:
   - Start Qdrant (Docker)
   - Start PostgreSQL
   - Run schema init (`postgres-metadata.py`)
   - Run collection init (`qdrant-vector-store.py`)

2. **Ingestion test**:
   - Process 3-5 sample PDFs through full pipeline
   - Verify deskewing (visual check on output images)
   - Verify OCR text quality
   - Verify chunks in Qdrant (count, payload)
   - Verify metadata in PostgreSQL (dates, types)

3. **Retrieval test**:
   - Query: "Quy chế đào tạo đại học mới nhất" → expect relevant regulation chunks
   - Query with filter: doc_type="quy_che_dao_tao" → filtered results
   - Query with date range → only docs within range

4. **API test**:
   - `/api/query` endpoint returns answers with sources
   - Health endpoint reports all services up

5. **Cleanup**:
   - Remove MongoDB dependencies from `requirements.txt` (pymongo, motor, llama-index-vector-stores-mongodb)
   - Update `.env.example`
   - Update any remaining MongoDB references

## Todo
- [ ] Test full pipeline with sample PDFs
- [ ] Verify Qdrant vector count matches expected chunks
- [ ] Verify PostgreSQL metadata matches source documents
- [ ] Test RAG query end-to-end
- [ ] Clean up MongoDB dependencies
- [ ] Update `.env.example` with new config vars

## Success Criteria
- Full pipeline runs without errors
- RAG answers are contextually relevant
- All old MongoDB code paths replaced
- `.env.example` documents all required config

## Risk Assessment
- **Data loss during migration**: Not a concern — fresh ingest from source PDFs
- **Missing env vars**: Document all required vars clearly in `.env.example`
