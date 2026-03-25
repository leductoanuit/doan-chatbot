# Phase 6: Integration & Pipeline Test

## Priority: High | Status: ✅ | Effort: Medium

## Overview
End-to-end test of full pipeline: scrape → OCR → clean → chunk → embed → ingest → query → chat.

## Files to Modify
- `src/scraper/run-pipeline.py` — ensure all references updated
- `src/scraper/qa-generator.py` — update to use Gemini for QA generation
- `src/scraper/qa-templates.py` — update question templates for đào tạo từ xa
- `.env.example` — final review of all env vars
- `docker-compose.yml` — no changes needed (MongoDB stays)
- `models/Modelfile` — remove or update (Ollama no longer used)

## Implementation Steps

### 1. Full Pipeline Smoke Test
```bash
# Step 1: Start MongoDB
docker compose up -d

# Step 2: Run scraper
python src/scraper/run-pipeline.py --max-pages 50

# Step 3: Start DeepSeek OCR on Colab (manual)
# Copy ngrok URL to .env DEEPSEEK_OCR_URL

# Step 4: Ingest into MongoDB
python src/embedding/ingest-pipeline.py

# Step 5: Start API
python src/api/main.py

# Step 6: Start Streamlit
streamlit run src/frontend/streamlit-app.py
```

### 2. Update QA Generator
- `qa-generator.py`: Switch from Gemini `google-generativeai` to use same LLM client
- `qa-templates.py`: Update templates for đào tạo từ xa domain questions

### 3. Update Environment Config
Final `.env.example`:
```
MONGODB_URI=mongodb+srv://<user>:<password>@cluster0.xxx.mongodb.net/
MONGODB_DB=chatbot
MONGODB_COLLECTION=documents
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.0-flash
EMBEDDING_MODEL=dangvantuan/vietnamese-embedding
DEEPSEEK_OCR_URL=
API_HOST=0.0.0.0
API_PORT=8000
STREAMLIT_PORT=8501
```

### 4. Cleanup
- Remove `models/Modelfile` (Ollama-specific)
- Remove `notebooks/finetuning.ipynb` if no longer relevant
- Update footer text in Streamlit app

## Todo
- [x] Run full scrape of daa.uit.edu.vn (50 pages test)
- [x] Verify PDFs downloaded and OCR'd
- [x] Verify chunks ingested into MongoDB
- [x] Test vector search with sample queries
- [x] Test RAG chat via API
- [x] Test Streamlit UI end-to-end
- [x] Update qa-generator.py and qa-templates.py
- [x] Clean up unused files
- [x] Final .env.example review

## Success Criteria
- Full pipeline runs without errors
- At least 20+ chunks ingested from daa.uit.edu.vn
- Chat answers questions about đào tạo từ xa accurately using RAG context
- Sources properly cited in responses
- Streamlit UI functional

## Risk
- daa.uit.edu.vn may have few pages — lower max_pages
- MongoDB Atlas free tier limits — monitor document count
- Gemini API quota — implement rate limiting
