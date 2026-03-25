# Code Review: daa.uit.edu.vn Refactor

**Date:** 2026-03-25
**Scope:** 10 files — scraper, PDF extractor, text cleaner, LLM client, Streamlit UI, API, env, requirements
**Focus:** Correctness, security, consistency, missing changes

## Overall Assessment

Refactoring is structurally complete — class renames, domain retargeting, dependency swaps all done. However, there is one **critical** runtime-breaking issue and several **high** priority stale references.

---

## CRITICAL Issues

### 1. Hyphenated filenames break ALL Python imports

**Every** `src/` module uses kebab-case filenames (`web-scraper.py`, `llm-client.py`, `pdf-extractor.py`, `text-cleaner.py`, `run-pipeline.py`) but **every** import uses underscore convention (`from src.scraper.web_scraper import ...`).

Python cannot import modules with hyphens via dot notation. **No import in the project works at runtime.**

Confirmed by test:
```
$ python3 -c "from src.scraper.web_scraper import UitDaaScraper"
ModuleNotFoundError: No module named 'src.scraper.web_scraper'
```

**Affected imports (all broken):**
- `src/scraper/run-pipeline.py` lines 10-12: `web_scraper`, `pdf_extractor`, `text_cleaner`
- `src/api/routes.py` lines 15-18: `word_exporter`, `llm_client`, `pipeline`, `retriever`
- `src/api/main.py` line 10: `routes` (this one works — `routes.py` has no hyphen)
- `src/rag/pipeline.py` lines 5-6: `retriever`, `llm_client`
- `src/scraper/qa-generator.py` lines 16, 22: `qa_templates`, `qa_validator`
- `src/embedding/ingest-pipeline.py` lines 19-21: `chunker`, `embedder`, `mongo_setup`

**Fix:** Rename ALL `.py` files from kebab-case to snake_case, OR add `__init__.py` re-exports with `importlib`. Snake_case rename is strongly preferred — it follows PEP 8 and is the simplest fix.

Files to rename:
```
src/scraper/web-scraper.py       -> web_scraper.py
src/scraper/pdf-extractor.py     -> pdf_extractor.py
src/scraper/text-cleaner.py      -> text_cleaner.py
src/scraper/run-pipeline.py      -> run_pipeline.py
src/scraper/qa-generator.py      -> qa_generator.py
src/scraper/qa-templates.py      -> qa_templates.py
src/scraper/qa-validator.py      -> qa_validator.py
src/rag/llm-client.py            -> llm_client.py
src/frontend/streamlit-app.py    -> streamlit_app.py
src/api/word-exporter.py         -> word_exporter.py
src/embedding/ingest-pipeline.py -> ingest_pipeline.py (if exists)
```

---

## HIGH Priority Issues

### 2. Stale `ctsv.uit.edu.vn` references in production code

| File | Line | Content |
|------|------|---------|
| `src/api/word-exporter.py` | 69 | `"truy cap website ctsv.uit.edu.vn."` |
| `src/api/word-exporter.py` | 111-112 | `"Ollama (OpenAI-compatible API)"`, `"Gradio"` in tech report table |
| `models/Modelfile` | 6 | `"truy cap ctsv.uit.edu.vn"` |
| `data/training/seed_pairs.json` | 6, 20, 27, 41, 69 | Multiple `ctsv.uit.edu.vn` references in training data |

**Fix for word-exporter.py:**
- Line 69: Change `ctsv.uit.edu.vn` to `daa.uit.edu.vn`
- Line 111: Change `"Ollama (OpenAI-compatible API)"` to `"Google Gemini API"`
- Line 112: Change `"Gradio"` to `"Streamlit"`

**Fix for Modelfile:** Change `ctsv.uit.edu.vn` to `daa.uit.edu.vn`. Note: This Modelfile references Ollama (`FROM ./uit-chatbot-q4_k_m.gguf`) which is now deprecated. Consider removing or documenting as legacy.

**Fix for seed_pairs.json:** Update all `ctsv.uit.edu.vn` to `daa.uit.edu.vn`.

### 3. CORS wildcard in production

`src/api/main.py` line 30: `allow_origins=["*"]`

The inline comment says "Restrict to specific origins in production" — good awareness, but this should be env-var-driven:
```python
allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")
```

---

## MEDIUM Priority Issues

### 4. LLM client error handling returns error details to user

`src/rag/llm-client.py` line 84: `return f"Loi he thong: {exc}"` — leaks exception details (potentially API keys, internal URLs) to the end user.

**Fix:** Log the full error, return a generic message:
```python
import logging
logger = logging.getLogger(__name__)
# ...
except Exception as exc:
    logger.error("Gemini API error: %s", exc, exc_info=True)
    return "Xin loi, he thong dang gap su co. Vui long thu lai sau."
```

### 5. Streamlit app usage comment is wrong

`src/frontend/streamlit-app.py` line 4: `streamlit run src/frontend/streamlit-app.py`

The correct Streamlit command is `streamlit run`, not `streamlit run`. But the filename has hyphens which Streamlit handles fine — just ensure the path is correct after any rename.

### 6. No timeout/retry on DeepSeek OCR calls

`src/scraper/pdf-extractor.py` — sends potentially large base64 images to a Colab endpoint with a 120s timeout but no retry logic. Colab endpoints are flaky (session disconnects, ngrok URL changes).

**Recommendation:** Add retry with backoff for transient errors (connection reset, 5xx):
```python
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
```

### 7. Deduplication is O(n^2)

`src/scraper/text-cleaner.py` `deduplicate_chunks()` — compares every chunk against all previous chunks with Jaccard. Acceptable for small datasets (<1000 chunks) but will be slow at scale.

Not blocking for current scope.

---

## LOW Priority

### 8. `queue.pop(0)` in BFS is O(n)

`src/scraper/web-scraper.py` line 122: `queue.pop(0)` — use `collections.deque` for O(1) popleft.

### 9. Missing `__init__.py` in some directories

Verify all `src/` subdirectories have `__init__.py` for proper package resolution.

---

## Positive Observations

- Clean separation of concerns across modules
- Proper `robots.txt` checking in scraper
- Good PDF classification (text-native vs image-based) with fallback
- Gemini system prompt correctly references `daa.uit.edu.vn`
- `.env.example` has no hardcoded secrets
- `requirements.txt` properly pins major versions
- Streamlit UI includes error handling and source citations
- Word export supports both chat and technical report types

---

## Recommended Actions (Priority Order)

1. **[CRITICAL]** Rename all `.py` files from kebab-case to snake_case
2. **[HIGH]** Update `word-exporter.py` stale references (ctsv, Ollama, Gradio)
3. **[HIGH]** Update `models/Modelfile` domain reference
4. **[HIGH]** Update `data/training/seed_pairs.json` domain references
5. **[HIGH]** Make CORS origins configurable via env var
6. **[MEDIUM]** Sanitize error messages in LLM client
7. **[LOW]** Use `deque` in BFS crawler

## Unresolved Questions

- Is `models/Modelfile` still needed? It references Ollama which is being replaced by Gemini.
- Are the `data/training/seed_pairs.json` entries still used for fine-tuning, or were they only for the Ollama model? If Gemini-only now, this file may be obsolete.
- The `notebooks/deepseek-ocr-colab.ipynb` was not reviewed in detail — should be checked for hardcoded URLs or API keys.
