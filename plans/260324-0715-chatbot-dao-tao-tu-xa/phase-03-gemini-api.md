# Phase 3: Switch LLM to Gemini API

## Priority: High | Status: ✅ | Effort: Low

## Overview
Replace Ollama LLM client with Google Gemini API. Already have `google-generativeai` in requirements.txt.

## Files to Modify
- `src/rag/llm-client.py` — replace OpenAI/Ollama client with Gemini
- `src/rag/pipeline.py` — update import if module name changes (it uses `llm_client`)
- `.env.example` — update LLM-related env vars

## Implementation Steps

### 1. Rewrite `llm-client.py`
Replace OpenAI client with `google.generativeai`:

```python
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

_DEFAULT_SYSTEM_PROMPT = (
    "Bạn là tư vấn viên thông tin đào tạo từ xa của Trường Đại học Công nghệ Thông tin "
    "- ĐHQG TP.HCM (UIT). Hãy trả lời câu hỏi của sinh viên dựa trên thông tin được cung cấp. "
    "Nếu không có thông tin liên quan, hãy nói rằng bạn không có đủ thông tin và đề nghị "
    "sinh viên liên hệ phòng đào tạo hoặc truy cập daa.uit.edu.vn."
)

class LLMClient:
    def __init__(self, model_name=None, system_prompt=_DEFAULT_SYSTEM_PROMPT):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        genai.configure(api_key=api_key)
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.system_prompt = system_prompt
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_prompt,
        )

    def generate(self, query, context="", history=None, temperature=0.7, max_tokens=512):
        contents = []
        if history:
            for msg in history[-6:]:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({"role": role, "parts": [msg["content"]]})

        user_content = f"Thông tin tham khảo:\n{context}\n\nCâu hỏi: {query}" if context else query
        contents.append({"role": "user", "parts": [user_content]})

        try:
            response = self.model.generate_content(
                contents,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return response.text
        except Exception as exc:
            return f"Lỗi hệ thống: {exc}"

    def health_check(self):
        try:
            self.model.generate_content("ping", generation_config=genai.GenerationConfig(max_output_tokens=5))
            return True
        except Exception:
            return False
```

### 2. Update `.env.example`
- Remove: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`
- Add: `GEMINI_MODEL=gemini-2.0-flash`
- Keep: `GEMINI_API_KEY=` (already exists)

### 3. Update `requirements.txt`
- Remove: `openai==1.*`
- Keep: `google-generativeai==0.8.*`

## Todo
- [x] Rewrite llm-client.py with Gemini SDK
- [x] Update system prompt (ctsv → daa)
- [x] Update .env.example
- [x] Update requirements.txt
- [x] Test generate() with sample query
- [x] Test health_check()

## Success Criteria
- LLMClient.generate() returns Vietnamese answers via Gemini API
- Conversation history works correctly
- RAG context injection works
- Health check returns True when API key is valid

## Risk
- Gemini API rate limits on free tier — handle 429 errors with retry
- Vietnamese response quality may differ from fine-tuned Ollama model — acceptable trade-off
