# CoT Prompting Implementation Plan

**Status:** Todo  
**Priority:** Medium  
**Effort:** Small (1 file, prompt engineering only)

## Goal

Add Chain-of-Thought (CoT) reasoning instructions to the LLM system prompt so Gemini thinks step-by-step before answering complex student queries.

## Phases

| # | Phase | Status |
|---|-------|--------|
| 1 | [Add CoT to system prompt](./phase-01-cot-system-prompt.md) | Todo |

## Files Affected

- `src/rag/llm_client.py` — only file to modify

## Success Criteria

- LLM produces structured reasoning for multi-step queries (tín chỉ, điều kiện tốt nghiệp, so sánh ngành)
- No regression on chitchat/simple queries (CoT skipped via pipeline fast-path)
- Response quality subjectively improves for complex factual questions
