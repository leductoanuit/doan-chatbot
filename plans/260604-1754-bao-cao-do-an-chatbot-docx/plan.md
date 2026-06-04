# Plan: Viết báo cáo đồ án (.docx mới) cho UIT Distance Learning Chatbot

**Created:** 2026-06-04 | **Status:** Pending

## Goal
Tạo file Word mới `Bao_Cao_Do_An_UIT_Chatbot.docx` — báo cáo đồ án cho chatbot tư vấn đào tạo từ xa UIT (repo này), dùng `Bao_Cao_Do_An_Chatbot_2105.docx` (đề tài khác — chatbot văn hóa) làm **template cấu trúc**. Chỗ cần hình kiến trúc → chèn placeholder, user vẽ sau.

## Key Facts (from codebase scout)
- **Đề tài:** Chatbot RAG tư vấn hệ đào tạo từ xa UIT (quy chế, tuyển sinh, học phí, CTĐT, học vụ)
- **Stack:** Gemini 2.0 Flash, BGE-M3 embedding (1024d), BGE-reranker-v2-m3, Qdrant, PostgreSQL, FastAPI, Streamlit, PyMuPDF+Tesseract OCR
- **Data:** 15 tài liệu (8 PDF, 5 DOCX, 2 JSON) → ~247 chunks (512 words, overlap 100)
- **RAG pipeline:** chitchat detect → query rewrite → intent classify → multi-query (RAG Fusion) → hybrid retrieval (vector 0.7 + keyword 0.3) → rerank → context build → CoT generation
- **Eval:** RAGAS (faithfulness 0.73, relevancy 0.76, ctx precision 0.752, ctx recall 0.87, correctness 0.59); MAP/MRR/Hit@K (commits gần đây); robustness eval; latency ~2.2 phút/câu
- Eval mới nhất: `eval-summary-20260604_173359.json` — phase 2 PHẢI đọc bản này

## Phases
| # | Phase | Status | File |
|---|-------|--------|------|
| 1 | Trích cấu trúc template + thu thập số liệu | Pending | [phase-01](phase-01-extract-template-and-gather-data.md) |
| 2 | Viết nội dung 7 chương (markdown drafts) | Pending | [phase-02](phase-02-write-chapter-content.md) |
| 3 | Build file .docx + placeholder hình | Pending | [phase-03](phase-03-build-docx-output.md) |
| 4 | Review & nghiệm thu | Pending | [phase-04](phase-04-review.md) |

## Dependencies
1 → 2 → 3 → 4 (tuần tự). python-docx đã có sẵn trong môi trường.
