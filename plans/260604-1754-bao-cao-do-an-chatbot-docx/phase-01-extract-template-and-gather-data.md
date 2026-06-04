# Phase 01 — Trích cấu trúc template + thu thập số liệu

**Priority:** High | **Status:** Pending | **Effort:** S

## Overview
Lấy đầy đủ outline + style của docx template, và số liệu chính xác từ codebase/eval để các chương không bịa số.

## Steps
1. Dump toàn bộ heading tree của `Bao_Cao_Do_An_Chatbot_2105.docx` (python-docx) — đã có sơ bộ, lưu lại thành `research/template-outline.md`
2. Đọc eval mới nhất: `tests/evaluation/results/eval-summary-20260604_173359.json` + `eval-summary-20260604_170319.json` (MAP/MRR/Hit@K, robustness) + `bao-cao-ket-qua-eval.md`
3. Xác nhận số liệu dataset: số tài liệu, số chunks (query Qdrant hoặc đọc script ingest), chunk_size/overlap thực tế trong `src/embedding/chunker.py`
4. Liệt kê API endpoints từ `src/api/` và tính năng UI từ Streamlit app
5. Ghi tổng hợp vào `research/so-lieu-bao-cao.md`

## Success Criteria
- `research/template-outline.md` + `research/so-lieu-bao-cao.md` tồn tại, số liệu khớp code thực tế

## Risks
- Lần chạy eval 20260604 có thể là metric khác (recall-based MAP/MRR) → phải đọc kỹ, không trộn lẫn với RAGAS
