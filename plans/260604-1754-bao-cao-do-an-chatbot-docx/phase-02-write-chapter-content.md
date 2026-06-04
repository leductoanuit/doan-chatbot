# Phase 02 — Viết nội dung 7 chương

**Priority:** High | **Status:** Pending | **Effort:** L | **Depends:** Phase 01

## Overview
Viết nội dung tiếng Việt học thuật cho từng chương theo outline template, thay nội dung "chatbot văn hóa" bằng "chatbot tư vấn đào tạo từ xa UIT". Lưu drafts markdown vào `drafts/chuong-0X-*.md` trong plan dir (nguồn để build docx ở phase 3).

## Chapter Mapping (template → đề tài mới)
| Chương | Nội dung mới |
|---|---|
| Tóm tắt | RAG chatbot tư vấn ĐTTX UIT, stack, kết quả chính |
| 1. Giới thiệu | Lý do (nhu cầu tư vấn ĐTTX), mục tiêu, phát biểu bài toán, đối tượng/phạm vi (quy chế, tuyển sinh, học phí, CTĐT, học vụ UIT), ý nghĩa, cấu trúc |
| 2. Tổng quan | Bài toán QA giáo dục, RAG, nghiên cứu liên quan (giữ refs phù hợp), khoảng trống |
| 3. Phương pháp | Quy trình; **[PLACEHOLDER hình kiến trúc]**; xây dựng dataset (OCR pipeline, cleaning, chunking 512w/100w); Qdrant + payload index; RAG pipeline 4 giai đoạn (tiền xử lý: chitchat/rewrite/intent; truy xuất: hybrid 0.7/0.3 + RAG Fusion; sàng lọc: rerank BGE; tổng hợp: Gemini CoT); cấu hình mô hình; prompt engineering (rewrite, multi-query, intent, CoT, anti-hallucination) |
| 4. Thực nghiệm | Tập kiểm thử 50 câu × 5 danh mục × 3 độ khó; phương pháp (RAGAS + MAP/MRR/Hit@K + robustness); kết quả số liệu từ phase 01; latency |
| 5. Cài đặt minh họa | Công nghệ; API (POST /chat, GET /search, POST /export, GET /health); UI Streamlit (session, sources, export docx); tính năng demo **[PLACEHOLDER screenshots]**; nhận xét |
| 6. Kết luận | Kết quả đạt được; hạn chế (latency ~2.2 phút, correctness 0.59, chunking word-based vượt token limit, OCR noise); hướng phát triển (streaming, caching, token-based chunking, mở rộng dữ liệu) |

## Rules
- Văn phong học thuật tiếng Việt, không markdown syntax lọt vào nội dung đoạn văn
- Số liệu chỉ lấy từ `research/so-lieu-bao-cao.md`
- Placeholder hình theo format: `[Hình X.Y: <mô tả> — sẽ bổ sung]`
- Bảng số liệu eval giữ dạng bảng

## Success Criteria
- `drafts/` có đủ file tóm tắt + 6 chương, mỗi mục heading template đều có nội dung
