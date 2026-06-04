# Phase 03 — Build file .docx

**Priority:** High | **Status:** Pending | **Effort:** M | **Depends:** Phase 02

## Overview
Tạo `Bao_Cao_Do_An_UIT_Chatbot.docx` ở repo root từ drafts markdown bằng python-docx.

## Steps
1. Viết script `tests/evaluation/results/` KHÔNG — đặt script tạm trong plan dir hoặc dùng inline python: parse drafts md → docx
   - Heading 1/2/3 map sang styles Heading 1/2/3
   - Bảng md → docx table (style 'Table Grid')
   - Placeholder hình: đoạn căn giữa, in nghiêng
2. Font Times New Roman 13, giãn dòng 1.5 (chuẩn báo cáo VN); mở style từ template gốc nếu copy được (dùng template docx gốc làm base: `Document('Bao_Cao_Do_An_Chatbot_2105.docx')` KHÔNG sửa file gốc — tạo Document mới, set styles thủ công)
3. Thêm trang Tóm tắt đầu; mục lục: chèn TOC field hoặc ghi chú "[Cập nhật mục lục: References → Update Table]"
4. Verify: mở lại bằng python-docx đếm headings khớp outline; báo user mở kiểm tra

## Success Criteria
- File docx mới mở được, đủ chương mục, bảng hiển thị đúng, file gốc không bị sửa
