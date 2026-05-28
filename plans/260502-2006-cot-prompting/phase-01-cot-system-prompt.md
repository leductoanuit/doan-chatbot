# Phase 01 — Add CoT to System Prompt

**Status:** Todo  
**File:** `src/rag/llm_client.py`

## Overview

Insert a CoT reasoning block into `_DEFAULT_SYSTEM_PROMPT`. Gemini sẽ được hướng dẫn suy luận từng bước trước khi đưa ra câu trả lời cuối, đặc biệt cho các câu hỏi phức tạp (tín chỉ, điều kiện, so sánh ngành).

## Implementation

Thêm section sau vào `_DEFAULT_SYSTEM_PROMPT` (trước "Quy tắc trả lời"):

```
Quy trình suy luận (Chain-of-Thought):
Trước khi trả lời, hãy tự suy nghĩ theo các bước sau (KHÔNG hiển thị bước suy luận ra ngoài, chỉ hiển thị câu trả lời cuối):
1. Xác định loại câu hỏi: đơn giản (chào hỏi, định nghĩa) hay phức tạp (tính toán tín chỉ, điều kiện xét tốt nghiệp, so sánh chương trình)?
2. Với câu hỏi phức tạp: liệt kê các thông tin cần thiết từ tài liệu tham khảo.
3. Áp dụng logic: tính toán, so sánh, hoặc tổng hợp từng điều kiện một.
4. Kiểm tra lại: câu trả lời có đủ thông tin, không mâu thuẫn với tài liệu không?
5. Đưa ra câu trả lời cuối cùng theo "Quy tắc trả lời".
```

## Notes

- CoT là **internal reasoning** — không xuất hiện trong output (đây là zero-shot CoT).
- Simple queries (chitchat) đã được bypass ở `pipeline.py` trước khi gọi LLM, nên không bị ảnh hưởng.
- Không cần thêm LLM call, không thay đổi architecture.

## Todo

- [ ] Thêm CoT block vào `_DEFAULT_SYSTEM_PROMPT` trong `llm_client.py`
- [ ] Test thủ công với 2-3 câu hỏi phức tạp
- [ ] Verify câu hỏi đơn giản không bị chậm hơn đáng kể
