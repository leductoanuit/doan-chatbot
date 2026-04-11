# Phase 1: PostgreSQL Schema + Storage Module

## Overview
Tạo 2 bảng mới và module storage để CRUD chat history.

## Schema

```sql
-- Mỗi session chat (1 lần mở app = 1 session)
CREATE TABLE chat_sessions (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    title       TEXT,                        -- preview câu hỏi đầu tiên (<=60 chars)
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Từng tin nhắn trong session
CREATE TABLE chat_messages (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID        NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role        TEXT        NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT        NOT NULL,
    sources     JSONB,                       -- metadata nguồn trích dẫn từ RAG
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_chat_messages_session_id ON chat_messages(session_id);
CREATE INDEX idx_chat_sessions_updated_at ON chat_sessions(updated_at DESC);
```

## File cần tạo

**`src/storage/chat-history-store.py`** (mới, ~100 LOC)

Functions:
- `init_chat_schema()` — tạo bảng nếu chưa có
- `create_session(title: str) -> str` — tạo session mới, trả UUID
- `update_session_title(session_id, title)` — cập nhật title sau tin đầu
- `save_message(session_id, role, content, sources=None)` — lưu 1 tin nhắn
- `get_session_messages(session_id) -> list[dict]` — load lại toàn bộ tin nhắn
- `list_sessions(limit=20) -> list[dict]` — danh sách sessions gần nhất

## Implementation Steps

1. Thêm `init_chat_schema()` vào `src/storage/postgres_metadata.py:init_schema()` để tự tạo bảng khi ingest
2. Tạo file `src/storage/chat-history-store.py` với các functions trên
3. Tái sử dụng `get_connection()` từ `postgres_metadata.py` (DRY)

## Todo
- [ ] Tạo `src/storage/chat-history-store.py`
- [ ] Update `init_schema()` trong `postgres_metadata.py` để tạo chat tables
- [ ] Test CRUD functions

## Success Criteria
- `init_chat_schema()` chạy không lỗi
- Lưu và đọc lại message đúng content + role + sources
