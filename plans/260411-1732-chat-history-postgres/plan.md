# Plan: Lưu lịch sử chat theo session vào PostgreSQL

**Status:** Ready  
**Priority:** Medium  
**Effort:** ~2h

## Tổng quan

Hiện tại chat history chỉ tồn tại trong `st.session_state` (mất khi refresh). Plan này persist lịch sử chat vào PostgreSQL theo session, cho phép user xem lại lịch sử các cuộc hội thoại.

## Phases

| Phase | File | Status |
|-------|------|--------|
| [Phase 1](phase-01-postgres-schema.md) | Schema + storage module | ⬜ Todo |
| [Phase 2](phase-02-streamlit-integration.md) | Tích hợp frontend Streamlit | ⬜ Todo |
