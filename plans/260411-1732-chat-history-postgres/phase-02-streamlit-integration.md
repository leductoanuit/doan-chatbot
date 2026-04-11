# Phase 2: Tích hợp Frontend Streamlit

## Overview
Update `src/frontend/streamlit_app.py` để tạo session khi mở app, lưu từng tin nhắn sau mỗi lượt chat, và hiển thị sidebar danh sách sessions.

## Thay đổi `streamlit_app.py`

### Session init (1 lần khi mở app)
```python
from src.storage.chat_history_store import create_session, save_message, list_sessions, get_session_messages

if "session_id" not in st.session_state:
    st.session_state.session_id = create_session()
```

### Sau mỗi lượt chat
```python
# Lưu user message
save_message(st.session_state.session_id, "user", prompt)

# Lưu assistant response + sources
save_message(st.session_state.session_id, "assistant", answer, sources=result["sources"])
```

### Sidebar — danh sách sessions
```python
with st.sidebar:
    st.header("Lịch sử hội thoại")
    sessions = list_sessions(limit=20)
    for s in sessions:
        if st.button(s["title"] or "Hội thoại mới", key=s["id"]):
            # Load lại messages của session đó
            st.session_state.session_id = s["id"]
            st.session_state.messages = get_session_messages(s["id"])
            st.rerun()
```

## File cần sửa

| File | Thay đổi |
|------|----------|
| `src/frontend/streamlit_app.py` | Thêm session init, save_message calls, sidebar |

## Todo
- [ ] Thêm session init vào session_state block
- [ ] Gọi `save_message()` sau user input
- [ ] Gọi `save_message()` sau assistant response
- [ ] Thêm sidebar list sessions
- [ ] Gọi `update_session_title()` sau tin nhắn đầu tiên của user

## Success Criteria
- Refresh page → session cũ vẫn còn trong sidebar
- Click session cũ → load lại đúng messages
- Tin nhắn mới được lưu real-time sau mỗi lượt
