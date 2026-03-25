# Phase 5: Streamlit Frontend

## Priority: High | Status: ✅ | Effort: Medium

## Overview
Replace Gradio frontend with Streamlit. Keep same functionality: chat interface, source citations, Word export.

## Files to Modify
- `src/frontend/gradio-app.py` → replace with `src/frontend/streamlit-app.py`
- `requirements.txt` — replace `gradio` with `streamlit`
- `.env.example` — update port config

## Files to Create
- `src/frontend/streamlit-app.py`

## Files to Delete
- `src/frontend/gradio-app.py` (after Streamlit is working)
- `src/frontend/index.html` (Gradio-specific)

## Implementation Steps

### 1. Create `streamlit-app.py`

```python
"""Streamlit chat UI for UIT Chatbot Tư Vấn Đào Tạo Từ Xa."""
import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="UIT - Chatbot Đào Tạo Từ Xa", page_icon="🎓", layout="centered")

st.title("🎓 Chatbot Tư Vấn Đào Tạo Từ Xa")
st.caption("Trường Đại học Công nghệ Thông tin — ĐHQG TP.HCM (UIT)")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Đặt câu hỏi về đào tạo từ xa..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm thông tin..."):
            try:
                resp = requests.post(
                    f"{API_URL}/chat",
                    json={"message": prompt, "history": st.session_state.messages[:-1]},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data["answer"]

                # Show sources
                sources = data.get("sources", [])
                if sources:
                    answer += "\n\n---\n**Nguồn tham khảo:**"
                    for s in sources[:3]:
                        answer += f"\n- `{s['source']}` trang {s['page']} (score: {s['score']})"

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as exc:
                error_msg = f"Lỗi kết nối: {exc}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar: export + examples
with st.sidebar:
    st.header("Câu hỏi mẫu")
    examples = [
        "Điều kiện tuyển sinh đào tạo từ xa?",
        "Chương trình đào tạo từ xa khóa 2024?",
        "Quy trình chuyển từ chính quy sang từ xa?",
        "Học phí đào tạo từ xa bao nhiêu?",
        "Quy chế đào tạo từ xa như thế nào?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["_example"] = ex
            st.rerun()

    st.divider()
    st.header("Xuất báo cáo")
    report_type = st.selectbox("Loại", ["chat", "technical"])
    if st.button("📄 Xuất Word (.docx)"):
        if st.session_state.messages:
            try:
                resp = requests.post(
                    f"{API_URL}/export",
                    json={"history": st.session_state.messages, "report_type": report_type},
                    timeout=15,
                )
                resp.raise_for_status()
                st.download_button("Tải file", resp.content, "bao-cao.docx")
            except Exception as exc:
                st.error(f"Lỗi: {exc}")

    st.divider()
    st.caption("Đồ án tốt nghiệp — UIT 2026")
    st.caption("Nguồn: daa.uit.edu.vn | LLM: Gemini API")
```

### 2. Update `requirements.txt`
- Remove: `gradio==5.*`
- Add: `streamlit==1.*`

### 3. Update `.env.example`
- Remove: `GRADIO_PORT=7860`
- Add: `STREAMLIT_PORT=8501`

### 4. Handle example questions click
Use session state `_example` key — when set, auto-submit as chat input on rerun.

## Todo
- [x] Create streamlit-app.py with chat interface
- [x] Add sidebar with examples and export
- [x] Update requirements.txt
- [x] Update .env.example
- [x] Remove gradio-app.py and index.html
- [x] Test full chat flow with Streamlit
- [x] Deploy command: `streamlit run src/frontend/streamlit-app.py`

## Success Criteria
- Chat interface works with message history
- Source citations displayed
- Word export works
- Example questions clickable from sidebar
- Responsive layout

## Risk
- Streamlit reruns entire script on interaction — use session_state correctly
- Chat message streaming not as smooth as Gradio — acceptable for MVP
