"""Streamlit chat UI — gọi RAG pipeline trực tiếp (không cần API server).

Usage:
    streamlit run src/frontend/streamlit_app.py
"""

import os
import re
import sys

import streamlit as st
from dotenv import load_dotenv


def _fix_markdown_table(text: str) -> str:
    """Fix single-line markdown tables — LLM sometimes concatenates all rows on one line."""
    # Also handle literal \\n sequences the LLM might emit
    text = text.replace("\\n", "\n")

    def split_table_line(line: str) -> str:
        if not line.startswith("|") or len(line) < 120:
            return line

        # Strategy: find all table rows using regex
        # A row is: starts with |, ends with |, contains at least one cell
        # Try to extract complete rows by finding | ... | patterns
        # that contain the same number of columns as the header

        # Count columns using the separator row (| --- | --- |) as anchor
        sep_idx = line.find('| ---')
        if sep_idx < 0:
            sep_idx = line.find('|---')
        if sep_idx < 0:
            return line
        header = line[:sep_idx]
        num_cols = header.count('|') - 1
        if num_cols < 2:
            return line

        # Extract all rows: each row has exactly num_cols cells
        # Pattern: | cell | cell | ... | (num_cols cells)
        cell = r'[^|]*'
        row_pattern = r'\|' + (r'[^|]+\|' * num_cols)
        rows = re.findall(row_pattern, line)
        if len(rows) <= 1:
            return line
        return "\n".join(rows)

    lines = text.split("\n")
    return "\n".join(split_table_line(ln) for ln in lines)

load_dotenv()

# Đảm bảo import từ project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.storage.chat_history_store import (
    init_chat_schema,
    create_session,
    save_message,
    update_session_title,
    get_session_messages,
    list_sessions,
    delete_all_sessions,
)

# Khởi tạo bảng chat nếu chưa có (idempotent)
try:
    init_chat_schema()
except Exception:
    pass  # Không block app nếu DB chưa sẵn sàng

st.set_page_config(
    page_title="UIT - Chatbot tư vấn đào tạo từ xa",
    page_icon="🎓",
    layout="centered",
)

st.title("🎓 Chatbot tư vấn đào tạo từ xa")
st.caption("Trường Đại học Công nghệ Thông tin — ĐHQG TP.HCM (UIT)")


@st.cache_resource(show_spinner="Đang khởi động hệ thống RAG...")
def load_rag():
    """Khởi tạo RAG pipeline một lần duy nhất, cache lại cho session."""
    from src.rag.pipeline import RAGPipeline
    return RAGPipeline()


# Khởi tạo RAG (cached — chỉ load model 1 lần)
try:
    rag = load_rag()
except Exception as e:
    st.error(f"Lỗi khởi động RAG: {e}")
    st.stop()

# ------------------------------------------------------------------
# Session state — session_id được tạo lazy khi user gửi tin đầu tiên
# ------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = None  # chưa tạo, chờ tin nhắn đầu tiên

if "messages" not in st.session_state:
    st.session_state.messages = []

if "radio_key" not in st.session_state:
    st.session_state.radio_key = 0

if "_cleared" not in st.session_state:
    st.session_state["_cleared"] = False

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Đặt câu hỏi về đào tạo từ xa...")

if prompt:
    # Hiển thị câu hỏi người dùng
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Tạo session DB lazy khi user gửi tin đầu tiên
    if st.session_state.session_id is None:
        try:
            st.session_state.session_id = create_session(prompt[:60])
        except Exception:
            pass

    # Lưu user message vào DB
    if st.session_state.session_id:
        try:
            save_message(st.session_state.session_id, "user", prompt)
        except Exception:
            pass

    # Tạo câu trả lời
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm thông tin..."):
            try:
                result = rag.query(
                    question=prompt,
                    history=st.session_state.messages[:-1],
                    top_k=5,
                )
                answer = _fix_markdown_table(result["answer"])
                sources = result.get("sources", [])

                # Build sources markdown separately to avoid breaking table rendering
                sources_md = ""
                if sources:
                    seen_sources: set = set()
                    lines = []
                    for s in sources:
                        src = s["source"]
                        if src in seen_sources or len(lines) >= 3:
                            continue
                        seen_sources.add(src)
                        if src.lower().endswith(".docx"):
                            display = src.removesuffix(".docx")
                            lines.append(f"- 📄 {display} (score: {s['score']})")
                        else:
                            lines.append(
                                f"- `{src}` trang {s['page']} (score: {s['score']})"
                            )
                    if lines:
                        sources_md = "**Nguồn tham khảo:**\n" + "\n".join(lines)

                # Render answer and sources in separate st.markdown calls
                st.markdown(answer)
                if sources_md:
                    st.divider()
                    st.markdown(sources_md)

                # Store combined for history/DB
                full_content = answer + (f"\n\n---\n{sources_md}" if sources_md else "")
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_content}
                )

                # Lưu assistant response vào DB
                if st.session_state.session_id:
                    try:
                        save_message(
                            st.session_state.session_id,
                            "assistant",
                            full_content,
                            sources=sources,
                        )
                    except Exception:
                        pass

            except Exception as exc:
                error_msg = f"Lỗi: {exc}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )

# ------------------------------------------------------------------
# Sidebar: xóa lịch sử + danh sách sessions
# ------------------------------------------------------------------
with st.sidebar:
    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        try:
            delete_all_sessions()
        except Exception:
            pass
        st.session_state.messages = []
        st.session_state.session_id = None
        st.session_state["_cleared"] = True
        st.session_state.radio_key += 1
        st.rerun()

    # Danh sách các cuộc hội thoại cũ
    st.divider()
    st.header("Lịch sử hội thoại")

    switch_to_session = None
    try:
        sessions = list_sessions(limit=20)
        if sessions:
            # Thêm option "Hội thoại mới" ở đầu để cho phép không chọn session nào
            options = ["__new__"] + [s["id"] for s in sessions]
            labels = ["＋ Hội thoại mới"] + [s["title"] or "Hội thoại mới" for s in sessions]

            # Tìm index session hiện tại
            current_sid = st.session_state.get("session_id")
            current_idx = 0  # default: "Hội thoại mới"
            if current_sid:
                for i, s in enumerate(sessions):
                    if s["id"] == current_sid:
                        current_idx = i + 1  # +1 vì có option đầu
                        break

            selected = st.radio(
                "Chọn hội thoại:",
                range(len(options)),
                index=current_idx,
                format_func=lambda i: labels[i],
                label_visibility="collapsed",
                key=f"session_radio_{st.session_state.radio_key}",
            )

            selected_id = options[selected]
            # Bỏ qua switch nếu vừa nhấn Xóa (tránh radio cũ tự load lại session)
            if st.session_state.get("_cleared"):
                st.session_state["_cleared"] = False
            elif selected_id == "__new__":
                if st.session_state.get("session_id") is not None:
                    switch_to_session = ("__new__", [])
            elif selected_id != st.session_state.get("session_id"):
                msgs = get_session_messages(selected_id)
                switch_to_session = (selected_id, msgs)
    except Exception as e:
        st.error(f"Lỗi: {e}")

    # Thực hiện rerun ngoài try-except để RerunException không bị bắt lại
    if switch_to_session:
        sid, msgs = switch_to_session
        if sid == "__new__":
            st.session_state.session_id = None
            st.session_state.messages = []
        else:
            st.session_state.session_id = sid
            st.session_state.messages = [
                {"role": m["role"], "content": m["content"]} for m in msgs
            ]
        st.rerun()

    st.divider()
    st.caption("Đồ án tốt nghiệp — UIT 2026")
    st.caption("LLM: Gemini API | Vector DB: Qdrant")
