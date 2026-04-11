"""Streamlit chat UI — gọi RAG pipeline trực tiếp (không cần API server).

Usage:
    streamlit run src/frontend/streamlit_app.py
"""

import os
import sys

import streamlit as st
from dotenv import load_dotenv

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
)

# Khởi tạo bảng chat nếu chưa có (idempotent)
try:
    init_chat_schema()
except Exception:
    pass  # Không block app nếu DB chưa sẵn sàng

st.set_page_config(
    page_title="UIT - Chatbot Tư Vấn Đào Tạo Từ Xa",
    page_icon="🎓",
    layout="centered",
)

st.title("🎓 Chatbot Tư Vấn Đào Tạo Từ Xa")
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
                answer = result["answer"]

                # Thêm trích dẫn nguồn
                sources = result.get("sources", [])
                if sources:
                    answer += "\n\n---\n**Nguồn tham khảo:**"
                    for s in sources[:3]:
                        answer += (
                            f"\n- `{s['source']}` trang {s['page']}"
                            f" (score: {s['score']})"
                        )

                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

                # Lưu assistant response vào DB
                if st.session_state.session_id:
                    try:
                        save_message(
                            st.session_state.session_id,
                            "assistant",
                            answer,
                            sources=result.get("sources"),
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
        st.session_state.messages = []
        st.session_state.session_id = None  # session mới sẽ tạo lazy khi chat
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
            )

            selected_id = options[selected]
            if selected_id == "__new__":
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
