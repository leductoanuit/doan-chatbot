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
# Session state
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Xử lý câu hỏi mẫu từ sidebar
if "_example" in st.session_state and st.session_state["_example"]:
    prompt = st.session_state.pop("_example")
else:
    prompt = st.chat_input("Đặt câu hỏi về đào tạo từ xa...")

if prompt:
    # Hiển thị câu hỏi người dùng
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

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
            except Exception as exc:
                error_msg = f"Lỗi: {exc}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )

# ------------------------------------------------------------------
# Sidebar: câu hỏi mẫu + xóa lịch sử
# ------------------------------------------------------------------
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

    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Đồ án tốt nghiệp — UIT 2026")
    st.caption("LLM: Gemini API | Vector DB: Qdrant")
