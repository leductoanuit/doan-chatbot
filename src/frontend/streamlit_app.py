"""Streamlit chat UI for UIT Chatbot Tu Van Dao Tao Tu Xa.

Usage:
    streamlit run src/frontend/streamlit-app.py
"""

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="UIT - Chatbot Dao Tao Tu Xa",
    page_icon="🎓",
    layout="centered",
)

st.title("🎓 Chatbot Tu Van Dao Tao Tu Xa")
st.caption("Truong Dai hoc Cong nghe Thong tin — DHQG TP.HCM (UIT)")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle example question click from sidebar
if "_example" in st.session_state and st.session_state["_example"]:
    prompt = st.session_state.pop("_example")
else:
    prompt = st.chat_input("Dat cau hoi ve dao tao tu xa...")

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Dang tim kiem thong tin..."):
            try:
                resp = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "message": prompt,
                        "history": st.session_state.messages[:-1],
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data["answer"]

                # Append source citations
                sources = data.get("sources", [])
                if sources:
                    answer += "\n\n---\n**Nguon tham khao:**"
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
                error_msg = f"Loi ket noi: {exc}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )

# ---------------------------------------------------------------------------
# Sidebar: example questions + Word export
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Cau hoi mau")
    examples = [
        "Dieu kien tuyen sinh dao tao tu xa?",
        "Chuong trinh dao tao tu xa khoa 2024?",
        "Quy trinh chuyen tu chinh quy sang tu xa?",
        "Hoc phi dao tao tu xa bao nhieu?",
        "Quy che dao tao tu xa nhu the nao?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["_example"] = ex
            st.rerun()

    st.divider()
    st.header("Xuat bao cao")
    report_type = st.selectbox("Loai", ["chat", "technical"])
    if st.button("📄 Xuat Word (.docx)"):
        if st.session_state.messages:
            try:
                resp = requests.post(
                    f"{API_URL}/export",
                    json={
                        "history": st.session_state.messages,
                        "report_type": report_type,
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                st.download_button(
                    "Tai file",
                    resp.content,
                    "bao-cao.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            except Exception as exc:
                st.error(f"Loi: {exc}")
        else:
            st.warning("Chua co lich su chat de xuat.")

    st.divider()
    st.caption("Do an tot nghiep — UIT 2026")
    st.caption("Nguon: daa.uit.edu.vn | LLM: Gemini API")
