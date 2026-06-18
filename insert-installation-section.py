# -*- coding: utf-8 -*-
"""
Chen muc "Quy trinh cai dat" vao Chuong 5 trong Word,
sau muc "Cac cong nghe su dung" va truoc "Thiet ke API".
Noi dung moi to do.
"""

import sys, io, shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from docx import Document
from docx.shared import RGBColor, Inches
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

SRC = "Bao_Cao_Do_An_UIT_Chatbot_WITH_FIGS.docx"
DST = "Bao_Cao_Do_An_UIT_Chatbot_WITH_FIGS.docx"
TMP = "tmp_install.docx"
shutil.copy2(SRC, TMP)
doc = Document(TMP)


def red_para(text, style_id=None):
    p = OxmlElement("w:p")
    if style_id:
        pPr = OxmlElement("w:pPr")
        pStyle = OxmlElement("w:pStyle")
        pStyle.set(qn("w:val"), style_id)
        pPr.append(pStyle)
        p.append(pPr)
    r = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "FF0000")
    rPr.append(color)
    r.append(rPr)
    t = OxmlElement("w:t")
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    r.append(t)
    p.append(r)
    return p


def red_heading2(text):
    """Tao Heading 2 mau do."""
    p = OxmlElement("w:p")
    pPr = OxmlElement("w:pPr")
    pStyle = OxmlElement("w:pStyle")
    pStyle.set(qn("w:val"), "Heading2")
    pPr.append(pStyle)
    p.append(pPr)
    r = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "FF0000")
    rPr.append(color)
    r.append(rPr)
    t = OxmlElement("w:t")
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    r.append(t)
    p.append(r)
    return p


# Tim heading "Thiet ke API" de chen truoc no
ref_para = None
for p in doc.paragraphs:
    if p.text.strip() == "Thiết kế API":
        ref_para = p
        break

if not ref_para:
    print("Khong tim thay heading 'Thiet ke API'")
    exit(1)

ref_elem = ref_para._element

# Danh sach cac phan tu can chen (theo thu tu nguoc de addprevious hoat dong dung)
# Moi lan addprevious chen vao truoc ref_elem nen insert nguoc tu duoi len tren

blocks = [
    # --- Buoc 5: Truy cap giao dien ---
    red_para("Bước 5 — Truy cập giao diện: mở trình duyệt và truy cập "
             "http://localhost:8501 để sử dụng giao diện Streamlit. "
             "Nhập câu hỏi bằng tiếng Việt vào khung chat và nhấn Enter để nhận câu trả lời.",
             style_id="ListParagraph"),

    # --- Buoc 4: Chay ung dung ---
    red_para("Bước 4 — Khởi chạy ứng dụng: chạy lệnh docker-compose up --build "
             "để khởi động toàn bộ hệ thống. "
             "Khi Qdrant và PostgreSQL đã sẵn sàng (log báo 'started'), "
             "chạy python ingest.py để nạp dữ liệu vào vector store, "
             "sau đó chạy uvicorn main:app --reload (FastAPI) "
             "và streamlit run app.py (giao diện) trên hai terminal riêng.",
             style_id="ListParagraph"),

    # --- Buoc 3: Cau hinh bien moi truong ---
    red_para("Bước 3 — Cấu hình biến môi trường: sao chép file .env.example thành .env "
             "và điền các giá trị: GEMINI_API_KEY (lấy từ Google AI Studio), "
             "QDRANT_HOST=localhost, QDRANT_PORT=6333, "
             "POSTGRES_URL=postgresql://user:pass@localhost:5432/chatbot. "
             "Cài đặt thư viện Python: pip install -r requirements.txt.",
             style_id="ListParagraph"),

    # --- Buoc 2: Cai dat Docker ---
    red_para("Bước 2 — Cài đặt hạ tầng bằng Docker Compose: "
             "đảm bảo Docker Desktop đang chạy, sau đó chạy lệnh "
             "docker-compose up -d qdrant postgres "
             "để khởi động Qdrant (cổng 6333) và PostgreSQL 16 (cổng 5432) ở chế độ nền.",
             style_id="ListParagraph"),

    # --- Buoc 1: Yeu cau he thong ---
    red_para("Bước 1 — Yêu cầu hệ thống: Python 3.12+, Docker Desktop (Windows/macOS) "
             "hoặc Docker Engine (Linux), RAM tối thiểu 8 GB (khuyến nghị 16 GB), "
             "ổ cứng trống tối thiểu 10 GB (cho model BGE-M3 ~1.5 GB và dữ liệu vector). "
             "Không yêu cầu GPU — reranker chạy trên CPU.",
             style_id="ListParagraph"),

    # --- Mo dau muc ---
    red_para("Hệ thống được cài đặt hoàn toàn cục bộ qua Docker Compose. "
             "Các bước thực hiện như sau:"),

    # --- Heading muc moi ---
    red_heading2("Quy trình cài đặt"),
]

# Chen tung block truoc ref_elem (thu tu nguoc)
for block in blocks:
    ref_elem.addprevious(block)

doc.save(TMP)
shutil.move(TMP, DST)
print(f"[OK] Da chen muc 'Quy trinh cai dat' vao {DST}")
