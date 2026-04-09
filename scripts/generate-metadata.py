"""Generate document-metadata.json from filenames + content — no API needed.

Uses regex pattern matching on filename and first-page text to extract:
title, document_number, issue_date, issuing_body, document_type, system_type.

Usage: python3 data/raw/generate-metadata.py
"""

import json
import re
from pathlib import Path

import fitz  # PyMuPDF
from docx import Document

RAW_DIR = Path(__file__).parent
OUTPUT_DIR = RAW_DIR.parent / "processed"
OUTPUT_FILE = OUTPUT_DIR / "document-metadata.json"

SUPPORTED_EXT = {".pdf", ".docx", ".json"}

# ---------------------------------------------------------------------------
# Regex patterns for Vietnamese legal documents
# ---------------------------------------------------------------------------

# Document number patterns (order matters — more specific first)
_DOC_NUM_PATTERNS = [
    r"(\d+/\d{4}/(?:TTLT|TT|QĐ|NĐ|CT)-[\w&/-]+)",  # 28/2023/TT-BGDĐT
    r"(\d+/\d{4}/(?:TTLT|TT|QĐ|NĐ|CT)[\w&/-]+)",    # without dash
    r"(\d+/(?:QĐ|TT|NĐ|CT)-[\w&/-]+)",               # 1499/QĐ-ĐHCNTT
    r"QĐ\s*(\d+)[/_-]",                               # QĐ256
    r"(\d+)[/_-]QĐ",                                  # 256_QD
]

# Date patterns
_DATE_PATTERNS = [
    (r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", "{2}-{1:02d}-{0:02d}"),   # DD/MM/YYYY
    (r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", "{0}-{1:02d}-{2:02d}"),   # YYYY-MM-DD
    (r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", "{2}-{1:02d}-{0:02d}"),
]

# Issuing body detection
_ISSUING_RULES = [
    (r"(?i)(BGDĐT|BỘ\s+GD[&À]ĐT|bộ giáo dục)", "Bộ GD&ĐT"),
    (r"(?i)(BTTTT|BỘ\s+TT[&À]TT|bộ thông tin)", "Bộ TT&TT"),
    (r"(?i)(TTLT.*BGDĐT.*BTTTT|TTLT.*BTTTT.*BGDĐT)", "Bộ GD&ĐT - Bộ TT&TT"),
    (r"(?i)(ĐHCNTT|UIT|đại học công nghệ thông tin)", "Trường ĐH Công nghệ Thông tin"),
    (r"(?i)(ĐHQG|đại học quốc gia)", "ĐHQG TP.HCM"),
]

# Document type detection (filename + content keywords)
_DOCTYPE_RULES = [
    (r"(?i)(quyết định|qd[-_]|[-_]qd|QĐ[-_]|[-_]QĐ)", "quyet_dinh"),
    (r"(?i)(thông tư liên tịch|TTLT)", "thong_tu"),
    (r"(?i)(thông tư|[-_]TT[-_]|/TT-)", "thong_tu"),
    (r"(?i)(chương trình đào tạo|CTĐT|cử nhân|kỹ sư)", "chuong_trinh_dao_tao"),
    (r"(?i)(thông báo|tuyển sinh|khai giảng)", "thong_bao"),
    (r"(?i)(hướng dẫn|câu hỏi thường gặp|FAQ|hồ sơ)", "huong_dan"),
    (r"(?i)(biểu mẫu|quy trình|bieu_mau|quy_trinh)", "bieu_mau"),
]

# System type detection
_SYSTYPE_RULES = [
    (r"(?i)(chứng chỉ|chung_chi|kỹ năng CNTT|sát hạch|UD.CNTT)", "chung_chi"),
    (r"(?i)(tuyển sinh|tuyen_sinh|khai giảng|hồ sơ tuyển)", "tuyen_sinh"),
    (r"(?i)(đào tạo|quy chế|chương trình|học phí|từ xa|DTTX|trực tuyến)", "dao_tao"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_text_pdf(path: Path, max_chars: int = 3000) -> str:
    """Read first 2 pages of PDF for pattern matching."""
    try:
        doc = fitz.open(str(path))
        text = ""
        for page in doc:
            text += page.get_text(sort=True)
            if len(text) >= max_chars:
                break
        doc.close()
        return text[:max_chars]
    except Exception:
        return ""


def extract_text_docx(path: Path, max_chars: int = 2000) -> str:
    """Read first paragraphs of DOCX."""
    try:
        doc = Document(str(path))
        parts = []
        for para in doc.paragraphs:
            if para.text.strip():
                parts.append(para.text.strip())
            if sum(len(p) for p in parts) >= max_chars:
                break
        return "\n".join(parts)[:max_chars]
    except Exception:
        return ""


def extract_text_json(path: Path) -> str:
    """Return title fields from JSON for pattern matching."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            titles = [item.get("title", "") for item in data[:5] if isinstance(item, dict)]
            return " ".join(titles)
        return ""
    except Exception:
        return ""


def get_snippet(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_text_pdf(path)
    elif ext == ".docx":
        return extract_text_docx(path)
    elif ext == ".json":
        return extract_text_json(path)
    return ""


def find_doc_number(text: str):
    """Try each regex pattern, return first match."""
    for pattern in _DOC_NUM_PATTERNS:
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()
    return None


def find_date(text: str):
    """Extract earliest plausible date from text."""
    for pattern, fmt in _DATE_PATTERNS:
        m = re.search(pattern, text)
        if m:
            g = [int(x) for x in m.groups()]
            try:
                result = fmt.format(*g)
                # Validate: year between 2000-2030, month 1-12, day 1-31
                parts = result.split("-")
                y, mo, d = int(parts[0]), int(parts[1]), int(parts[2])
                if 2000 <= y <= 2030 and 1 <= mo <= 12 and 1 <= d <= 31:
                    return result
            except (IndexError, ValueError):
                continue
    return None


def find_issuing_body(text: str) -> str:
    for pattern, body in _ISSUING_RULES:
        if re.search(pattern, text):
            return body
    return "Trường ĐH Công nghệ Thông tin"  # default


def find_doc_type(text: str) -> str:
    for pattern, dtype in _DOCTYPE_RULES:
        if re.search(pattern, text):
            return dtype
    return "huong_dan"  # default


def find_sys_type(text: str) -> str:
    for pattern, stype in _SYSTYPE_RULES:
        if re.search(pattern, text):
            return stype
    return "dao_tao"  # default


def build_title(filename: str, doc_num, doc_type: str) -> str:
    """Construct a readable Vietnamese title from filename clues."""
    # Remove extension and replace separators
    name = Path(filename).stem
    name = re.sub(r"[_\-]+", " ", name).strip()

    type_prefix = {
        "quyet_dinh": "Quyết định",
        "thong_tu": "Thông tư",
        "chuong_trinh_dao_tao": "Chương trình đào tạo",
        "thong_bao": "Thông báo",
        "huong_dan": "Hướng dẫn",
        "bieu_mau": "Biểu mẫu",
    }.get(doc_type, "Tài liệu")

    if doc_num:
        return f"{type_prefix} {doc_num}"
    return f"{type_prefix} – {name}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(
        f for f in RAW_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXT
    )
    print(f"Found {len(files)} files")
    print("=" * 60)

    results = []

    for i, path in enumerate(files, 1):
        filename = path.name
        print(f"\n[{i}/{len(files)}] {filename}")

        # Combine filename + content for pattern matching
        snippet = get_snippet(path)
        combined = filename + "\n" + snippet

        doc_num = find_doc_number(combined)
        date = find_date(combined)
        issuer = find_issuing_body(combined)
        doc_type = find_doc_type(combined)
        sys_type = find_sys_type(combined)
        title = build_title(filename, doc_num, doc_type)

        entry = {
            "source_file": filename,
            "title": title,
            "document_number": doc_num,
            "issue_date": date,
            "issuing_body": issuer,
            "document_type": doc_type,
            "system_type": sys_type,
        }
        results.append(entry)
        print(f"  type={doc_type} | sys={sys_type} | num={doc_num} | date={date}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Done. {len(results)} entries → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
