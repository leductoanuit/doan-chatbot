"""Auto-generate document-metadata.json from files in data/raw/.

Scans PDF, DOCX, JSON files → extracts text snippet → asks Gemini to
classify and extract structured metadata → saves to data/processed/document-metadata.json.

Usage:
    python src/scraper/generate-document-metadata.py [--source data/raw] [--output data/processed/document-metadata.json]
    python src/scraper/generate-document-metadata.py --update   # only process new files
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

# ---------------------------------------------------------------------------
# Supported file types
# ---------------------------------------------------------------------------

SUPPORTED_EXT = {".pdf", ".docx", ".json"}


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def extract_text_pdf(path: Path, max_chars: int = 2000) -> str:
    """Extract text from first few pages of a PDF."""
    try:
        import fitz
        doc = fitz.open(str(path))
        text = ""
        for page in doc:
            text += page.get_text(sort=True)
            if len(text) >= max_chars:
                break
        doc.close()
        return text[:max_chars].strip()
    except Exception as exc:
        return f"[PDF read error: {exc}]"


def extract_text_docx(path: Path, max_chars: int = 2000) -> str:
    """Extract text from a DOCX file."""
    try:
        from docx import Document
        doc = Document(str(path))
        parts = []
        for para in doc.paragraphs:
            if para.text.strip():
                parts.append(para.text.strip())
            if sum(len(p) for p in parts) >= max_chars:
                break
        return "\n".join(parts)[:max_chars]
    except Exception as exc:
        return f"[DOCX read error: {exc}]"


def extract_text_json(path: Path, max_chars: int = 1000) -> str:
    """Return first portion of a JSON file as raw text."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read(max_chars)
    except Exception as exc:
        return f"[JSON read error: {exc}]"


def extract_snippet(path: Path) -> str:
    """Dispatch to the correct extractor based on file extension."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_text_pdf(path)
    elif ext == ".docx":
        return extract_text_docx(path)
    elif ext == ".json":
        return extract_text_json(path)
    return ""


# ---------------------------------------------------------------------------
# Gemini metadata extraction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "Bạn là trợ lý phân loại tài liệu pháp lý và học thuật của Trường ĐH Công nghệ Thông tin - ĐHQG TP.HCM. "
    "Nhiệm vụ: từ tên file và nội dung đầu tài liệu, trích xuất metadata chính xác theo JSON schema được yêu cầu."
)

_USER_TEMPLATE = """\
Tên file: {filename}
Nội dung đầu tài liệu:
---
{snippet}
---

Hãy trả về JSON với đúng các trường sau (không thêm trường khác):
{{
  "title": "Tiêu đề đầy đủ của tài liệu bằng tiếng Việt",
  "document_number": "Số văn bản (ví dụ: 28/2023/TT-BGDĐT) hoặc null nếu không có",
  "issue_date": "YYYY-MM-DD hoặc null nếu không rõ",
  "issuing_body": "Cơ quan ban hành (ví dụ: Bộ GD&ĐT, Trường ĐH Công nghệ Thông tin)",
  "document_type": "Một trong: quyet_dinh | thong_tu | huong_dan | chuong_trinh_dao_tao | thong_bao | bieu_mau",
  "system_type": "Một trong: dao_tao | chung_chi | tuyen_sinh | hanh_chinh"
}}
Chỉ trả về JSON thuần túy, không giải thích.
"""


def extract_metadata_with_gemini(model, filename: str, snippet: str) -> dict:
    """Call Gemini to extract structured metadata. Returns dict with all fields."""
    prompt = _USER_TEMPLATE.format(filename=filename, snippet=snippet or "(không có nội dung)")
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=300,
            ),
        )
        text = response.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except json.JSONDecodeError:
        print(f"  [WARN] Gemini returned non-JSON for {filename}")
        return {}
    except Exception as exc:
        print(f"  [ERROR] Gemini call failed for {filename}: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_metadata(source_dir: str, output_path: str, update_only: bool) -> None:
    source = Path(source_dir)
    output = Path(output_path)

    # Load existing metadata if updating
    existing: dict[str, dict] = {}
    if update_only and output.exists():
        with open(output, "r", encoding="utf-8") as fh:
            for entry in json.load(fh):
                existing[entry["source_file"]] = entry
        print(f"Loaded {len(existing)} existing entries")

    # Collect all supported files (exclude directories)
    all_files = sorted(
        f for f in source.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXT
    )
    print(f"Found {len(all_files)} files in {source_dir}")

    # Setup Gemini
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set")
        sys.exit(1)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        system_instruction=_SYSTEM_PROMPT,
    )

    results: list[dict] = []

    for i, file_path in enumerate(all_files, 1):
        filename = file_path.name
        print(f"\n[{i}/{len(all_files)}] {filename}")

        # Skip already-processed files in update mode
        if update_only and filename in existing:
            print("  -> already in metadata, skipping")
            results.append(existing[filename])
            continue

        snippet = extract_snippet(file_path)
        meta = extract_metadata_with_gemini(model, filename, snippet)

        entry = {
            "source_file": filename,
            "title": meta.get("title"),
            "document_number": meta.get("document_number"),
            "issue_date": meta.get("issue_date"),
            "issuing_body": meta.get("issuing_body"),
            "document_type": meta.get("document_type"),
            "system_type": meta.get("system_type"),
        }
        results.append(entry)
        print(f"  -> {entry['document_type']} | {entry['system_type']} | {entry['title'][:60] if entry['title'] else 'N/A'}")

    # Save
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(results)} entries saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-generate document metadata using Gemini")
    parser.add_argument("--source", default="data/raw", help="Directory containing source files")
    parser.add_argument("--output", default="data/processed/document-metadata.json")
    parser.add_argument("--update", action="store_true", help="Only process new files not in existing metadata")
    args = parser.parse_args()
    generate_metadata(args.source, args.output, update_only=args.update)
