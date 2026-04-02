"""PDF text extraction — auto-detects text-native vs image-based PDFs.

Uses PyMuPDF for native text PDFs; falls back to DeepSeek OCR API (Colab) for scanned/image PDFs.
"""

import base64
import os
from typing import Literal

import requests as req

from src.scraper.image_preprocessor import preprocess_for_ocr


# ---------------------------------------------------------------------------
# PDF classification
# ---------------------------------------------------------------------------

def classify_pdf(pdf_path: str) -> Literal["text_native", "image_based"]:
    """Check first 3 pages — if enough text found, classify as text_native."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    for i in range(min(3, len(doc))):
        text = doc[i].get_text().strip()
        if len(text) > 50:
            doc.close()
            return "text_native"
    doc.close()
    return "image_based"


# ---------------------------------------------------------------------------
# Native text extraction (PyMuPDF)
# ---------------------------------------------------------------------------

def extract_text_native(pdf_path: str) -> list[dict]:
    """Extract text page-by-page from a text-native PDF."""
    import fitz

    doc = fitz.open(pdf_path)
    chunks: list[dict] = []
    for page_num, page in enumerate(doc):
        text = page.get_text(sort=True).strip()
        if text:
            chunks.append({
                "content": text,
                "page": page_num + 1,
                "source": os.path.basename(pdf_path),
                "method": "pymupdf",
            })
    doc.close()
    return chunks


# ---------------------------------------------------------------------------
# OCR extraction (DeepSeek API on Colab)
# ---------------------------------------------------------------------------

def extract_text_ocr(pdf_path: str) -> list[dict]:
    """Send each PDF page as base64 image to DeepSeek OCR API on Colab."""
    import fitz

    ocr_url = os.getenv("DEEPSEEK_OCR_URL", "http://localhost:5000/ocr")
    doc = fitz.open(pdf_path)
    chunks: list[dict] = []

    for page_num, page in enumerate(doc):
        # Render at 2x zoom for better OCR quality
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_bytes = pix.tobytes("png")
        # Deskew, denoise, and binarize before OCR to improve accuracy
        img_bytes = preprocess_for_ocr(img_bytes)
        img_b64 = base64.b64encode(img_bytes).decode()

        try:
            resp = req.post(
                ocr_url,
                json={"image": img_b64},
                headers={"ngrok-skip-browser-warning": "true"},
                timeout=120,
            )
            resp.raise_for_status()
            text = resp.json().get("text", "").strip()
        except req.RequestException as exc:
            print(f"[pdf-extractor] OCR error page {page_num + 1}: {exc}")
            text = ""

        if text:
            chunks.append({
                "content": text,
                "page": page_num + 1,
                "source": os.path.basename(pdf_path),
                "method": "deepseek_ocr",
            })

    doc.close()
    return chunks


# ---------------------------------------------------------------------------
# Auto-detect and process
# ---------------------------------------------------------------------------

def process_pdf(pdf_path: str) -> list[dict]:
    """Auto-detect PDF type and extract text accordingly.

    Falls back from native -> OCR when native extraction yields < 100 chars.
    """
    pdf_type = classify_pdf(pdf_path)

    if pdf_type == "text_native":
        chunks = extract_text_native(pdf_path)
        total_chars = sum(len(c["content"]) for c in chunks)
        if total_chars < 100:
            # Sparse — likely scanned despite having some text layer
            chunks = extract_text_ocr(pdf_path)
    else:
        chunks = extract_text_ocr(pdf_path)

    return chunks


# ---------------------------------------------------------------------------
# Standalone run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python pdf-extractor.py <path/to/file.pdf>")
        sys.exit(1)

    path = sys.argv[1]
    results = process_pdf(path)
    print(json.dumps(results[:2], ensure_ascii=False, indent=2))
    print(f"Total pages extracted: {len(results)}")
