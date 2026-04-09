"""PDF text extraction — extracts text from text-native PDFs using PyMuPDF.

Image-based (scanned) PDFs should be processed separately via scripts/ocr-local.py.
"""

import logging
import os
from typing import Literal

logger = logging.getLogger(__name__)


def classify_pdf(pdf_path: str) -> Literal["text_native", "image_based"]:
    """Check first 3 pages — if enough text found, classify as text_native."""
    import fitz

    doc = fitz.open(pdf_path)
    for i in range(min(3, len(doc))):
        text = doc[i].get_text().strip()
        if len(text) > 50:
            doc.close()
            return "text_native"
    doc.close()
    return "image_based"


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


def process_pdf(pdf_path: str) -> list[dict]:
    """Extract text from PDF. Only handles text-native PDFs.

    Image-based PDFs return empty list with a warning log.
    Use scripts/ocr-local.py for scanned/image PDFs.
    """
    pdf_type = classify_pdf(pdf_path)

    if pdf_type == "image_based":
        logger.warning(
            "Skipping image-based PDF '%s' — use scripts/ocr-local.py for OCR",
            os.path.basename(pdf_path),
        )
        return []

    return extract_text_native(pdf_path)


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py <path/to/file.pdf>")
        sys.exit(1)

    results = process_pdf(sys.argv[1])
    print(json.dumps(results[:2], ensure_ascii=False, indent=2))
    print(f"Total pages extracted: {len(results)}")
