#!/usr/bin/env python3
"""Clean OCR JSON: remove garbled pages and normalize text noise.

Usage:
    python3 scripts/clean-ocr-json.py                         # default paths
    python3 scripts/clean-ocr-json.py -i INPUT -o OUTPUT
"""
import argparse
import json
import re
from pathlib import Path


# ── Text normalisation ────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Light normalisation: collapse whitespace, fix repeated chars."""
    # Collapse 3+ spaces to single space
    text = re.sub(r' {3,}', ' ', text)
    # Collapse 3+ newlines to double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove lines consisting only of separator symbols
    text = re.sub(r'^\s*[_\-=\*\.]{3,}\s*$', '', text, flags=re.MULTILINE)
    # Fix doubled Vietnamese special chars
    for old, new in [('đđ', 'đ'), ('óó', 'ó'), ('àà', 'à'), ('ữữ', 'ữ')]:
        text = text.replace(old, new)
    return text.strip()


# ── Garbled page detection ────────────────────────────────────────────────────

# Characters common in valid Vietnamese/Latin text
_LONG_WORD = re.compile(r'[a-zA-Z\u00C0-\u024F\u1E00-\u1EFF]{3,}')


def _long_word_ratio(text: str) -> float:
    """Fraction of words containing ≥3 consecutive alphabetic chars (no digits/symbols).
    Garbled OCR pages have near-zero ratio; real Vietnamese text is typically 0.4+.
    """
    words = text.split()
    if not words:
        return 1.0
    return sum(1 for w in words if _LONG_WORD.search(w)) / len(words)


def is_garbled(text: str, min_long_word_ratio: float = 0.20) -> bool:
    """Return True if the page content looks like OCR garbage."""
    if not text or len(text.strip()) < 20:
        return True
    return _long_word_ratio(text) < min_long_word_ratio


# ── Main processing ───────────────────────────────────────────────────────────

def clean_documents(docs: list[dict]) -> tuple[list[dict], dict]:
    """Clean and filter documents. Returns (cleaned_docs, stats)."""
    cleaned = []
    stats = {"total": len(docs), "removed_garbled": 0, "cleaned": 0}

    for doc in docs:
        content = doc.get("content", "")

        if is_garbled(content):
            stats["removed_garbled"] += 1
            continue  # Drop the page entirely

        normalized = _normalize(content)
        if normalized != content:
            stats["cleaned"] += 1

        cleaned.append({**doc, "content": normalized})

    stats["kept"] = len(cleaned)
    return cleaned, stats


def main():
    parser = argparse.ArgumentParser(description="Clean OCR JSON documents")
    base = Path(__file__).parent.parent / "data" / "processed"
    parser.add_argument("-i", "--input", default=str(base / "all_documents_ocr.json"))
    parser.add_argument("-o", "--output", default=str(base / "all_documents_ocr_cleaned.json"))
    args = parser.parse_args()

    print(f"Reading {args.input} …")
    with open(args.input, encoding="utf-8") as f:
        docs = json.load(f)

    cleaned, stats = clean_documents(docs)

    print(f"  Total pages  : {stats['total']}")
    print(f"  Garbled (removed): {stats['removed_garbled']}")
    print(f"  Normalized   : {stats['cleaned']}")
    print(f"  Kept         : {stats['kept']}")

    print(f"Writing {args.output} …")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
