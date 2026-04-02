"""Batch OCR all PDFs in a directory → output JSON for ingestion pipeline.

Usage:
    python src/scraper/run-ocr-batch.py [--source data/raw] [--output data/processed/all_documents.json]
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
load_dotenv()

from src.scraper.pdf_extractor import process_pdf


def batch_process(source_dir: str, output_path: str) -> None:
    """Process all PDFs in source_dir, save extracted text chunks to JSON."""
    source = Path(source_dir)
    pdf_files = sorted(source.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {source_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files in {source_dir}")
    print(f"OCR URL: {os.getenv('DEEPSEEK_OCR_URL', 'NOT SET')}")
    print("=" * 60)

    all_chunks: list[dict] = []

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
        try:
            chunks = process_pdf(str(pdf_path))
            all_chunks.extend(chunks)
            print(f"  -> {len(chunks)} pages extracted")
        except Exception as exc:
            print(f"  -> ERROR: {exc}")

    # Save results
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"Done. {len(all_chunks)} total chunks from {len(pdf_files)} PDFs")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch OCR all PDFs")
    parser.add_argument("--source", default="data/raw", help="PDF source directory")
    parser.add_argument("--output", default="data/processed/all_documents.json", help="Output JSON path")
    args = parser.parse_args()
    batch_process(args.source, args.output)
