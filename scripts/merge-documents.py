"""Merge document metadata into chunks → all_documents_final.json.

Reads all_documents_ocr_cleaned.json + document-metadata.json (dict format)
and produces a single file where every chunk carries its own metadata.
This eliminates the join step in ingest_pipeline.py.

Usage: python3 scripts/merge-documents.py
"""

import json
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
CHUNKS_FILE = PROCESSED_DIR / "all_documents_ocr_cleaned.json"
METADATA_FILE = PROCESSED_DIR / "document-metadata.json"
OUTPUT_FILE = PROCESSED_DIR / "all_documents_final.json"

# Metadata fields to embed into each chunk
METADATA_FIELDS = ("doc_id", "title", "document_number", "issue_date",
                   "issuing_body", "document_type", "system_type")


def load_metadata(path: Path) -> dict[str, dict]:
    """Load metadata — supports both dict format (new) and array format (legacy)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        # Legacy array format → convert to dict on the fly
        return {e["source_file"]: e for e in data}
    return data  # Already dict keyed by source_file


def main():
    print(f"Loading chunks from {CHUNKS_FILE.name} …")
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"  {len(chunks)} chunks loaded")

    print(f"Loading metadata from {METADATA_FILE.name} …")
    meta_map = load_metadata(METADATA_FILE)
    print(f"  {len(meta_map)} source files in metadata")

    merged = []
    missing_meta = set()

    for chunk in chunks:
        source = chunk.get("source", "")
        meta = meta_map.get(source, {})

        if not meta:
            missing_meta.add(source)

        merged_chunk = {**chunk}
        for field in METADATA_FIELDS:
            merged_chunk[field] = meta.get(field)

        merged.append(merged_chunk)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Done. {len(merged)} chunks → {OUTPUT_FILE}")
    if missing_meta:
        print(f"WARNING: {len(missing_meta)} sources had no metadata:")
        for s in sorted(missing_meta):
            print(f"  - {s}")


if __name__ == "__main__":
    main()
