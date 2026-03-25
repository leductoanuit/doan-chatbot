"""Data collection pipeline orchestrator — scrape → extract → clean → save."""

import json
import os
import sys

# Allow running from project root or src/scraper/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.scraper.web_scraper import UitDaaScraper
from src.scraper.pdf_extractor import process_pdf
from src.scraper.text_cleaner import clean_vietnamese_text, deduplicate_chunks


def run(max_pages: int = 500) -> None:
    # ------------------------------------------------------------------
    # Step 1: Crawl website
    # ------------------------------------------------------------------
    print("=== Step 1: Crawling daa.uit.edu.vn ===")
    scraper = UitDaaScraper()
    pages = scraper.crawl(max_pages=max_pages)

    os.makedirs("data/raw/html", exist_ok=True)
    with open("data/raw/html/pages.json", "w", encoding="utf-8") as fh:
        json.dump(pages, fh, ensure_ascii=False, indent=2)
    print(f"Saved {len(pages)} pages to data/raw/html/pages.json")

    # ------------------------------------------------------------------
    # Step 2: Download PDFs
    # ------------------------------------------------------------------
    print("\n=== Step 2: Downloading PDFs ===")
    all_pdf_links: list[dict] = []
    for page in pages:
        all_pdf_links.extend(page.get("pdf_links", []))

    # Deduplicate PDF URLs
    seen_urls: set[str] = set()
    unique_pdfs = []
    for p in all_pdf_links:
        if p["url"] not in seen_urls:
            seen_urls.add(p["url"])
            unique_pdfs.append(p)

    scraper.download_pdfs(unique_pdfs)
    print(f"Found {len(unique_pdfs)} unique PDF links")

    # ------------------------------------------------------------------
    # Step 3: Build chunks from HTML pages
    # ------------------------------------------------------------------
    print("\n=== Step 3: Processing HTML content ===")
    all_chunks: list[dict] = []

    for page in pages:
        text = page.get("text", "")
        if len(text) > 50:
            cleaned = clean_vietnamese_text(text, strip_stop_words=True)
            if cleaned:
                all_chunks.append({
                    "content": cleaned,
                    "source": page["url"],
                    "page": 0,
                    "method": "html",
                })

    print(f"HTML chunks: {len(all_chunks)}")

    # ------------------------------------------------------------------
    # Step 4: Extract text from PDFs
    # ------------------------------------------------------------------
    print("\n=== Step 4: Processing PDFs ===")
    pdf_dir = "data/raw/pdfs"
    if os.path.exists(pdf_dir):
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
        for filename in pdf_files:
            filepath = os.path.join(pdf_dir, filename)
            try:
                chunks = process_pdf(filepath)
                for chunk in chunks:
                    chunk["content"] = clean_vietnamese_text(chunk["content"], strip_stop_words=True)
                    if chunk["content"]:
                        all_chunks.append(chunk)
                print(f"  {filename}: {len(chunks)} pages")
            except Exception as exc:
                print(f"  ERROR {filename}: {exc}")
    else:
        print("  No PDFs directory found, skipping")

    # ------------------------------------------------------------------
    # Step 5: Deduplicate and save
    # ------------------------------------------------------------------
    print("\n=== Step 5: Deduplication & Save ===")
    all_chunks = deduplicate_chunks(all_chunks)

    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/all_documents.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(all_chunks, fh, ensure_ascii=False, indent=2)

    # Summary
    method_counts = {}
    for c in all_chunks:
        m = c.get("method", "unknown")
        method_counts[m] = method_counts.get(m, 0) + 1

    print(f"\nTotal chunks: {len(all_chunks)}")
    for method, count in method_counts.items():
        print(f"  {method}: {count}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run data collection pipeline")
    parser.add_argument("--max-pages", type=int, default=500)
    args = parser.parse_args()
    run(max_pages=args.max_pages)
