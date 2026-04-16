"""Scraper for citd.edu.vn — distance learning (he tu xa) regulations and documents.

Reuses UitDaaScraper (Playwright BFS crawler) with citd.edu.vn seed URLs.
Output paths are relative to project root (two levels up from this file).

Output:
  data/raw/html/citd_pages.json   -- scraped page text and PDF links
  data/raw/pdfs/                  -- downloaded PDF files

Usage (from project root):
    python src/scraper/citd-scraper.py [--max-pages N] [--no-pdf]
"""

import argparse
import json
import os
from pathlib import Path

from web_scraper import UitDaaScraper


BASE_URL = "https://www.citd.edu.vn"

# Seed URLs covering all main sections and sub-menus on citd.edu.vn
SEED_URLS = [
    # Homepage
    "https://www.citd.edu.vn/",
    # Gioi thieu
    "https://www.citd.edu.vn/gioi-thieu/",
    # Dao tao - main sections
    "https://www.citd.edu.vn/chuyen-muc/dao-tao/",
    "https://www.citd.edu.vn/lich-hoc/",
    "https://www.citd.edu.vn/ke-hoach-nam-hoc/",
    "https://www.citd.edu.vn/chuyen-muc/dao-tao/thong-bao-chung/",
    "https://www.citd.edu.vn/chuyen-muc/dao-tao/thong-bao-hoc-vu/",
    # CTDT - CNTT
    "https://www.citd.edu.vn/chuyen-muc/dao-tao/cong-nghe-thong-tin/",
    "https://www.citd.edu.vn/chuyen-muc/dao-tao/cu-nhan-van-bang-1/",
    "https://www.citd.edu.vn/chuyen-muc/dao-tao/cu-nhan-van-bang-2/",
    "https://www.citd.edu.vn/chuyen-muc/dao-tao/hoan-chinh-dai-hoc/",
    # CTDT - Tri tue nhan tao
    "https://www.citd.edu.vn/chuyen-muc/dao-tao/tri-tue-nhan-tao/",
    "https://www.citd.edu.vn/chuyen-muc/dao-tao/tri-tue-nhan-tao/cu-nhan-van-bang-1-tri-tue-nhan-tao/",
    "https://www.citd.edu.vn/chuyen-muc/dao-tao/tri-tue-nhan-tao/cu-nhan-van-bang-2-tri-tue-nhan-tao/",
    "https://www.citd.edu.vn/chuyen-muc/dao-tao/tri-tue-nhan-tao/lien-thong-dai-hoc/",
    # Chung chi - van bang
    "https://www.citd.edu.vn/chuyen-muc/chung-chi-van-bang/",
    "https://www.citd.edu.vn/van-ban-phap-ly/",
    "https://www.citd.edu.vn/ho-so-chung-chi/",
    "https://www.citd.edu.vn/quy-trinh-va-bieu-mau-chung-chi/",
    "https://www.citd.edu.vn/chuyen-muc/chung-chi-van-bang/chung-chi-cntt-co-ban/",
    "https://www.citd.edu.vn/chuyen-muc/chung-chi-van-bang/chung-chi-ung-dung-cntt-nang-cao/",
    # Tuyen sinh
    "https://www.citd.edu.vn/chuyen-muc/tuyen-sinh/",
    "https://www.citd.edu.vn/chuyen-muc/tuyen-sinh/thong-tin-chung-tuyen-sinh/",
    "https://www.citd.edu.vn/chuyen-muc/tuyen-sinh/ho-so-tuyen-sinh/",
    "https://www.citd.edu.vn/chuyen-muc/tuyen-sinh/cau-hoi-thuong-gap/",
    # Ho tro sinh vien
    "https://www.citd.edu.vn/quy-trinh-bieu-mau-cho-sinh-vien/",
    "https://www.citd.edu.vn/cong-cu-hoc-tap/",
    # Quy dinh
    "https://citd.edu.vn/quy-dinh-quy-che",
    "https://citd.edu.vn/quy-dinh",
]

# Project root is two levels up from this file (src/scraper/citd-scraper.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HTML_OUTPUT = PROJECT_ROOT / "data/raw/html/citd_pages.json"
PDF_DIR = str(PROJECT_ROOT / "data/raw/pdfs")

# Old PDF links point to beta.citd.vn (defunct). Rewrite to www.citd.vn.
_DOMAIN_REWRITES = {
    "http://beta.citd.vn": "https://www.citd.vn",
    "beta.citd.vn": "www.citd.vn",
}

# Also accept citd.vn subdomains as internal for link discovery
EXTRA_INTERNAL_NETLOCS = {"citd.edu.vn", "www.citd.edu.vn", "citd.vn", "www.citd.vn"}


def _rewrite_pdf_urls(pages: list[dict]) -> list[dict]:
    """Replace defunct beta.citd.vn PDF links with www.citd.vn equivalents."""
    for page in pages:
        for pdf in page.get("pdf_links", []):
            url: str = pdf["url"]
            for old, new in _DOMAIN_REWRITES.items():
                if old in url:
                    pdf["url"] = url.replace(old, new)
                    break
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape citd.edu.vn for tu-xa data")
    parser.add_argument("--max-pages", type=int, default=200, help="Max pages to crawl (default: 200)")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF downloads")
    args = parser.parse_args()

    scraper = UitDaaScraper(base_url=BASE_URL, seed_urls=SEED_URLS)
    pages = scraper.crawl(max_pages=args.max_pages)
    pages = _rewrite_pdf_urls(pages)

    HTML_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(HTML_OUTPUT, "w", encoding="utf-8") as fh:
        json.dump(pages, fh, ensure_ascii=False, indent=2)
    print(f"[citd-scraper] Saved {HTML_OUTPUT}")

    all_pdfs = [p for page in pages for p in page.get("pdf_links", [])]
    print(f"[citd-scraper] Found {len(all_pdfs)} PDF links across {len(pages)} pages")

    if not args.no_pdf and all_pdfs:
        scraper.download_pdfs(all_pdfs, save_dir=PDF_DIR)

    print("[citd-scraper] Done")


if __name__ == "__main__":
    main()
