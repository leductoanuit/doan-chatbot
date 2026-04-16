"""Web scraper for daa.uit.edu.vn — BFS crawl with Playwright (JS rendering)."""

import json
import os
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Page


# Seed URLs for distance learning (dao tao tu xa) pages
SEED_URLS = [
    "https://daa.uit.edu.vn",
    "https://daa.uit.edu.vn/09-quyet-dinh-ve-viec-ban-hanh-quy-che-dao-tao-cho-sinh-vien-he-dao-tao-tu-xa-trinh-do-dai-hoc",
    "https://daa.uit.edu.vn/tu-xa/ctdt-khoa-2024",
    "https://daa.uit.edu.vn/33-quy-che-tuyen-sinh-hinh-thuc-dao-tao-tu-xa-trinh-do-dai-hoc",
    "https://daa.uit.edu.vn/34-quy-trinh-chuyen-sinh-vien-tu-hinh-thuc-dao-tao-chinh-quy-sang-hinh-thuc-dao-tao-tu-xa",
]

# Wait this long (ms) after page load for JS to render content
JS_RENDER_WAIT_MS = 2000


class UitDaaScraper:
    """Crawls daa.uit.edu.vn with Playwright, collects page text and PDF links."""

    def __init__(
        self,
        base_url: str = "https://daa.uit.edu.vn",
        seed_urls: list[str] | None = None,
    ):
        self.base_url = base_url
        self.seed_urls = seed_urls or SEED_URLS
        self.visited: set[str] = set()
        self.delay = 1  # seconds between page navigations
        # requests.Session used only for PDF binary downloads
        self._dl_session = requests.Session()
        self._dl_session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/pdf,*/*",
            "Accept-Language": "vi,en;q=0.9",
            "Referer": self.base_url,
        })

    # ------------------------------------------------------------------
    # Page scraping (Playwright)
    # ------------------------------------------------------------------

    def _extract_page_data(self, pw_page: Page, url: str) -> dict:
        """Navigate to url, wait for JS render, extract text + links."""
        try:
            pw_page.goto(url, wait_until="domcontentloaded", timeout=20000)
            # Wait for JS to finish rendering content
            pw_page.wait_for_timeout(JS_RENDER_WAIT_MS)
        except Exception as exc:
            print(f"[scraper] SKIP {url} — {exc}")
            return None

        content = pw_page.content()
        soup = BeautifulSoup(content, "html.parser")

        # Prefer <main> or common content wrappers; fall back to <body>
        main = (
            soup.find("main")
            or soup.find("div", class_="content")
            or soup.find("div", id="content")
            or soup.body
        )

        page_data: dict = {
            "url": url,
            "title": (soup.title.string.strip() if soup.title else ""),
            "text": (main.get_text(separator="\n", strip=True) if main else ""),
            "pdf_links": [],
            "internal_links": [],
        }

        base_netloc = urlparse(self.base_url).netloc

        for anchor in soup.find_all("a", href=True):
            href: str = urljoin(url, anchor["href"])
            href = href.split("#")[0]  # strip fragment
            if not href.startswith("http"):
                continue

            if href.lower().endswith(".pdf"):
                page_data["pdf_links"].append({
                    "url": href,
                    "title": anchor.get_text(strip=True),
                })
            elif urlparse(href).netloc == base_netloc:
                page_data["internal_links"].append(href)

        return page_data

    def scrape_page(self, pw_page: Page, url: str) -> dict | None:
        """Fetch a single page, extract text, PDF links, and internal links."""
        if url in self.visited:
            return None
        self.visited.add(url)
        result = self._extract_page_data(pw_page, url)
        time.sleep(self.delay)
        return result

    def crawl(self, max_pages: int = 500) -> list[dict]:
        """BFS crawl starting from all seed URLs up to max_pages pages."""
        queue: list[str] = list(self.seed_urls)
        all_pages: list[dict] = []

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (compatible; UIT-DoAn-Bot/1.0)"
            )
            pw_page = context.new_page()

            while queue and len(all_pages) < max_pages:
                url = queue.pop(0)
                page = self.scrape_page(pw_page, url)
                if page is None:
                    continue
                all_pages.append(page)
                for link in page["internal_links"]:
                    if link not in self.visited:
                        queue.append(link)

            browser.close()

        print(f"[scraper] Crawled {len(all_pages)} pages from daa.uit.edu.vn")
        return all_pages

    # ------------------------------------------------------------------
    # PDF download (requests — binary, no JS needed)
    # ------------------------------------------------------------------

    def download_pdfs(
        self,
        pdf_links: list[dict],
        save_dir: str = "data/raw/pdfs",
    ) -> None:
        """Download all PDFs, skip already-downloaded ones (by URL and by file)."""
        os.makedirs(save_dir, exist_ok=True)
        seen_urls: set[str] = set()
        for pdf in pdf_links:
            url = pdf["url"]
            # Skip duplicate URLs within the same batch
            if url in seen_urls:
                continue
            seen_urls.add(url)

            # URL-encode non-ASCII characters in the path so requests can handle them
            from urllib.parse import quote, urlsplit, urlunsplit
            parts = urlsplit(url)
            safe_url = urlunsplit(parts._replace(path=quote(parts.path, safe="/%")))

            # Use the last path segment (decoded) as the local filename
            from urllib.parse import unquote
            filename = unquote(safe_url.split("/")[-1]) or "document.pdf"
            filepath = os.path.join(save_dir, filename)
            # Skip if already downloaded
            if os.path.exists(filepath):
                safe_name = filename.encode("ascii", errors="replace").decode()
                print(f"[scraper] Already exists, skip {safe_name}")
                continue
            try:
                resp = self._dl_session.get(safe_url, timeout=60)
                resp.raise_for_status()
                with open(filepath, "wb") as fh:
                    fh.write(resp.content)
                safe_name = filename.encode("ascii", errors="replace").decode()
                print(f"[scraper] Downloaded {safe_name}")
            except requests.RequestException as exc:
                # Encode error message safely for Windows terminals
                print(f"[scraper] PDF error {safe_url} - {str(exc).encode('ascii', errors='replace').decode()}")
            time.sleep(self.delay)


# ------------------------------------------------------------------
# Standalone run
# ------------------------------------------------------------------

if __name__ == "__main__":
    scraper = UitDaaScraper()
    pages = scraper.crawl(max_pages=500)

    os.makedirs("data/raw/html", exist_ok=True)
    with open("data/raw/html/pages.json", "w", encoding="utf-8") as fh:
        json.dump(pages, fh, ensure_ascii=False, indent=2)
    print("[scraper] Saved data/raw/html/pages.json")

    all_pdfs = [p for page in pages for p in page.get("pdf_links", [])]
    scraper.download_pdfs(all_pdfs)
    print(f"[scraper] Done — {len(all_pdfs)} PDF links found")
