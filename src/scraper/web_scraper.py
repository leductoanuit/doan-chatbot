"""Web scraper for daa.uit.edu.vn — BFS crawl with respectful rate limiting."""

import json
import os
import time
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup


# Seed URLs for distance learning (dao tao tu xa) pages
SEED_URLS = [
    "https://daa.uit.edu.vn",
    "https://daa.uit.edu.vn/09-quyet-dinh-ve-viec-ban-hanh-quy-che-dao-tao-cho-sinh-vien-he-dao-tao-tu-xa-trinh-do-dai-hoc",
    "https://daa.uit.edu.vn/tu-xa/ctdt-khoa-2024",
    "https://daa.uit.edu.vn/33-quy-che-tuyen-sinh-hinh-thuc-dao-tao-tu-xa-trinh-do-dai-hoc",
    "https://daa.uit.edu.vn/34-quy-trinh-chuyen-sinh-vien-tu-hinh-thuc-dao-tao-chinh-quy-sang-hinh-thuc-dao-tao-tu-xa",
]


class UitDaaScraper:
    """Crawls daa.uit.edu.vn, collects page text and PDF links."""

    def __init__(
        self,
        base_url: str = "https://daa.uit.edu.vn",
        seed_urls: list[str] | None = None,
    ):
        self.base_url = base_url
        self.seed_urls = seed_urls or SEED_URLS
        self.visited: set[str] = set()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; UIT-DoAn-Bot/1.0)"
        })
        self.delay = 2  # seconds between requests
        self._robots: RobotFileParser | None = None

    # ------------------------------------------------------------------
    # robots.txt
    # ------------------------------------------------------------------

    def _load_robots(self) -> RobotFileParser:
        if self._robots is None:
            rp = RobotFileParser()
            rp.set_url(f"{self.base_url}/robots.txt")
            try:
                rp.read()
            except Exception:
                pass  # If robots.txt unreachable, allow all
            self._robots = rp
        return self._robots

    def _can_fetch(self, url: str) -> bool:
        return self._load_robots().can_fetch("*", url)

    # ------------------------------------------------------------------
    # Scraping
    # ------------------------------------------------------------------

    def scrape_page(self, url: str) -> dict | None:
        """Fetch a single page, extract text, PDF links, and internal links."""
        if url in self.visited or not self._can_fetch(url):
            return None
        self.visited.add(url)

        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            resp.encoding = "utf-8"
        except requests.RequestException as exc:
            print(f"[scraper] SKIP {url} — {exc}")
            return None

        soup = BeautifulSoup(resp.content, "html.parser")

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
            # Normalise: strip fragment
            href = href.split("#")[0]
            if not href.startswith("http"):
                continue

            if href.lower().endswith(".pdf"):
                page_data["pdf_links"].append({
                    "url": href,
                    "title": anchor.get_text(strip=True),
                })
            elif urlparse(href).netloc == base_netloc:
                page_data["internal_links"].append(href)

        time.sleep(self.delay)
        return page_data

    def crawl(self, max_pages: int = 500) -> list[dict]:
        """BFS crawl starting from all seed URLs up to max_pages pages."""
        # Initialize queue with all seed URLs
        queue: list[str] = list(self.seed_urls)
        all_pages: list[dict] = []

        while queue and len(all_pages) < max_pages:
            url = queue.pop(0)
            page = self.scrape_page(url)
            if page is None:
                continue
            all_pages.append(page)
            for link in page["internal_links"]:
                if link not in self.visited:
                    queue.append(link)

        print(f"[scraper] Crawled {len(all_pages)} pages from daa.uit.edu.vn")
        return all_pages

    # ------------------------------------------------------------------
    # PDF download
    # ------------------------------------------------------------------

    def download_pdfs(
        self,
        pdf_links: list[dict],
        save_dir: str = "data/raw/pdfs",
    ) -> None:
        """Download all PDFs, skip already-downloaded ones."""
        os.makedirs(save_dir, exist_ok=True)
        for pdf in pdf_links:
            filename = pdf["url"].split("/")[-1] or "document.pdf"
            filepath = os.path.join(save_dir, filename)
            # Avoid overwriting — append counter if file already exists
            if os.path.exists(filepath):
                name, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(filepath):
                    filepath = os.path.join(save_dir, f"{name}_{counter}{ext}")
                    counter += 1
            try:
                resp = self.session.get(pdf["url"], timeout=60)
                resp.raise_for_status()
                with open(filepath, "wb") as fh:
                    fh.write(resp.content)
                print(f"[scraper] Downloaded {filename}")
            except requests.RequestException as exc:
                print(f"[scraper] PDF error {pdf['url']} — {exc}")
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
