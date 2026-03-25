# Phase 1: Update Web Scraper

## Priority: High | Status: ✅ | Effort: Medium

## Overview
Retarget `UitCtsvScraper` from `ctsv.uit.edu.vn` → `daa.uit.edu.vn`. Use seed URLs as starting points instead of just base URL. Follow all internal links found on pages recursively.

## Files to Modify
- `src/scraper/web-scraper.py` — retarget domain, add seed URLs, rename class
- `src/scraper/run-pipeline.py` — update imports and class references

## Implementation Steps

### 1. Update `web-scraper.py`
- Rename class `UitCtsvScraper` → `UitDaaScraper`
- Change `base_url` default to `https://daa.uit.edu.vn`
- Add `seed_urls` parameter — list of specific pages to start crawling from
- Update `crawl()` to initialize queue with all seed URLs instead of just base_url
- Keep BFS logic, delay, robots.txt checking as-is
- Internal links: any URL with netloc `daa.uit.edu.vn` should be followed

### 2. Seed URLs
```python
SEED_URLS = [
    "https://daa.uit.edu.vn",
    "https://daa.uit.edu.vn/09-quyet-dinh-ve-viec-ban-hanh-quy-che-dao-tao-cho-sinh-vien-he-dao-tao-tu-xa-trinh-do-dai-hoc",
    "https://daa.uit.edu.vn/tu-xa/ctdt-khoa-2024",
    "https://daa.uit.edu.vn/33-quy-che-tuyen-sinh-hinh-thuc-dao-tao-tu-xa-trinh-do-dai-hoc",
    "https://daa.uit.edu.vn/34-quy-trinh-chuyen-sinh-vien-tu-hinh-thuc-dao-tao-chinh-quy-sang-hinh-thuc-dao-tao-tu-xa",
]
```

### 3. Update `run-pipeline.py`
- Change import from `UitCtsvScraper` → `UitDaaScraper`
- Update print messages from `ctsv.uit.edu.vn` → `daa.uit.edu.vn`

## Todo
- [x] Rename class and update default base_url
- [x] Add seed_urls parameter to constructor
- [x] Update crawl() to use seed URLs as initial queue
- [x] Update run-pipeline.py imports
- [x] Test crawl on daa.uit.edu.vn (verify pages are fetched)

## Success Criteria
- Scraper fetches pages from all 5 seed URLs
- Internal links on those pages are followed recursively
- PDF links on daa.uit.edu.vn are detected and downloaded
- Output saved to `data/raw/html/pages.json`

## Risk
- `daa.uit.edu.vn` may block bots — increase delay or add retry logic if needed
- Some pages may use JavaScript rendering — current approach (requests + BS4) won't work for JS-rendered content. Check if target pages serve static HTML.
