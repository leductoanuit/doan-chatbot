# Phase 4: Update Text Cleaner

## Priority: Medium | Status: ✅ | Effort: Low

## Overview
Update boilerplate patterns in `text-cleaner.py` from `ctsv.uit.edu.vn` patterns to `daa.uit.edu.vn` patterns. Need to inspect actual page content first.

## Files to Modify
- `src/scraper/text-cleaner.py` — update `_BOILERPLATE_PATTERNS`

## Implementation Steps

### 1. Update Boilerplate Patterns
Replace ctsv-specific patterns with daa.uit.edu.vn patterns:

```python
_BOILERPLATE_PATTERNS = [
    r"Trang chủ\s*[»›|/].*",
    r"Copyright\s*©.*",
    r"Bản quyền.*",
    r"Phòng Đào tạo.*Đại học.*",
    r"Số lượt truy cập.*",
    r"Đăng nhập.*",
    r"Tìm kiếm.*",
    r"Menu\s*chính.*",
    r"Tin tức\s*mới.*sidebar.*",
    r"Email:.*daa.*",
    r"Email:.*uit\.edu\.vn.*",
    r"Điện thoại:.*",
]
```

### 2. Keep Core Logic
- Unicode NFC normalization — keep
- URL removal — keep
- Whitespace compression — keep
- Jaccard deduplication — keep

## Todo
- [x] Crawl a sample page from daa.uit.edu.vn to identify boilerplate
- [x] Update _BOILERPLATE_PATTERNS list
- [x] Update email pattern from ctsv to daa
- [x] Test with sample crawled content

## Success Criteria
- Boilerplate text (headers, footers, menus) removed from daa.uit.edu.vn pages
- Vietnamese content preserved with correct diacritics
- Deduplication still works
