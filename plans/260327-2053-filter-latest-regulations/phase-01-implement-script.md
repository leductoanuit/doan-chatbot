# Phase 01: Implement Filter Script

## Context
- [Actual filenames analyzed from data/raw/pdfs/](#analysis-below)
- 188 PDFs, Vietnamese university regulations
- Script: `src/data_processing/filter-latest-regulations.py`

## Overview
- **Priority**: P1
- **Status**: pending
- **Effort**: ~2h

## Key Insights from Filename Analysis

### Date Formats Found
1. `_DD-MM-YYYY_` — most common: `28-9-22`, `29-12-2023`, `10-02-2026`
2. `_DD_MM_YYYY_` — underscores: `04_10_2022`, `07_01_2025`
3. `_YYMMDD_` — compact: `240307`, `241029`, `251025`
4. `_DD-M-YY_` — short year: `28-9-22`
5. No date — some files have no embedded date (e.g., `CTDT_CNPM_khoa1.pdf`, `de_an.pdf`)

### Topic Groups Identified (from actual files)

| Group Key | Keyword Pattern | Example Files |
|-----------|----------------|---------------|
| `quy_che_dao_tao` | `quy_che_dao_tao` | `08_2021...quy_che_dao_tao`, `790-qd...quy_che_dao_tao`, `1393-qd...cap_nhat_quy_che_dao_tao`, `507-qd...quy_che_dao_tao` |
| `ngoai_ngu` | `ngoai_ngu` OR `tieng_anh` OR `day_va_hoc_nn` | `125-qd...tieng_anh_khoa_2015`, `828_qd...ngoai_ngu...khoa_2022`, `560-qd...ngoai_ngu` |
| `van_bang_chung_chi` | `van_bang_chung_chi` OR `vbcc` | `1372-qd-dhqg...van_bang_chung_chi`, `172-qd-dhcntt...van_bang_chung_chi` |
| `mo_nganh` | `mo_nganh` | `02_2022...mo_nganh`, `540_qd_dhqg...mo_nganh`, `15_qd_dhqg...mo_nganh` |
| `to_chuc_thi` | `to_chuc_thi` | `1139_qd...to_chuc_thi`, `1376_qd...to_chuc_thi`, `135-qd...to_chuc_thi` |
| `tai_nang` | `tai_nang` | `131_qd...tai_nang`, `1032-qd...tai_nang` |
| `tien_tien` | `tien_tien` | `434-qd...tien_tien`, `1451-qd...tien_tien` |
| `chat_luong_cao` | `chat_luong_cao` | `435-qd...chat_luong_cao` |
| `song_nganh` | `song_nganh` | `1195-qd-dhqg...song_nganh`, `112-qd-dhcntt...song_nganh` |
| `lien_thong` | `lien_thong` | `335-qd...lien_thong`, `650-qd-dhqg...lien_thong` |
| `kltn` | `kltn` | `159-qd...kltn` |
| `do_an` | `do_an_tot_nghiep` | `697-qd...do_an_tot_nghiep` |
| `tuyen_sinh` | `tuyen_sinh` | `707_qd_dhqg...tuyen_sinh`, `836_qd_dhqg...tuyen_sinh` |
| `giao_duc_the_chat` | `giao_duc_the_chat` | `1476_qd_dhqg...giao_duc_the_chat` |
| `truc_tuyen` | `truc_tuyen` OR `dao_tao_truc_tuyen` | `196-qd...truc_tuyen`, `1141-qd...truc_tuyen` |
| `phan_cong_cbct` | `phan_cong` OR `coi_thi` | `672-qd...phan_cong_cbct`, `936_qd...phan_cong` |
| `cong_nhan_tin_chi` | `cong_nhan_tin_chi` | `qd2062...cong_nhan_tin_chi`, `412-qd...cong_nhan_tin_chi` |
| `mooc` | `mooc` | `qd1537...mooc`, `2217...mooc` |
| `xlhv` | `xlhv` (xu ly hoc vu) | `803-qd...xlhv` |
| `CTDT_{nganh}` | `CTDT_{CNPM,HTTT,KHMT,KTMT,MMT,...}` | `CTDT_CNPM_khoa1.pdf` ... `CTDT_MMT_khoa%202.pdf` |
| `lien_ket_dao_tao` | `lien_ket_dao_tao` | `1758-qd-dhqg...lien_ket_dao_tao`, `07-bgd...lien_ket_dao_tao` |
| `dttx` | `dttx` (dao tao tu xa) | `tt10...dttx`, `28_2023...dttx` |

### Issuer Hierarchy (for same-topic conflicts)
- `dhcntt` — UIT-specific (school-level, overrides DHQG for UIT context)
- `dhqg` — VNU-HCM level
- `bgddt` / `bgd` — Ministry level
- `ttg` — Prime Minister
- `cp` — Government
- `qh` — National Assembly

**Decision**: Group by (topic) only, NOT by (issuer, topic). Reason: later UIT decisions (`dhcntt`) often supersede/implement earlier `dhqg` or `bgddt` rules. Keep the latest regardless of issuer.

**Exception**: If topic is a national-level regulation (e.g., `luat_giao_duc`) vs school-level implementation, they are different documents — keep both. Handle via keyword: `luat_` prefix stays separate.

## Requirements

### Functional
- Parse date from filename using regex (multiple formats)
- Group files by topic keyword matching
- CTDT files: group by `CTDT_{nganh}`, keep all khoa versions (they're different programs, not versions)
- For each topic group: select file with latest date
- Copy selected files to `data/raw/pdfs/latest/`
- Files matching no topic group → copy as-is (ungrouped = unique doc)
- `--dry-run` flag: print what would be copied without copying
- Print summary report

### Non-Functional
- Single file, <200 lines
- No external dependencies (stdlib only: `re`, `os`, `shutil`, `argparse`, `pathlib`, `datetime`)
- Idempotent: safe to re-run (clears `latest/` dir first or overwrites)

## Architecture

```
filter-latest-regulations.py
├── extract_date(filename) → datetime.date | None
├── extract_topic(filename) → str | None
├── group_files(filenames) → dict[str, list[tuple[filename, date]]]
├── select_latest(groups) → list[filename]
└── main() → parse args, orchestrate, copy/report
```

## Related Code Files

### Files to Create
- `src/data_processing/filter-latest-regulations.py`

### Directories to Create
- `src/data_processing/` (with `__init__.py`)
- `data/raw/pdfs/latest/` (at runtime)

## Implementation Steps

### 1. Create `src/data_processing/__init__.py`
Empty init file for package.

### 2. Implement `extract_date(filename: str) -> date | None`

Date regex patterns to try **in order** (first match wins):

```python
patterns = [
    # DD-MM-YYYY or DD-M-YYYY (separator: - or _)
    r'[\-_](\d{1,2})[\-_](\d{1,2})[\-_](\d{4})[\-_]',
    # DD-MM-YY (2-digit year)
    r'[\-_](\d{1,2})[\-_](\d{1,2})[\-_](\d{2})[\-_]',
    # YYMMDD compact (6 digits after underscore, e.g., qd1537_241029_)
    r'_(\d{6})_',
    # Standalone YYYY between separators (fallback, use Jan 1)
    r'[\-_](20\d{2})[\-_]',
]
```

For 2-digit years: `<= 50` → `20XX`, `> 50` → `19XX`.

For compact `YYMMDD`: first 2 = year, next 2 = month, last 2 = day.

### 3. Implement `extract_topic(filename: str) -> str | None`

Keyword list ordered **longest match first** to avoid partial matches:

```python
TOPIC_KEYWORDS = [
    'quy_che_dao_tao',
    'van_bang_chung_chi',
    'cong_nhan_tin_chi',
    'giao_duc_the_chat',
    'do_an_tot_nghiep',
    'chat_luong_cao',
    'lien_ket_dao_tao',
    'to_chuc_thi',
    'phan_cong',
    'song_nganh',
    'lien_thong',
    'truc_tuyen',
    'tuyen_sinh',
    'ngoai_ngu',
    'tien_tien',
    'tai_nang',
    'mo_nganh',
    'kltn',
    'xlhv',
    'mooc',
    'dttx',
]
```

Special cases:
- `tieng_anh` or `day_va_hoc_nn` → map to `ngoai_ngu`
- `CTDT_{X}_khoa` → return `CTDT_{X}` (keep all khoa versions, see below)
- `luat_` prefix → return `luat_{rest}` (national law, separate from implementing regulations)
- `de_an_mo_nganh_{X}` → return `de_an_mo_nganh_{X}` (each is unique per nganh)
- `dcct_` or `pl_bang_mo_ta_dcct_` → return full match (each subject's DCCT is unique)

### 4. Implement `group_files()`

```python
def group_files(pdf_dir: Path) -> dict[str, list[tuple[str, date | None]]]:
    groups = defaultdict(list)
    for f in pdf_dir.glob('*.pdf'):
        if f.parent.name == 'latest':
            continue  # skip output dir
        topic = extract_topic(f.name)
        dt = extract_date(f.name)
        if topic:
            groups[topic].append((f.name, dt))
        else:
            groups[f'__unique__{f.name}'].append((f.name, dt))
    return groups
```

### 5. Implement `select_latest()`

For each group:
- If only 1 file → select it
- If multiple files with dates → select max date
- If multiple files, some without dates → prefer those with dates, pick latest
- If multiple files, none with dates → select all (can't determine which is latest)

**CTDT exception**: Groups starting with `CTDT_` → select ALL (different khoa = different curriculum versions, all needed).

### 6. Implement `main()`

```python
def main():
    parser = argparse.ArgumentParser(description='Filter latest regulation PDFs')
    parser.add_argument('--dry-run', action='store_true', help='Preview without copying')
    parser.add_argument('--src', default='data/raw/pdfs', help='Source PDF directory')
    args = parser.parse_args()

    src = Path(args.src)
    dst = src / 'latest'

    groups = group_files(src)
    selected = select_latest(groups)

    if not args.dry_run:
        dst.mkdir(exist_ok=True)

    for filename in sorted(selected):
        print(f"  COPY: {filename}")
        if not args.dry_run:
            shutil.copy2(src / filename, dst / filename)

    print(f"\nTotal: {len(selected)} files selected from {len(groups)} groups")
```

### 7. Add verbose summary output

Print per-group report:
```
[quy_che_dao_tao] 4 versions found
  - 08_2021...pdf (2021-03-18)
  - 790-qd...pdf (2022-09-28)
  - 1393-qd...pdf (2023-12-29)
  - 507-qd...pdf (2024-05-27) ← LATEST
```

## Todo List
- [ ] Create `src/data_processing/__init__.py`
- [ ] Implement `filter-latest-regulations.py` with all functions
- [ ] Test with `--dry-run` on actual data
- [ ] Verify grouping output looks correct
- [ ] Run actual copy

## Success Criteria
- Script runs without errors on the 188 PDFs
- Each topic group correctly identifies the latest document
- `--dry-run` output shows sensible grouping decisions
- `latest/` folder contains deduplicated set
- No regulation topic is missing its latest version
- CTDT files all preserved (different programs)
- Ungrouped unique documents preserved

## Risk Assessment
| Risk | Impact | Mitigation |
|------|--------|------------|
| Date parsing misses a format | Wrong "latest" selected | Print all dates in verbose mode for manual verification |
| Topic keyword too broad/narrow | Files mis-grouped | Review `--dry-run` output before actual copy |
| URL-encoded filenames (`%20`) | Matching fails | Decode with `urllib.parse.unquote` before matching |
| Same topic, different scope (DH vs ThS) | Wrong dedup | Add `trinh_do` qualifier to group key if `thac_si` or `tien_si` in name |

## Security Considerations
- Read-only on source files (copy, not move)
- No network access needed
- No credential handling

## Unresolved Questions
1. **Should `cap_nhat` / `sua_doi` (amendment) files supersede originals?** Current assumption: yes, they have later dates and represent the latest version. But some amendments only modify specific articles — the original base document might still be needed. Recommend: keep latest, note in output when amendment supersedes a base document.
2. **CTDT files with `khoa` — are ALL khoa versions needed or just latest?** Current assumption: all needed (different cohorts). Confirm with user.
3. **Files with no date AND no topic** (e.g., `de_an.pdf`, `mau_don.pdf`, `dkhp_doc_0.pdf`) — copy to latest or skip? Current assumption: copy as-is.
