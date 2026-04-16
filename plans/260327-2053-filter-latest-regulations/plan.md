---
title: "Filter Latest Regulation PDFs"
description: "Python script to deduplicate 188 regulation PDFs, keeping only latest version per topic"
status: pending
priority: P1
effort: 2h
branch: main
tags: [data-processing, pdf, regulations]
created: 2026-03-27
---

# Filter Latest Regulation PDFs

## Goal
From 188 Vietnamese university regulation PDFs, identify groups by topic, keep only the latest version of each, copy to `data/raw/pdfs/latest/`.

## Phases

| # | Phase | Status | File |
|---|-------|--------|------|
| 1 | Implement filter script | pending | [phase-01](phase-01-implement-script.md) |

## Key Decisions
- Single script at `src/data_processing/filter-latest-regulations.py`
- ~150 lines, modularized into functions
- Two-tier grouping: topic keywords + issuer
- CTDT files grouped by nganh (program), not by topic keyword
- `--dry-run` flag for safe preview
- Files without parseable dates → use document number as proxy for recency
- Uncategorized files → copied as-is (no filtering)
