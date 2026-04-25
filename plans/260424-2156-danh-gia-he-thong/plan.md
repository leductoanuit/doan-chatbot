---
title: "Đánh giá hệ thống RAG Chatbot UIT"
description: "Xây dựng pipeline đánh giá định lượng hệ thống RAG dùng RAGAS metrics + tạo bộ test dataset"
status: pending
priority: P1
effort: 6h
issue:
branch: main
tags: [evaluation, rag, ragas, testing]
created: 2026-04-25
---

# Đánh giá hệ thống RAG Chatbot UIT

## Overview

Hệ thống hiện chỉ có đánh giá định tính (manual). Plan này xây dựng đánh giá định lượng dùng **RAGAS framework** với bộ test dataset 30 câu hỏi thực tế từ domain UIT.

## Phases

| # | Phase | Status | Effort | Link |
|---|-------|--------|--------|------|
| 1 | Tạo evaluation dataset | Pending | 2h | [phase-01](./phase-01-create-eval-dataset.md) |
| 2 | Setup RAGAS pipeline | Pending | 2h | [phase-02](./phase-02-setup-ragas-pipeline.md) |
| 3 | Chạy evaluation & báo cáo | Pending | 2h | [phase-03](./phase-03-run-evaluation-report.md) |

## Dependencies

- Python + venv đã setup tại `requirements.txt`
- Hệ thống RAG đang chạy (`src/rag/pipeline.py`)
- Gemini API key (dùng làm LLM judge cho RAGAS)
- Qdrant + PostgreSQL running (Docker)

## Key Files

- `tests/evaluation/` — thư mục mới cho eval scripts
- `tests/evaluation/eval-dataset.json` — bộ câu hỏi ground truth
- `tests/evaluation/ragas-evaluator.py` — RAGAS pipeline
- `tests/evaluation/run-evaluation.py` — entry point
- `tests/evaluation/results/` — kết quả đánh giá
