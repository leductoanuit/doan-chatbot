# Phase 01 — Tạo Evaluation Dataset

## Overview
- **Priority**: P1
- **Status**: Completed
- **Effort**: 2h

Tạo bộ 30 câu hỏi ground truth bao phủ toàn bộ domain UIT đào tạo từ xa. Dataset dùng format RAGAS: `question`, `ground_truth`, `reference_contexts`.

## Key Insights

- RAGAS cần 3 trường: `question`, `ground_truth` (answer chuẩn), `reference_contexts` (chunks đúng)
- Câu hỏi phải lấy từ nội dung tài liệu thực tế trong `data/raw/` — không hallucinate
- Cần bao phủ cả câu hỏi dễ (FAQ) lẫn câu hỏi khó (cross-document, multi-hop)
- Phân bổ câu hỏi theo document type để đánh giá coverage

## Requirements

- 30 câu hỏi tối thiểu
- Ít nhất 5 category: tuyển sinh, học vụ, học phí, CTĐT, quy chế
- Mỗi câu có `ground_truth` answer + `reference_contexts` (chunk text gốc)
- Format JSON tương thích RAGAS Dataset

## Dataset Structure

```json
[
  {
    "question": "Điều kiện tuyển sinh hệ đào tạo từ xa UIT là gì?",
    "ground_truth": "...",
    "reference_contexts": ["chunk text 1", "chunk text 2"],
    "category": "tuyen_sinh",
    "difficulty": "easy"
  }
]
```

## Question Categories

| Category | Số câu | Nguồn tài liệu |
|----------|--------|----------------|
| Tuyển sinh | 6 | FAQ tuyển sinh, 507-QĐ |
| Học vụ | 8 | 507-QĐ, 1499-QĐ |
| Học phí | 5 | Thông báo học phí |
| CTĐT (ngành TTNT, CNTT) | 6 | Chương trình đào tạo |
| Quy chế chung | 5 | TT 28/2023, TT 21/2019 |

## Related Code Files

- **Create**: `tests/evaluation/eval-dataset.json`
- **Create**: `tests/evaluation/` directory
- **Read**: `data/raw/*.pdf`, `data/raw/*.json`, `data/raw/*.docx`
- **Read**: `data/processed/all_documents_ocr_cleaned.json` — lấy reference_contexts

## Implementation Steps

1. Tạo thư mục `tests/evaluation/` và `tests/evaluation/results/`
2. Đọc các tài liệu trong `data/raw/` để lấy nội dung thực tế
3. Viết 30 câu hỏi thủ công, mỗi câu có:
   - `question`: câu hỏi tự nhiên như user thật
   - `ground_truth`: trả lời dựa trên tài liệu gốc (không paraphrase quá nhiều)
   - `reference_contexts`: copy paste đoạn chunk liên quan từ `all_documents_ocr_cleaned.json`
   - `category`: một trong 5 loại trên
   - `difficulty`: `easy` / `medium` / `hard`
4. Validate JSON format hợp lệ
5. Kiểm tra: mỗi `reference_contexts` phải xuất hiện trong Qdrant (dùng keyword search verify)

## Todo List

- [x] Tạo `tests/evaluation/` directory
- [x] Đọc tài liệu gốc để xác định ground truth
- [x] Viết 30 câu hỏi với đầy đủ trường
- [x] Validate JSON format
- [x] Cross-check reference_contexts với Qdrant data

## Success Criteria

- File `tests/evaluation/eval-dataset.json` tồn tại với >= 30 entries
- JSON valid, đầy đủ 5 trường mỗi entry
- Coverage: đủ 5 category, có cả easy/medium/hard

## Risk Assessment

- **OCR noise**: một số chunks từ PDF scan có thể sai chữ → dùng chunks từ DOCX/JSON trước, PDF chỉ khi cần
- **Ground truth chủ quan**: 2 người có thể viết answer khác nhau → ưu tiên copy trực tiếp từ tài liệu

## Next Steps

→ Phase 02: dùng dataset này làm input cho RAGAS pipeline
