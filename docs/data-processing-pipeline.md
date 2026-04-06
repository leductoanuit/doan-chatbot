# Phương pháp xử lý dữ liệu

## Tổng quan luồng

```
data/raw/                            data/processed/
├── *.pdf (+ thư mục PNG/)  ──→     all_documents_ocr.json
├── *.docx              ──→     (append)
├── *.json              ──→     (append)
└── *.pdf, *.docx, *.json ──→     document-metadata.json
                                          │
                                          ▼
                                ingest_pipeline.py
                                ├── Qdrant (vectors)
                                └── PostgreSQL (metadata)
```

---

## Bước 1 — OCR PDF (`data/raw/ocr-local.py`)

**Input:** Thư mục ảnh PNG trong `data/raw/` (mỗi thư mục = 1 PDF đã render sẵn)

**Xử lý mỗi trang:**
1. Đọc ảnh PNG bằng OpenCV
2. **Deskew** — phát hiện góc nghiêng qua `minAreaRect`, xoay thẳng bằng Warp Affine
3. **Binarize** — `adaptiveThreshold` để tách chữ khỏi nền
4. **Tesseract OCR** — nhận dạng text tiếng Việt + Anh (`vie+eng`, psm=6)

**Output chunk:**
```json
{ "content": "...", "page": 1, "source": "ten-file.pdf", "method": "tesseract_warp_affine" }
```

**Phụ thuộc:** `opencv-python`, `pytesseract`, Tesseract binary + gói ngôn ngữ `vie`

---

## Bước 2 — Trích xuất DOCX + JSON (`data/raw/extract-docx-json.py`)

**DOCX:** `python-docx` đọc paragraph + bảng → chunk ~2000 ký tự

**JSON:** Đọc mảng `[{title, text}]` → ghép title + text thành chunk

Cả 2 **append** vào `all_documents_ocr.json` đã có từ bước 1.

**Output chunk:**
```json
{ "content": "...", "page": 2, "source": "ten-file.docx", "method": "python_docx" }
{ "content": "...", "page": 1, "source": "ten-file.json", "method": "json_extract" }
```

---

## Bước 3 — Sinh metadata (`data/raw/generate-metadata.py`)

Hoàn toàn **offline**, không cần API — dùng regex + pattern matching trên tên file và nội dung trang đầu.

| Trường | Cách xác định |
|--------|--------------|
| `document_number` | Regex trên tên file + nội dung trang 1 PDF |
| `issue_date` | Regex ngày `DD/MM/YYYY`, `ngày X tháng Y năm Z` |
| `issuing_body` | Keyword: `BGDĐT`, `BTTTT`, `ĐHCNTT`... |
| `document_type` | Keyword: `quyết định`, `thông tư`, `TTLT`... |
| `system_type` | Keyword: `đào tạo`, `chứng chỉ`, `tuyển sinh`... |

**Map key:** `source_file` (metadata) = `source` (chunk) = tên file gốc

**Output:** `data/processed/document-metadata.json`

---

## Bước 4 — Ingest (`src/embedding/ingest_pipeline.py`)

1. Load `all_documents_ocr.json` → chunk nhỏ hơn qua `chunker`
2. Lookup `document-metadata.json` theo `source_file` → gắn metadata vào chunk
3. **Embed** bằng `VietnameseEmbedder` (sentence-transformers)
4. **Upsert** vector vào Qdrant
5. **Insert** metadata vào PostgreSQL

---

## Thứ tự chạy

```bash
python3 data/raw/ocr-local.py              # Bước 1: OCR PDF
python3 data/raw/extract-docx-json.py      # Bước 2: DOCX + JSON
python3 data/raw/generate-metadata.py      # Bước 3: Metadata
python3 src/embedding/ingest_pipeline.py   # Bước 4: Embed + Store
```

---

## Kết quả hiện tại

| Nguồn | Số chunks |
|-------|-----------|
| PDF (8 file, Tesseract OCR) | 163 |
| DOCX (5 file) | 58 |
| JSON (2 file) | 24 |
| **Tổng** | **245** |
| Metadata entries | 15 (map 100%) |
