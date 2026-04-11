# RAG System Parameters

## Embedding

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| Model | `BAAI/bge-m3` | Đọc từ `.env` → `EMBEDDING_MODEL` |
| Dimension | 1024 | `src/storage/qdrant_vector_store.py` |
| Max seq length | 512 tokens | `src/embedding/embedder.py` |

---

## Chunking (`src/embedding/chunker.py`)

| Tham số | Giá trị | Đơn vị | Ghi chú |
|---|---|---|---|
| `chunk_size` | 512 | **words** | ~700–900 tokens thực tế |
| `overlap` | 100 | **words** | ~19.5% overlap |

> ⚠️ **Vấn đề:** Đơn vị là `words` (dùng `text.split()`), không phải tokens. Tiếng Việt 1 từ ≈ 1.5–2 subword tokens → chunk thực tế **vượt giới hạn 512 tokens** của BGE-M3, phần đuôi bị cắt thầm lặng.

---

## Retriever (`src/rag/retriever.py`)

### `vector_search()`
| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `k` | 10 | Số chunk lấy từ Qdrant |

### `keyword_search()`
| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `k` | 10 | Số chunk per keyword |
| keywords cap | 4 | Tối đa 4 keywords (bigrams ưu tiên) |
| keyword score | 0.5 | Score cố định cho mọi keyword hit |
| min word len | > 2 ký tự | Bỏ stop words ngắn |

### `hybrid_search()` — hàm chính
| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `k` | 5 | Số kết quả trả về cuối |
| `vector_weight` | 0.7 | Trọng số vector search |
| `keyword_weight` | 0.3 | Trọng số keyword search |
| vector pull | `k × 4 = 20` | Lấy 20 vector results trước khi merge |
| `MIN_SCORE` | 0.25 | Ngưỡng lọc: `final_score ≥ 0.25 × 0.7 = 0.175` |

### `build_context()`
| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `max_tokens` | 1500 | Giới hạn context gửi LLM (đo bằng **words**) |

---

## OCR Quality

### Công cụ hiện tại
- **Text-native PDF**: PyMuPDF (`src/scraper/pdf_extractor.py`)
- **Scanned PDF**: Tesseract (`scripts/ocr-local.py`)

### Tài liệu bị OCR lỗi

| File | Avg Noise | Max Noise | Trang lỗi nặng |
|---|---|---|---|
| `28_2023_tt_bgddt._quy_che_dttx_nam_2023.pdf` | 0.41 | **1.45** | p14, p16, p17 |
| `21_2019_TT_BGDDT.signed.pdf` | 0.26 | **1.42** | p21, p22 |
| `1499-qd-dhcntt-10-12-2024.pdf` | 0.33 | 0.74 | p4, p5 (sơ đồ/flowchart) |

### Phân loại lỗi
- **Trang scan mờ/nghiêng** → ký tự rác nặng (`score > 1.0`)
- **Sơ đồ/flowchart** → Tesseract đọc layout thành text vô nghĩa (`score 0.5–0.8`)
