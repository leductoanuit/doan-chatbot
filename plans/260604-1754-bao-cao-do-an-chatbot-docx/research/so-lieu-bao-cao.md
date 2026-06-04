# Số liệu chính xác cho báo cáo (verified 2026-06-04)

## Đề tài
Chatbot RAG tư vấn hệ đào tạo từ xa, Trường ĐH Công nghệ Thông tin (UIT) — ĐHQG-HCM.
Phạm vi tri thức: tuyển sinh, chương trình đào tạo, học vụ, học phí, quy chế ĐTTX.

## Dữ liệu
- Nguồn: 19 sources (8 PDF quyết định/thông tư + DOCX CTĐT/FAQ/hồ sơ + JSON quy trình biểu mẫu CITD scrape)
- Văn bản chính: 1499/QĐ-ĐHCNTT (2024), 507/QĐ-ĐHCNTT (2024), 213/QĐ-ĐHCNTT (2024), 790/QĐ-ĐHCNTT (2022), TT 28/2023/TT-BGDĐT, TT 21/2019, TTLT 17/2016; CTĐT CNTT VB1/LTĐH, CTĐT TTNT từ xa 2024; FAQ; hồ sơ tuyển sinh
- Pipeline: PyMuPDF (PDF native) + Tesseract OCR (scan, tiền xử lý deskew/binarize OpenCV) + python-docx → cleaning → `all_documents_final.json` = **448 records (theo trang)**
- Metadata mỗi record: source, page, doc_id, title, document_number, issue_date, issuing_body, document_type, system_type (sinh bằng Gemini 2.0 Flash, temp 0.1)
- Chunking: 512 từ/chunk, overlap 100 từ, recursive theo đoạn/câu → **513 chunks** (tự tính lại từ chunker thực tế)
- Embedding: **BAAI/bge-m3**, 1024-dim, normalized, batch 16
- Vector DB: **Qdrant**, collection `documents`, cosine, payload index: document_type/system_type/document_id; metadata + chunk content lưu PostgreSQL

## RAG Pipeline (online)
1. Chitchat detection (regex) → trả lời trực tiếp, bỏ qua RAG
2. Query rewriting (LLM, temp 0.1, 3 lượt hội thoại gần nhất) khi câu hỏi phụ thuộc ngữ cảnh
3. Intent classification (keyword) → filter dao_tao/tuyen_sinh/chung_chi; fallback bỏ filter nếu <3 kết quả
4. Multi-query expansion (RAG Fusion, temp 0.5) cho câu phức tạp/so sánh; câu so sánh: top_k×2
5. Hybrid retrieval: vector (Qdrant ANN, weight 0.7) + keyword (unigram+bigram, weight 0.3), k=10, MIN_SCORE=0.25, dedupe
6. Reranking: **BAAI/bge-reranker-v2-m3** (cross-encoder), top 10 → top 5
7. Context building: ~1500 tokens, citation [Nguồn: ...]; so sánh ×2 budget
8. Generation: **Gemini 2.5 Flash** (`src/rag/llm_client.py`), temp 0.3, max_tokens 1024, system prompt CoT tiếng Việt (anti-hallucination, ưu tiên quy định UIT > Bộ GD > FAQ, bảng cho so sánh, từ chối ngoài phạm vi)

LƯU Ý: LLM hiện tại = gemini-2.5-flash (không phải 2.0). Metadata generator dùng gemini-2.0-flash, QA generator gemini-1.5-flash.

## Prompt engineering (6 kỹ thuật — map sang mục 3.6 template)
1. Query rewriting & chuẩn hóa ngữ nghĩa (temp 0.1)
2. Multi-query / RAG Fusion
3. Intent classification + metadata filter
4. CoT generation prompt (47 dòng, tiếng Việt)
5. Anti-hallucination + citation
6. Chitchat/out-of-domain rejection

(Template có HyDE + nhận biết ảnh — hệ này KHÔNG có, không bịa.)

## Đánh giá
### Tập kiểm thử
- `eval-dataset.json`: **100 câu** (tuyen_sinh 20, hoc_vu 25, hoc_phi 15, ctdt 20, quy_che 20), fields: question/ground_truth/reference_contexts/category/difficulty (easy/medium/hard)
- `eval-dataset-robustness.json`: **11 câu** (tuyen_sinh 4, hoc_vu 1, out_of_domain 6) với noisy_contexts
- Sinh bán tự động (qa_generator Gemini) + kiểm tra xác thực (qa_validator)

### Phương pháp
- Generation (RAGAS-style, LLM-as-judge = gemini-2.5-flash): Faithfulness, Answer Relevancy, Context Precision, Context Recall, Correctness
- Retrieval rank: MAP@10, MRR, Hit@10 (recall-based relevance)
- Robustness (Auepora): Noise Robustness, Negative Rejection
- Latency: P50/P90/P99

### Kết quả chính — Run 50 câu (2026-06-02, lần 2, judge gemini-2.5-flash)
| Metric | Điểm |
|---|---|
| Faithfulness | 0.730 |
| Answer Relevancy | 0.760 |
| Context Precision | 0.752 |
| Context Recall | 0.870 |
| Correctness | 0.590 |
| Trung bình | 0.740 |

Theo danh mục: ctdt .786/.786/.757/.821; hoc_phi .600/.750/.900/1.000; hoc_vu .818/.818/.546/.909; quy_che .700/.700/.800/.800; tuyen_sinh N/A (lỗi).
Theo độ khó: easy .759/.796/.800/.796; medium .643/.750/.786/.929; hard .778/.667/.556/1.000.
Latency: P50 130.1s, P90 152.2s, P99 166.9s, avg 130.6s.

### Run 11 câu chuẩn (2026-06-04 17:03)
- Faithfulness .727, Relevancy .909, Ctx Precision .909, Ctx Recall .818, Correctness .682
- MAP@10 .840, MRR .864, Hit@10 .909 (hoc_vu: .94/1.0/1.0; tuyen_sinh: .757/.75/.833)
- Latency: P50 159.0s, avg 163.0s

### Run 11 câu robustness (2026-06-04 17:33)
- Faithfulness .80, Relevancy 1.0, Ctx Precision 1.0, Ctx Recall 1.0, Correctness .50
- **Noise Robustness 0.80, Negative Rejection 1.00**
- MAP@10 .390/MRR .409/Hit@10 .455 tổng (kéo xuống bởi out_of_domain=0 — đúng kỳ vọng vì không có tài liệu liên quan)
- Latency: P50 155.6s, avg 156.1s

## API (FastAPI 0.115)
- `POST /chat` — {message, history} → {answer, sources[], context_used}
- `GET /search?query=&top_k=` (1–20, default 5) — truy xuất không qua LLM
- `POST /export` — xuất hội thoại ra DOCX (2 định dạng)
- `GET /health`, `GET /`

## UI (Streamlit)
Chat interface, quản lý phiên (tạo/tải/xóa, lưu PostgreSQL chat_sessions/chat_messages, 6 lượt ngữ cảnh), hiển thị nguồn trích dẫn, export Word, giao diện tiếng Việt.

## Hạ tầng
Docker Compose: Qdrant (6333), PostgreSQL 16 (5434), MongoDB 7 (tùy chọn/legacy).

## Hạn chế (cho chương Kết luận)
- Correctness thấp (0.59 run 50 câu) — hallucination còn tồn tại (faithfulness 0.73)
- Latency ~2.2 phút/câu — chưa phù hợp production
- Chunking theo từ (512 words ≈ 700–900 tokens) vượt giới hạn tối ưu 512 token của BGE-M3 → cắt đuôi
- OCR nhiễu ở 3 PDF scan chất lượng thấp
- tuyen_sinh không có kết quả ở run 50 câu (lỗi cần điều tra)
