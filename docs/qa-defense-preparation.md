# Chuẩn bị câu hỏi bảo vệ đồ án

Hệ thống: **Chatbot tư vấn đào tạo từ xa UIT** — RAG + Gemini + Qdrant + BGE-M3

---

## 1. Tổng quan & Kiến trúc

**Q: Hệ thống của bạn hoạt động như thế nào? Mô tả flow từ đầu đến cuối.**

A: Flow gồm 2 giai đoạn:

**Offline (tiền xử lý):**
```
Tài liệu PDF/DOCX → OCR (Tesseract) → JSON → Chunking → BGE-M3 Embedding → Qdrant + PostgreSQL
```

**Online (inference):**
```
User query → Query Expansion → Hybrid Search (Vector + Keyword) → Context Assembly → Gemini API → Response
```

---

**Q: Tại sao chọn kiến trúc RAG thay vì fine-tune LLM?**

A:
- **Fine-tune** đòi hỏi dữ liệu labeled lớn, chi phí GPU cao, và cần retrain mỗi khi tài liệu thay đổi.
- **RAG** cho phép cập nhật kiến thức chỉ bằng cách re-embed tài liệu mới, không cần retrain model.
- Với bài toán tư vấn học vụ — nội dung thay đổi theo năm học — RAG phù hợp hơn về mặt bảo trì.
- RAG cũng cho phép trích dẫn nguồn cụ thể (điều, khoản), tăng tính minh bạch.

---

**Q: Sự khác biệt giữa Vector Search và Keyword Search trong hệ thống của bạn?**

A:
- **Vector Search** (Qdrant, cosine similarity): tìm kiếm theo nghĩa — `"lúc nào tuyển sinh"` khớp với `"khi nào tổ chức tuyển sinh"`. Chiếm 70% trọng số.
- **Keyword Search** (BM25-style, Qdrant full-text): tìm kiếm theo từ khóa chính xác — đảm bảo không bỏ sót khi user dùng đúng thuật ngữ kỹ thuật. Chiếm 30% trọng số.
- Kết hợp hai loại (Hybrid Search) giúp xử lý được cả câu hỏi colloquial lẫn câu hỏi kỹ thuật.

---

## 2. Embedding Model

**Q: Tại sao chọn BAAI/bge-m3 thay vì các model khác?**

A:
- Model trước (`dangvantuan/vietnamese-embedding`, PhoBERT-based, 768-dim) chủ yếu dựa vào lexical similarity — không hiểu đồng nghĩa.
- `BAAI/bge-m3` là state-of-the-art multilingual embedding model, được train trên nhiều ngôn ngữ kể cả tiếng Việt, hỗ trợ semantic understanding tốt hơn.
- Kết quả thực tế: query `"lúc nào tuyển sinh"` với bge-m3 tìm đúng chunk `"Khi nào tổ chức tuyển sinh"` (score 0.661), trong khi PhoBERT trả về sai chunk (score 0.632).
- Nhược điểm: nặng hơn (~2GB), chậm hơn khi embed, nhưng chỉ chạy offline nên chấp nhận được.

---

**Q: `normalize_embeddings=True` trong BGE-M3 có ý nghĩa gì?**

A: BGE-M3 được thiết kế để dùng với cosine similarity. Normalize đưa vector về unit length (|v|=1), khi đó cosine similarity = dot product — tính toán nhanh hơn và kết quả ổn định hơn. Không normalize sẽ khiến score bị lệch theo magnitude của vector.

---

## 3. Chunking & Storage

**Q: Bạn chia chunk tài liệu như thế nào? Tại sao?**

A: Xem `src/embedding/chunker.py`. Chunk size **512 từ**, overlap **100 từ**. Logic ưu tiên:
1. Split theo **paragraph** (`\n\n`) — không cắt ngang đoạn văn.
2. Nếu paragraph quá dài → split tiếp theo **câu** (`. `).
3. Mỗi chunk mang theo **100 từ cuối** của chunk trước để đảm bảo context liên tục qua ranh giới chunk.

Thực tế, hầu hết trang tài liệu UIT < 512 từ → **1 trang = 1 chunk**, overlap chỉ kích hoạt khi trang dài hơn 512 từ. BGE-M3 hỗ trợ tối đa 512 tokens — phù hợp với chunk size này.

---

**Q: Tại sao dùng cả Qdrant lẫn PostgreSQL? Một cơ sở dữ liệu không đủ sao?**

A:
- **Qdrant**: chuyên biệt cho vector search — hỗ trợ HNSW index, filtered search theo metadata, tốc độ query nhanh. Không phù hợp để lưu metadata dạng relational.
- **PostgreSQL**: lưu metadata dạng relational (document_id, title, issuing_body, issue_date, document_type) — cho phép query phức tạp, join, filter. Không hỗ trợ vector search hiệu quả.
- Kết hợp: Qdrant lưu vector + payload cơ bản, PostgreSQL lưu metadata đầy đủ → tối ưu từng loại cho đúng use case.

---

## 4. Query Processing

**Q: Query expansion là gì? Tại sao cần?**

A: Query expansion thêm context vào query ngắn để cải thiện retrieval. Ví dụ: `"học phí bao nhiêu"` → `"học phí bao nhiêu (hệ đào tạo từ xa UIT)"`.

Tuy nhiên, expansion **phải có điều kiện**: nếu query đã chứa domain keywords như `"quy chế"`, `"học vụ"`, `"sinh viên"` thì không expand. Nếu expand mù quáng, query như `"xử lý học vụ"` sẽ bị nhiễu từ `"đào tạo từ xa"` và trả về sai tài liệu (đã gặp và đã fix).

---

**Q: `top_k` là gì và bạn chọn giá trị nào?**

A: `top_k` = số lượng chunks đưa vào context cho LLM. Tăng top_k → nhiều thông tin hơn nhưng tốn token và có thể gây nhiễu. Hiện tại `top_k=10` sau khi thực nghiệm: với top_k=5, file quy chế 507-QĐ (rank 6) bị bỏ qua; top_k=10 đảm bảo file này luôn được đưa vào context.

---

**Q: Ngưỡng cosine similarity để lấy chunk là bao nhiêu? Cách tính final_score?**

A: Hybrid search tính **final_score** cho từng chunk riêng lẻ (không trung bình):

| Loại | Score gốc | Trọng số | final_score |
|------|-----------|----------|-------------|
| Vector search | cosine similarity (0–1) | × 0.7 | cosine × 0.7 |
| Keyword search | cố định 0.5 | × 0.3 | 0.5 × 0.3 = **0.15** |

**Ví dụ**: chunk có cosine = 0.661 → final_score = 0.661 × 0.7 = **0.463**

**Filter ngưỡng** — áp dụng từng chunk độc lập:
```
threshold = MIN_SCORE × vector_weight = 0.25 × 0.7 = 0.175

chunk A: cosine=0.66 → final=0.462 ✅ giữ
chunk B: cosine=0.30 → final=0.210 ✅ giữ
chunk C: cosine=0.20 → final=0.140 ❌ loại
```
- Tương đương cosine score tối thiểu = **0.25**
- Keyword-only chunks (final_score = 0.15) luôn dưới ngưỡng → chỉ xuất hiện qua fallback
- Nếu không có chunk nào đạt ngưỡng → fallback trả về top-k để Gemini tự đánh giá

Kết quả cuối sort giảm dần theo final_score, lấy top-k.

---

**Q: Trọng số 0.7 và 0.3 trong hybrid search có thể thay đổi không?**

A: Có — là tham số mặc định trong hàm `hybrid_search()`, có thể truyền giá trị khác khi gọi:

```python
retriever.hybrid_search(query, vector_weight=0.5, keyword_weight=0.5)
```

Chọn **70/30** vì tài liệu pháp lý tiếng Việt cần semantic understanding là chủ yếu; keyword chỉ hỗ trợ khi user dùng đúng thuật ngữ kỹ thuật (ví dụ: "507-QĐ", "Điều 16"). Nếu tài liệu có nhiều số liệu/mã code → nên tăng keyword weight.

---

## 5. LLM & Generation

**Q: Tại sao chọn Gemini thay vì GPT-4 hay các model khác?**

A:
- Gemini API có free tier đủ dùng cho demo/prototype.
- `gemini-2.0-flash` cân bằng tốt giữa tốc độ và chất lượng.
- Hỗ trợ tiếng Việt tốt, context window lớn (1M tokens).
- GPT-4 tốt hơn nhưng tốn chi phí hơn — không phù hợp cho đồ án sinh viên.

---

**Q: System prompt của bạn làm gì?**

A: System prompt định nghĩa:
1. **Vai trò**: tư vấn viên UIT hệ đào tạo từ xa.
2. **Phạm vi**: chỉ tư vấn đào tạo từ xa, từ chối câu hỏi hệ chính quy.
3. **Độ ưu tiên nguồn**: Quy chế UIT (507-QĐ) > Thông tư Bộ GD&ĐT > FAQ — đảm bảo trích dẫn quy định trực tiếp của trường thay vì quy định cấp Bộ chung chung.
4. **Tone**: thân thiện, ngắn gọn, có cấu trúc.

---

## 6. Đánh giá & Hạn chế

**Q: Bạn đánh giá chất lượng chatbot như thế nào?**

A: Hiện tại đánh giá định tính qua:
- Kiểm tra manual các câu hỏi điển hình (tuyển sinh, học vụ, CTĐT).
- Kiểm tra score retrieval — chunk đúng có score ≥ 0.55.
- So sánh nguồn trích dẫn với tài liệu gốc.

Chưa có đánh giá định lượng (RAGAS, MRR, NDCG) — đây là hướng cải thiện trong tương lai.

---

**Q: Hạn chế lớn nhất của hệ thống là gì?**

A:
1. **Chất lượng OCR**: PDF scan có thể bị nhận sai chữ (dấu tiếng Việt), ảnh hưởng embedding quality.
2. **Chunk boundary**: nội dung 1 điều khoản trải dài qua 2 trang → bị cắt thành 2 chunks, mỗi chunk thiếu context.
3. **Hallucination**: Gemini đôi khi thêm thông tin không có trong tài liệu — cần cải thiện prompt để ràng buộc chặt hơn.
4. **Không có memory dài hạn**: conversation history chỉ giữ 6 turns gần nhất.
5. **Tài liệu chưa đầy đủ**: một số file như lịch học kỳ, thông báo mới chưa được cập nhật.

---

**Q: Nếu có thêm thời gian, bạn sẽ cải thiện điều gì?**

A:
1. **Đánh giá định lượng**: implement RAGAS pipeline để đo Faithfulness, Answer Relevancy, Context Precision.
2. **Re-ranking**: dùng Cross-Encoder để re-rank top-k chunks sau retrieval.
3. **Chunking thông minh hơn**: chunk theo điều khoản (semantic chunking) thay vì theo trang.
4. **Update pipeline tự động**: khi có tài liệu mới → tự động embed và upsert vào Qdrant.
5. **Guardrails**: phát hiện và từ chối câu hỏi ngoài phạm vi chính xác hơn.

---

## 7. Công nghệ & Triển khai

**Q: Mô tả tech stack của hệ thống.**

A:

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Backend API | FastAPI + Uvicorn |
| Embedding | BAAI/bge-m3 (1024-dim, local) |
| Vector Store | Qdrant (Docker) |
| Metadata Store | PostgreSQL (Docker) |
| LLM | Gemini 2.0 Flash (API) |
| OCR | Tesseract + OpenCV deskew |
| Tunnel | ngrok |

---

**Q: Vì sao dùng FastAPI tách biệt thay vì để Streamlit gọi trực tiếp?**

A: Thực tế trong codebase hiện tại, Streamlit import trực tiếp `RAGPipeline` (không qua API). FastAPI tồn tại để:
- Cung cấp REST API cho các client khác (mobile app, web app khác).
- Expose endpoint `/docs` (Swagger) để demo và test.
- Tách biệt concern: frontend và business logic độc lập nhau.

---

**Q: Dữ liệu của bạn gồm những gì?**

A: 15 nguồn tài liệu, 269 documents, 300 chunks sau khi embed:
- Quy chế đào tạo UIT (507-QĐ-2024)
- Thông tư Bộ GD&ĐT (28/2023, 21/2019, 17/2016)
- Chương trình đào tạo TTNT và CNTT
- FAQ tuyển sinh, hồ sơ tuyển sinh
- Quy trình biểu mẫu chứng chỉ CNTT
- Thông báo khai giảng
