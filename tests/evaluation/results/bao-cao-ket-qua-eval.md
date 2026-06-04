# Báo Cáo Kết Quả Đánh Giá RAG Chatbot UIT

**Ngày tạo:** 2026-06-04  
**Số câu hỏi:** 50 câu/lần chạy  

---

## Tổng Quan 2 Lần Chạy

| | Lần 1 (17:43) | Lần 2 (19:36) |
|---|---|---|
| **Thời điểm** | 2026-06-02 17:43:29 | 2026-06-02 19:36:53 |
| **Judge model** | gemini-2.0-flash-001 | gemini-2.5-flash |
| **Tổng câu hỏi** | 50 | 50 |
| **Trạng thái** | ❌ Thất bại (toàn 0.0) | ✅ Thành công |

> **Lần 1 bị lỗi** — toàn bộ metrics trả về 0.0, không có dữ liệu hợp lệ. Phân tích chỉ dựa trên **Lần 2**.

---

## Kết Quả Lần 2 — Metrics Tổng Thể

| Metric | Điểm | Đánh giá |
|---|---|---|
| **Faithfulness** | 0.730 | 🟡 Trung bình |
| **Answer Relevancy** | 0.760 | 🟡 Trung bình |
| **Context Precision** | 0.752 | 🟡 Trung bình |
| **Context Recall** | 0.870 | 🟢 Tốt |
| **Correctness** | 0.590 | 🔴 Cần cải thiện |

**Điểm trung bình tổng hợp:** `0.740`

---

## Phân Tích Theo Danh Mục

| Danh mục | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| **ctdt** (Chương trình đào tạo) | 0.786 | 0.786 | 0.757 | 0.821 |
| **hoc_phi** (Học phí) | 0.600 | 0.750 | 0.900 | **1.000** |
| **hoc_vu** (Học vụ) | **0.818** | **0.818** | 0.546 | 0.909 |
| **quy_che** (Quy chế) | 0.700 | 0.700 | 0.800 | 0.800 |
| **tuyen_sinh** (Tuyển sinh) | N/A | N/A | N/A | N/A |

> **tuyen_sinh** không có dữ liệu trong Lần 2 — có thể bị lỗi hoặc không có câu hỏi.

**Nhận xét:**
- `hoc_vu` đạt faithfulness & relevancy cao nhất (0.818) nhưng context_precision thấp nhất (0.546)
- `hoc_phi` có context_recall hoàn hảo (1.0) nhưng faithfulness thấp (0.600)
- `ctdt` và `quy_che` ổn định ở mức trung bình

---

## Phân Tích Theo Độ Khó

| Độ khó | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| **easy** | 0.759 | 0.796 | 0.800 | 0.796 |
| **medium** | 0.643 | 0.750 | 0.786 | 0.929 |
| **hard** | **0.778** | 0.667 | 0.556 | **1.000** |

**Nhận xét:**
- Câu hỏi **hard** có context_recall cao nhất (1.0) nhưng context_precision thấp (0.556) → retrieval tìm đủ nhưng kém chính xác
- Câu hỏi **medium** có faithfulness thấp nhất (0.643)
- Câu hỏi **easy** cân bằng nhất

---

## Latency (Thời Gian Phản Hồi)

| Metric | Lần 1 (thất bại) | Lần 2 (thành công) |
|---|---|---|
| **P50** | 411,910 ms (~6.9 phút) | 130,062 ms (~2.2 phút) |
| **P90** | 443,539 ms (~7.4 phút) | 152,207 ms (~2.5 phút) |
| **P99** | 521,465 ms (~8.7 phút) | 166,905 ms (~2.8 phút) |
| **Avg** | 403,381 ms (~6.7 phút) | 130,551 ms (~2.2 phút) |
| **Min** | 275,760 ms | 94,022 ms |
| **Max** | 521,465 ms | 166,905 ms |

**Latency theo danh mục (Lần 2):**

| Danh mục | Avg (ms) | Avg (giây) |
|---|---|---|
| ctdt | 117,619 | ~117s |
| hoc_phi | 120,783 | ~121s |
| hoc_vu | 145,567 | ~146s |
| quy_che | 138,121 | ~138s |

> Lần 2 nhanh hơn **~3x** so với Lần 1 — do đổi judge model sang `gemini-2.5-flash`.  
> Latency vẫn cao (~2 phút/câu) — cần xem xét tối ưu nếu dùng thực tế.

---

## Câu Hỏi Kém Nhất (Worst Context Recall — Lần 2)

| Câu hỏi | Danh mục | Context Recall | Faithfulness |
|---|---|---|---|
| Tôi có thể tìm thông tin về lịch học ở đâu? | ctdt | 0.5 | 0.0 |
| Mình có thể đến địa điểm nào để nhận thẻ sinh viên... | hoc_vu | 1.0 | 0.0 |

> **Vấn đề:** Một số câu có context_recall tốt nhưng faithfulness = 0 → chatbot trả lời không bám sát vào tài liệu được retrieve.

---

## Tóm Tắt & Khuyến Nghị

### Điểm mạnh
- **Context Recall** tốt (0.87) → hệ thống retrieve đúng tài liệu liên quan
- **hoc_phi** category ổn định nhất
- Latency Lần 2 cải thiện đáng kể so với Lần 1

### Điểm yếu
- **Correctness** thấp nhất (0.59) → câu trả lời chưa đúng về mặt nội dung
- **Faithfulness** trung bình (0.73) → vẫn có trường hợp hallucination
- **Context Precision** trung bình (0.752) → retrieve còn nhiễu, kéo vào context không liên quan
- **tuyen_sinh** không có kết quả → cần điều tra lỗi

### Khuyến nghị
1. **Tăng Correctness**: Cải thiện prompt hướng dẫn model trả lời chính xác hơn
2. **Giảm hallucination**: Thêm instruction "chỉ trả lời dựa trên context được cung cấp"
3. **Cải thiện Context Precision**: Điều chỉnh chunking strategy hoặc reranker
4. **Điều tra tuyen_sinh**: Tìm hiểu tại sao category này không có kết quả Lần 2
5. **Giảm latency**: Xem xét streaming hoặc caching cho production

---

*Báo cáo được tạo từ: `eval-summary-20260602_174329.json` & `eval-summary-20260602_193653.json`*
