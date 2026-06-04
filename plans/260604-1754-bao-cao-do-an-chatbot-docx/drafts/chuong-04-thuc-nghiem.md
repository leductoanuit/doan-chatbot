# CHƯƠNG 4. THỰC NGHIỆM VÀ ĐÁNH GIÁ

## 4.1. Tập kiểm thử

### 4.1.1. Xây dựng tập kiểm thử

Đề tài xây dựng hai tập kiểm thử:

- **Tập kiểm thử chuẩn** (`eval-dataset.json`): gồm **100 câu hỏi** phủ 5 danh mục tri thức: tuyển sinh (20 câu), học vụ (25), học phí (15), chương trình đào tạo (20) và quy chế (20). Mỗi câu gồm: câu hỏi, đáp án chuẩn (ground truth), các đoạn văn bản tham chiếu (reference contexts), danh mục và độ khó (dễ/trung bình/khó). Câu hỏi dễ là tra cứu trực tiếp một dữ kiện; trung bình yêu cầu tổng hợp trong một tài liệu; khó yêu cầu tổng hợp nhiều tài liệu hoặc suy luận điều kiện. Khi đánh giá, tập được chia thành 2 batch, mỗi batch 50 câu.
- **Tập kiểm thử độ bền vững** (`eval-dataset-robustness.json`): gồm **11 câu hỏi** bổ sung trường ngữ cảnh nhiễu (noisy contexts), trong đó **5 câu nhiễu** (noise) trong miền dùng đo Noise Robustness và **6 câu ngoài miền tri thức** (out-of-domain, ví dụ hỏi về nấu ăn, thời tiết, giá vàng) dùng đo Negative Rejection — khả năng từ chối trả lời khi không có căn cứ.

Tập kiểm thử được sinh bán tự động: mô-đun sinh câu hỏi — đáp án từ nội dung tài liệu nguồn bằng LLM theo các mẫu câu hỏi định sẵn, sau đó rà soát thủ công.

### 4.1.2. Kiểm tra và xác thực tập kiểm thử

Mỗi cặp câu hỏi — đáp án được kiểm tra qua mô-đun xác thực tự động (qa_validator): đối chiếu đáp án với đoạn tham chiếu để loại các cặp đáp án không có căn cứ trong tài liệu; kiểm tra trùng lặp ngữ nghĩa giữa các câu hỏi; cân bằng phân bố theo danh mục và độ khó. Các câu không đạt được loại bỏ hoặc soạn lại thủ công, bảo đảm tập kiểm thử phản ánh trung thực nhu cầu hỏi đáp thực tế của người học hệ từ xa.

## 4.2. Các phương pháp đánh giá

Bộ khung đánh giá được thiết kế theo định hướng khảo sát Auepora, đo đồng thời hai mục tiêu truy xuất và tạo sinh, bổ sung khía cạnh độ bền vững theo benchmark RGB. Cấu hình thực nghiệm được tổng hợp ở Bảng 4.1.

Bảng 4.1: Cấu hình thực nghiệm đánh giá

| Hạng mục | Cấu hình |
|---|---|
| Tổng số câu hỏi đánh giá | 100 câu (2 batch × 50 câu) |
| Tập dữ liệu | eval-dataset.json (100 câu) |
| Mô hình embedding | BAAI/bge-m3 |
| Mô hình reranker | BAAI/bge-reranker-v2-m3 |
| Mô hình LLM (RAG) | gemini-2.5-flash |
| Mô hình LLM giám khảo | gemini-2.5-flash |
| Vector store | Qdrant (513 chunks) |
| Metadata store | PostgreSQL (19 nguồn tài liệu) |
| Khung đánh giá | RAGAS + Retrieval Rank Metrics + Robustness |

### 4.2.1. Đánh giá chất lượng truy xuất

Chất lượng xếp hạng của khâu truy xuất được đo bằng ba chỉ số trên top k=10 kết quả, với mức độ liên quan của mỗi kết quả xác định theo độ phủ (recall-based) so với đoạn tham chiếu:

- **MAP@10 (Mean Average Precision):** trung bình độ chính xác tại các vị trí xuất hiện kết quả liên quan, phản ánh đồng thời độ chính xác và thứ tự xếp hạng.
- **MRR (Mean Reciprocal Rank):** nghịch đảo thứ hạng của kết quả liên quan đầu tiên, phản ánh tốc độ "tìm trúng" tài liệu đúng.
- **Hit@10:** tỷ lệ câu hỏi có ít nhất một kết quả liên quan trong top 10.

### 4.2.2. Đánh giá chất lượng tạo sinh phản hồi

Chất lượng câu trả lời được đánh giá tự động theo phương pháp LLM-as-a-judge (giám khảo là Gemini 2.5 Flash) với năm chỉ số theo khung RAGAS, thang điểm 0–1:

- **Faithfulness (độ trung thực):** mức độ câu trả lời bám sát ngữ cảnh được truy xuất, đo lường hiện tượng ảo giác.
- **Answer Relevancy (độ liên quan):** mức độ câu trả lời đúng trọng tâm câu hỏi.
- **Context Precision (độ chính xác ngữ cảnh):** tỷ lệ ngữ cảnh truy xuất thực sự hữu ích cho câu trả lời.
- **Context Recall (độ bao phủ ngữ cảnh):** mức độ ngữ cảnh truy xuất bao phủ thông tin cần thiết so với đáp án chuẩn.
- **Correctness (độ đúng đắn):** mức độ trùng khớp nội dung giữa câu trả lời và đáp án chuẩn.

### 4.2.3. Đánh giá độ bền vững và độ trễ

Trên tập kiểm thử độ bền vững, hai chỉ số bổ sung được đo:

- **Noise Robustness:** khả năng trả lời đúng khi ngữ cảnh truy xuất bị trộn các đoạn nhiễu không liên quan.
- **Negative Rejection:** khả năng từ chối trả lời các câu hỏi ngoài miền tri thức thay vì bịa đặt.

Ngoài ra, độ trễ phản hồi (latency) được ghi nhận theo trung bình và các phân vị P50/P90/P99 cho toàn pipeline. Toàn bộ quá trình đánh giá được tự động hóa bằng bộ script trong thư mục `tests/evaluation`, xuất báo cáo JSON/Markdown/HTML/DOCX.

## 4.3. Kết quả đánh giá

### 4.3.1. Kết quả tổng hợp trên 100 câu

Kết quả trung bình của 2 batch trên toàn bộ 100 câu hỏi được trình bày ở Bảng 4.2.

Bảng 4.2: Kết quả các chỉ số tạo sinh tổng hợp (100 câu, trung bình 2 batch)

| Chỉ số | Điểm | Đánh giá |
|---|---|---|
| Faithfulness (độ trung thực) | 0,730 | Khá |
| Answer Relevancy (độ liên quan) | 0,745 | Khá |
| Context Precision (độ chính xác ngữ cảnh) | 0,846 | Khá |
| Context Recall (độ bao phủ ngữ cảnh) | 0,895 | Tốt |
| Correctness (độ đúng đắn) | 0,585 | Cần cải thiện |

### 4.3.2. Kết quả theo batch

**Batch 1 (câu 1–50).** Các chỉ số tạo sinh: Faithfulness 0,730; Answer Relevancy 0,730; Context Precision 0,940; Context Recall 0,920; Correctness 0,580. Các chỉ số xếp hạng truy xuất đạt rất cao: **MAP@10 = 0,926; MRR = 0,990; Hit@10 = 1,000** — với mọi câu hỏi, tài liệu liên quan đều xuất hiện trong top 10 và hầu như luôn đứng ở vị trí đầu tiên. Kết quả theo danh mục (Bảng 4.3) cho thấy cả 5 danh mục đều đạt Hit@10 tuyệt đối; riêng các chỉ số tạo sinh, danh mục học phí bất thường thấp (Faithfulness 0,400; Answer Relevancy 0,200) dù truy xuất tốt — cho thấy dữ liệu học phí trong kho tri thức chưa đủ chi tiết để mô hình trả lời đúng trọng tâm.

Bảng 4.3: Batch 1 — Chỉ số xếp hạng truy xuất theo danh mục (k=10)

| Danh mục | MAP@10 | MRR | Hit@10 |
|---|---|---|---|
| Chương trình đào tạo | 0,988 | 1,000 | 1,000 |
| Học phí | 0,963 | 1,000 | 1,000 |
| Học vụ | 0,901 | 1,000 | 1,000 |
| Quy chế | 0,898 | 1,000 | 1,000 |
| Tuyển sinh | 0,924 | 0,975 | 1,000 |

Bảng 4.4: Batch 1 — Chỉ số tạo sinh theo danh mục

| Danh mục | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| Chương trình đào tạo | 0,833 | 0,667 | 1,000 | 1,000 |
| Học phí | 0,400 | 0,200 | 1,000 | 1,000 |
| Học vụ | 0,786 | 0,857 | 1,000 | 0,929 |
| Quy chế | 0,800 | 0,900 | 1,000 | 0,600 |
| Tuyển sinh | 0,725 | 0,750 | 0,850 | 0,950 |

**Batch 2 (câu 51–100).** Các chỉ số tạo sinh: Faithfulness 0,730; Answer Relevancy 0,760; Context Precision 0,752; Context Recall 0,870; Correctness 0,590. Chỉ số truy xuất: **MAP@10 = 0,903; MRR = 0,990; Hit@10 = 1,000**. Phân tích theo danh mục (Bảng 4.5) cho thấy học vụ đạt Faithfulness và Answer Relevancy cao nhất (0,818) nhưng Context Precision thấp nhất (0,546); học phí có Context Recall tuyệt đối (1,0).

Bảng 4.5: Batch 2 — Chỉ số tạo sinh theo danh mục

| Danh mục | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| Chương trình đào tạo | 0,786 | 0,786 | 0,757 | 0,821 |
| Học phí | 0,600 | 0,750 | 0,900 | 1,000 |
| Học vụ | 0,818 | 0,818 | 0,546 | 0,909 |
| Quy chế | 0,700 | 0,700 | 0,800 | 0,800 |

Phân tích theo độ khó trên cả hai batch cho quy luật nhất quán: câu hỏi khó có Context Recall cao (batch 2 đạt 1,0) nhưng Context Precision thấp (0,556) — hệ thống tìm đủ tài liệu cần thiết nhưng kéo theo nhiều ngữ cảnh nhiễu; câu trung bình có Faithfulness thấp nhất ở batch 2 (0,643); câu dễ cân bằng nhất trên các chỉ số.

Bảng 4.6: Kết quả theo độ khó (batch 2)

| Độ khó | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| Dễ | 0,759 | 0,796 | 0,800 | 0,796 |
| Trung bình | 0,643 | 0,750 | 0,786 | 0,929 |
| Khó | 0,778 | 0,667 | 0,556 | 1,000 |

### 4.3.3. So sánh hai batch

Bảng 4.7: So sánh chỉ số giữa hai batch

| Chỉ số | Batch 1 (Q1–50) | Batch 2 (Q51–100) | Chênh lệch |
|---|---|---|---|
| Faithfulness | 0,730 | 0,730 | 0,000 |
| Answer Relevancy | 0,730 | 0,760 | +0,030 |
| Context Precision | 0,940 | 0,752 | −0,188 |
| Context Recall | 0,920 | 0,870 | −0,050 |
| Correctness | 0,580 | 0,590 | +0,010 |
| MAP@10 | 0,926 | 0,903 | −0,023 |
| MRR | 0,990 | 0,990 | 0,000 |
| Hit@10 | 1,000 | 1,000 | 0,000 |

Kết quả hai batch nhất quán ở các chỉ số Faithfulness, Correctness và các chỉ số xếp hạng truy xuất, cho thấy độ ổn định của hệ thống. Khác biệt đáng kể nhất là Context Precision giảm 0,188 ở batch 2 — nhóm 50 câu sau chứa nhiều câu hỏi khó về truy xuất hơn, kéo vào nhiều ngữ cảnh nhiễu; tuy nhiên MRR và Hit@10 vẫn giữ mức tối đa, nghĩa là tài liệu đúng vẫn luôn được tìm thấy và xếp hạng đầu.

### 4.3.4. Kết quả đánh giá độ bền vững

Trên tập kiểm thử độ bền vững 11 câu (5 câu nhiễu + 6 câu ngoài miền), kết quả được trình bày ở Bảng 4.8.

Bảng 4.8: Kết quả đánh giá độ bền vững (11 câu)

| Chỉ số | Điểm | Đánh giá |
|---|---|---|
| Faithfulness | 0,800 | Tốt |
| Answer Relevancy | 1,000 | Tốt |
| Context Precision | 1,000 | Tốt |
| Context Recall | 1,000 | Tốt |
| Correctness | 0,500 | Cần cải thiện |
| Noise Robustness | 0,800 | Tốt |
| Negative Rejection | 1,000 | Tốt |

Ba nhận xét chính:

- **Negative Rejection = 1,0:** hệ thống từ chối đúng 100% câu hỏi ngoài miền tri thức (nấu ăn, thời tiết, giá vàng...) thay vì bịa đặt câu trả lời — năng lực then chốt với chatbot đại diện đơn vị giáo dục.
- **Noise Robustness = 0,8:** hệ thống xử lý tốt câu hỏi có ngữ cảnh nhiễu, chỉ 1/5 câu bị ảnh hưởng.
- Chỉ số xếp hạng truy xuất tổng trên tập này thấp (MAP@10 = 0,390; MRR = 0,409; Hit@10 = 0,455) do nhóm câu ngoài miền có MAP = 0 — đây là kết quả đúng kỳ vọng vì các câu này vốn không có tài liệu liên quan trong kho tri thức; riêng các câu trong miền vẫn đạt cao (học vụ: MAP 1,0; tuyển sinh: MAP 0,823, Hit@10 1,0).

## 4.4. Đánh giá độ phản hồi của hệ thống

Độ trễ phản hồi toàn pipeline của hai batch được ghi nhận ở Bảng 4.9.

Bảng 4.9: Độ trễ phản hồi của hệ thống

| Chỉ số | Batch 1 | Batch 2 |
|---|---|---|
| Trung bình | 403,7 giây (~6,7 phút) | 130,6 giây (~2,2 phút) |
| P50 | 416,6 giây | 130,1 giây |
| P90 | 445,5 giây | 152,2 giây |
| P99 | 495,1 giây | 166,9 giây |
| Nhỏ nhất | 149,2 giây | 94,0 giây |
| Lớn nhất | 495,1 giây | 166,9 giây |

Độ trễ trung bình ở batch 1 lên tới ~6,7 phút/câu, chủ yếu do mô hình reranker chạy trên CPU; batch 2 giảm còn ~2,2 phút/câu. Mức độ này chấp nhận được cho mục đích minh họa và đánh giá, nhưng cần tối ưu đáng kể (triển khai reranker trên GPU, streaming phản hồi, cache, giảm số lần gọi LLM) trước khi triển khai thực tế.

**Nhận xét chung:** hệ thống thể hiện thế mạnh rõ rệt ở khâu truy xuất — MRR 0,99 và Hit@10 1,0 trên cả 100 câu, Context Recall 0,87–0,92 — và khả năng kiểm soát phạm vi tuyệt đối (Negative Rejection 1,0). Điểm yếu tập trung ở khâu tạo sinh: Correctness chỉ đạt 0,58–0,59 dù truy xuất tốt, danh mục học phí ở batch 1 đạt điểm tạo sinh rất thấp cho thấy cần bổ sung dữ liệu học phí chi tiết hơn, và độ trễ còn cao do reranker chạy trên CPU. Các hướng khắc phục được thảo luận tại Chương 6.
