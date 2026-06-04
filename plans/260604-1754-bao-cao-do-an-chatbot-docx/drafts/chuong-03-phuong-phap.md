# CHƯƠNG 3. PHƯƠNG PHÁP XÂY DỰNG VÀ TRIỂN KHAI

## 3.1. Quy trình thực hiện

Đề tài được thực hiện theo quy trình gồm năm giai đoạn nối tiếp:

1. **Thu thập và xử lý dữ liệu:** thu thập các văn bản quy định về hệ đào tạo từ xa UIT; trích xuất văn bản từ PDF (gồm cả tài liệu scan bằng OCR), DOCX và dữ liệu web; làm sạch, chuẩn hóa và gắn metadata.
2. **Xây dựng cơ sở dữ liệu vector:** phân đoạn văn bản (chunking), sinh vector nhúng bằng BGE-M3 và đánh chỉ mục vào Qdrant kèm chỉ mục metadata.
3. **Xây dựng hệ thống RAG:** cài đặt bốn giai đoạn tiền xử lý truy vấn — truy xuất — sàng lọc — tổng hợp, cùng các kỹ thuật prompt engineering.
4. **Thực nghiệm và đánh giá:** xây dựng tập kiểm thử, cài đặt bộ khung đánh giá tự động và phân tích kết quả để tinh chỉnh hệ thống.
5. **Cài đặt chương trình minh họa:** xây dựng REST API (FastAPI) và giao diện trò chuyện (Streamlit).

[Hình 3.1: Quy trình thực hiện đề tài — sẽ bổ sung]

## 3.2. Kiến trúc hệ thống và môi trường triển khai

Hệ thống được tổ chức theo kiến trúc phân tầng gồm: tầng giao diện (Streamlit), tầng dịch vụ API (FastAPI), tầng lõi RAG (tiền xử lý truy vấn, truy xuất lai, rerank, tổng hợp) và tầng lưu trữ (Qdrant lưu vector, PostgreSQL lưu metadata tài liệu và lịch sử hội thoại). Mô hình ngôn ngữ lớn Gemini được gọi qua API của Google; các mô hình nhúng và rerank chạy cục bộ.

[Hình 3.2: Kiến trúc tổng thể hệ thống chatbot RAG — sẽ bổ sung]

[Hình 3.3: Luồng xử lý một truy vấn trong hệ thống — sẽ bổ sung]

Môi trường triển khai sử dụng Docker Compose để khởi chạy các dịch vụ hạ tầng: Qdrant (cổng 6333), PostgreSQL 16 (cổng 5434) và MongoDB 7 (tùy chọn, phục vụ tương thích llama-index). Mã nguồn viết bằng Python 3.12, tổ chức theo các mô-đun: `scraper` (thu thập, trích xuất), `data_processing` (lọc văn bản hiệu lực), `embedding` (chunking, nhúng, ingest), `rag` (pipeline hỏi đáp), `api` (REST API), `frontend` (giao diện), `storage` (kết nối Qdrant/PostgreSQL).

## 3.3. Xây dựng bộ dữ liệu

### 3.3.1. Xây dựng tập dữ liệu

**Nguồn dữ liệu.** Đề tài thu thập 19 nguồn tài liệu chính thống về hệ đào tạo từ xa UIT, chia thành bốn nhóm:

- *Văn bản quy phạm của Bộ GD&ĐT:* Thông tư 28/2023/TT-BGDĐT (quy chế đào tạo từ xa), Thông tư 21/2019/TT-BGDĐT, Thông tư liên tịch 17/2016/TTLT-BGDĐT-BTTTT.
- *Văn bản của nhà trường:* Quyết định 1499/QĐ-ĐHCNTT (2024), 507/QĐ-ĐHCNTT (2024), 213/QĐ-ĐHCNTT (2024), 790/QĐ-ĐHCNTT (2022) về quy chế đào tạo.
- *Tài liệu chương trình và tuyển sinh:* chương trình đào tạo Cử nhân CNTT (văn bằng 1, liên thông), Cử nhân Trí tuệ Nhân tạo hệ từ xa (áp dụng từ khóa 2024), hồ sơ tuyển sinh, các câu hỏi thường gặp.
- *Dữ liệu web:* quy trình, biểu mẫu, chứng chỉ thu thập từ trang Trung tâm CITD bằng công cụ scraper tự xây dựng.

**Trích xuất văn bản.** Do tài liệu không thuần nhất, đề tài xây dựng pipeline trích xuất ba nhánh:

- PDF dạng văn bản (native): trích xuất trực tiếp bằng PyMuPDF.
- PDF dạng scan: kết xuất từng trang thành ảnh, tiền xử lý bằng OpenCV (khử nghiêng bằng minAreaRect, nhị phân hóa thích nghi adaptiveThreshold) rồi nhận dạng ký tự bằng Tesseract OCR với gói ngôn ngữ tiếng Việt.
- DOCX và JSON: đọc bằng python-docx và bộ phân tích cấu trúc riêng.

**Làm sạch và chuẩn hóa.** Văn bản OCR chứa nhiều nhiễu (ký tự lỗi, dòng vô nghĩa do trang scan mờ, nghiêng), đề tài cài đặt bước làm sạch tự động: chuẩn hóa bảng mã, loại ký tự lạc, lọc các dòng có tỷ lệ nhiễu cao. Kết quả khảo sát cho thấy 3 tệp PDF scan có mức nhiễu trung bình 0,26–0,41 (cao nhất 1,45 ở các trang mờ), được xử lý qua bước làm sạch chuyên biệt. Song song, đề tài lọc giữ lại các văn bản còn hiệu lực (mô-đun lọc quy định mới nhất) để tránh xung đột giữa quy chế cũ và mới.

**Gắn metadata.** Mỗi bản ghi văn bản được gắn metadata: nguồn, số trang, mã tài liệu, tiêu đề, số hiệu văn bản, ngày ban hành, cơ quan ban hành, loại văn bản và loại hệ thống (system_type). Phần metadata mô tả được sinh bán tự động bằng Gemini 2.0 Flash (temperature 0,1) và rà soát thủ công. Kết quả cuối cùng là tập `all_documents_final.json` gồm **448 bản ghi theo trang** từ 19 nguồn.

[Hình 3.4: Pipeline thu thập và xử lý dữ liệu — sẽ bổ sung]

### 3.3.2. Xây dựng cơ sở dữ liệu vector

**Phân đoạn văn bản (chunking).** Văn bản mỗi bản ghi được phân đoạn theo chiến lược đệ quy: ưu tiên tách theo đoạn văn, nếu đoạn quá dài thì tách tiếp theo câu; kích thước mỗi chunk xấp xỉ 512 từ với phần gối đầu (overlap) 100 từ (~19,5%) nhằm bảo toàn ngữ cảnh tại ranh giới. Toàn bộ kho dữ liệu sau phân đoạn gồm **513 chunks**.

**Sinh vector nhúng.** Mỗi chunk được mã hóa thành vector 1024 chiều bằng mô hình **BAAI/bge-m3** — mô hình nhúng đa ngôn ngữ hỗ trợ tốt tiếng Việt, với chuẩn hóa L2 để dùng độ đo cosine; quá trình nhúng thực hiện theo lô (batch size 16).

**Đánh chỉ mục Qdrant.** Vector được lưu vào collection `documents` của Qdrant với độ đo cosine. Mã định danh điểm (point ID) được sinh tất định từ bộ ba (mã tài liệu, trang, chỉ số chunk) giúp ingest lặp lại không tạo bản ghi trùng. Các trường `document_type`, `system_type`, `document_id` được tạo chỉ mục payload dạng keyword phục vụ lọc nhanh theo metadata. Nội dung đầy đủ của chunk cùng metadata được lưu song song trong PostgreSQL.

[Hình 3.5: Quy trình xây dựng cơ sở dữ liệu vector — sẽ bổ sung]

## 3.4. Xây dựng hệ thống RAG

Hệ thống RAG được tổ chức thành bốn giai đoạn nối tiếp: tiền xử lý truy vấn, truy xuất, sàng lọc và tổng hợp.

### 3.4.1. Tiền xử lý truy vấn

Giai đoạn này biến truy vấn thô của người dùng thành truy vấn chuẩn hóa, sẵn sàng cho truy xuất:

- **Nhận diện chitchat:** các câu chào hỏi, xã giao được phát hiện bằng tập mẫu regex và trả lời trực tiếp bởi LLM, bỏ qua toàn bộ pipeline truy xuất để tiết kiệm chi phí và thời gian.
- **Viết lại truy vấn theo ngữ cảnh:** với hội thoại nhiều lượt, câu hỏi thường chứa đại từ hoặc tín hiệu nối tiếp ("vậy còn...", "cái đó..."). Hệ thống phát hiện các tín hiệu này và dùng LLM (temperature 0,1) kết hợp 3 lượt hội thoại gần nhất để viết lại thành câu hỏi độc lập, đầy đủ ngữ nghĩa.
- **Phân loại ý định:** truy vấn được phân loại bằng đối sánh từ khóa vào các nhóm (đào tạo, tuyển sinh, chứng chỉ) để sinh bộ lọc metadata tương ứng, thu hẹp không gian truy xuất.

### 3.4.2. Truy xuất

Khâu truy xuất áp dụng chiến lược lai (hybrid retrieval) kết hợp hai kênh:

- **Tìm kiếm vector:** truy vấn được nhúng bằng BGE-M3 và tìm k=10 láng giềng gần nhất trong Qdrant (ANN, cosine), trọng số 0,7.
- **Tìm kiếm từ khóa:** đối sánh unigram và bigram của truy vấn với nội dung chunk, trọng số 0,3, bắt các trường hợp thuật ngữ chính xác (số hiệu văn bản, tên môn học) mà tìm kiếm ngữ nghĩa có thể bỏ sót.

Kết quả hai kênh được hợp nhất theo tổng trọng số, khử trùng lặp và loại các kết quả dưới ngưỡng điểm tối thiểu 0,25. Với câu hỏi phức tạp hoặc câu so sánh, hệ thống kích hoạt **mở rộng đa truy vấn (RAG Fusion)**: LLM sinh nhiều biến thể của truy vấn gốc, truy xuất song song rồi hợp nhất kết quả; riêng câu so sánh còn được tăng gấp đôi số lượng kết quả truy xuất (top_k×2). Nếu bộ lọc ý định khiến số kết quả quá ít (<3), hệ thống tự động truy xuất lại không kèm bộ lọc (cơ chế fallback).

### 3.4.3. Sàng lọc

Danh sách ứng viên sau truy xuất (top 10) được sàng lọc tinh bằng mô hình cross-encoder **BAAI/bge-reranker-v2-m3**: mô hình nhận trực tiếp cặp (truy vấn, chunk) và chấm điểm mức độ liên quan với độ chính xác cao hơn nhiều so với điểm tương đồng vector thuần túy, từ đó chọn ra các chunk tốt nhất (top 5) đưa vào ngữ cảnh. Cơ chế hai giai đoạn "truy xuất rộng — sàng lọc tinh" cân bằng giữa độ phủ và độ chính xác.

### 3.4.4. Tổng hợp

Các chunk sau sàng lọc được ghép thành ngữ cảnh (giới hạn ~1500 token, câu so sánh được nới gấp đôi), mỗi đoạn kèm nhãn nguồn dạng [Nguồn: tên tài liệu]. Ngữ cảnh cùng câu hỏi được đưa vào Gemini 2.5 Flash với system prompt suy luận chuỗi tiếng Việt, yêu cầu mô hình chỉ trả lời dựa trên ngữ cảnh, trích dẫn nguồn, trình bày dạng bảng khi so sánh và từ chối lịch sự khi câu hỏi ngoài phạm vi. Câu trả lời cuối cùng được trả về kèm danh sách nguồn tài liệu và lưu vào lịch sử hội thoại.

## 3.5. Lựa chọn và cấu hình mô hình

Bảng 3.1 tổng hợp các mô hình và cấu hình chính của hệ thống.

| Thành phần | Mô hình / Công nghệ | Cấu hình chính |
|---|---|---|
| Mô hình nhúng | BAAI/bge-m3 | 1024 chiều, chuẩn hóa L2, batch 16 |
| Mô hình rerank | BAAI/bge-reranker-v2-m3 | Cross-encoder, top 10 → top 5 |
| Mô hình sinh | Google Gemini 2.5 Flash | temperature 0,3; max_tokens 1024 |
| Viết lại truy vấn | Gemini 2.5 Flash | temperature 0,1 |
| Mở rộng đa truy vấn | Gemini 2.5 Flash | temperature 0,5 |
| CSDL vector | Qdrant | cosine, collection `documents`, payload index |
| Chunking | Đệ quy đoạn/câu | 512 từ/chunk, overlap 100 từ |
| Truy xuất lai | Vector + từ khóa | trọng số 0,7/0,3; k=10; ngưỡng 0,25 |

Việc chọn Gemini 2.5 Flash xuất phát từ cân bằng giữa chất lượng tiếng Việt, tốc độ và chi phí API; temperature thấp (0,1–0,3) cho các tác vụ cần tính nhất quán (viết lại truy vấn, trả lời), cao hơn (0,5) cho tác vụ cần đa dạng (sinh biến thể truy vấn). BGE-M3 được chọn sau khi cân nhắc các mô hình nhúng đa ngôn ngữ phổ biến nhờ hiệu năng tốt với tiếng Việt và hỗ trợ ngữ cảnh dài.

## 3.6. Thiết kế prompt engineering

### 3.6.1. Kỹ thuật viết lại câu hỏi và chuẩn hóa ngữ nghĩa

Prompt viết lại truy vấn cung cấp cho LLM 3 lượt hội thoại gần nhất cùng câu hỏi hiện tại, yêu cầu thay thế đại từ, bổ sung chủ thể bị tỉnh lược và giữ nguyên ý định gốc, xuất ra duy nhất câu hỏi độc lập. Temperature 0,1 bảo đảm kết quả ổn định, tránh việc viết lại làm "trôi" ngữ nghĩa.

### 3.6.2. Kỹ thuật mở rộng đa truy vấn (RAG Fusion)

Với câu hỏi phức tạp, prompt yêu cầu LLM sinh các cách diễn đạt khác nhau của cùng một nhu cầu thông tin (đồng nghĩa, cụ thể hóa, khái quát hóa). Mỗi biến thể được truy xuất độc lập, kết quả hợp nhất giúp tăng độ phủ khi cách dùng từ của người hỏi khác xa văn phong văn bản hành chính.

### 3.6.3. Kỹ thuật phân loại ý định và lọc metadata

Tập từ khóa đặc trưng cho từng nhóm chủ đề (đào tạo, tuyển sinh, chứng chỉ) được đối sánh với truy vấn để xác định bộ lọc `system_type` áp lên Qdrant, giúp loại bỏ sớm các tài liệu không liên quan và cải thiện độ chính xác truy xuất.

### 3.6.4. Kỹ thuật tổng hợp câu trả lời theo suy luận chuỗi (CoT)

System prompt tổng hợp (47 dòng, tiếng Việt) hướng dẫn mô hình theo trình tự: đọc hiểu câu hỏi → rà soát từng đoạn ngữ cảnh → đối chiếu và ưu tiên nguồn (quy định của UIT trước, văn bản Bộ GD&ĐT sau, FAQ cuối) → soạn câu trả lời có cấu trúc. Yêu cầu trình bày bảng khi người dùng hỏi so sánh giúp câu trả lời dễ đọc.

### 3.6.5. Kỹ thuật chống ảo giác và trích dẫn nguồn

Prompt quy định nghiêm ngặt: chỉ sử dụng thông tin trong ngữ cảnh được cung cấp; thông tin không có trong ngữ cảnh phải trả lời "không tìm thấy trong tài liệu" thay vì suy diễn; mỗi luận điểm phải kèm trích dẫn [Nguồn: ...] và URL nếu có. Đây là tuyến phòng thủ chính chống hiện tượng ảo giác.

### 3.6.6. Kỹ thuật từ chối truy vấn ngoài phạm vi

Kết hợp nhận diện chitchat ở tầng tiền xử lý và chỉ dẫn trong system prompt, hệ thống từ chối lịch sự các câu hỏi không liên quan đến đào tạo từ xa (ví dụ câu hỏi đời sống, chính trị), đồng thời gợi ý người dùng đặt câu hỏi trong phạm vi hỗ trợ. Năng lực này được kiểm chứng định lượng qua chỉ số Negative Rejection ở Chương 4.
