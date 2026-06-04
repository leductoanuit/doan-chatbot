# CHƯƠNG 1. GIỚI THIỆU ĐỀ TÀI

## 1.1. Lý do chọn đề tài

Đào tạo từ xa đang trở thành một trong những hướng phát triển quan trọng của giáo dục đại học Việt Nam, đặc biệt sau giai đoạn chuyển đổi số được thúc đẩy mạnh mẽ. Tại Trường Đại học Công nghệ Thông tin (UIT) — ĐHQG-HCM, hệ đào tạo từ xa được triển khai với nhiều chương trình như Cử nhân Công nghệ Thông tin (văn bằng 1, liên thông đại học) và Cử nhân Trí tuệ Nhân tạo, thu hút lượng lớn người học là người đi làm, ở nhiều độ tuổi và vùng miền khác nhau.

Đặc thù của người học từ xa là ít có điều kiện tiếp xúc trực tiếp với bộ phận quản lý đào tạo, do đó nhu cầu được giải đáp các thắc mắc về tuyển sinh, học phí, quy chế học vụ, chương trình đào tạo... là rất thường xuyên. Trong khi đó, các thông tin này nằm rải rác trong nhiều loại văn bản: quy chế đào tạo từ xa của Bộ Giáo dục và Đào tạo (Thông tư 28/2023/TT-BGDĐT), các quyết định ban hành quy chế của nhà trường (507/QĐ-ĐHCNTT, 1499/QĐ-ĐHCNTT...), chương trình đào tạo, biểu mẫu quy trình và danh sách câu hỏi thường gặp. Người học khó tự tra cứu chính xác, còn cán bộ tư vấn phải trả lời lặp lại những câu hỏi tương tự nhau, gây tốn thời gian và nguồn lực.

Sự phát triển của các mô hình ngôn ngữ lớn (Large Language Model — LLM) mở ra khả năng xây dựng chatbot hỏi đáp tự nhiên bằng tiếng Việt. Tuy nhiên, LLM thuần túy có hai hạn chế lớn khi áp dụng vào miền tri thức hẹp: (1) hiện tượng "ảo giác" (hallucination) — trả lời trôi chảy nhưng sai sự thật; (2) không nắm được tri thức nội bộ, cập nhật của một đơn vị cụ thể. Kiến trúc RAG (Retrieval-Augmented Generation) khắc phục hai hạn chế này bằng cách truy xuất các đoạn văn bản liên quan từ kho tài liệu chính thống rồi mới yêu cầu LLM tổng hợp câu trả lời dựa trên đó. Đây chính là động lực để đề tài lựa chọn xây dựng chatbot tư vấn hệ đào tạo từ xa UIT theo kiến trúc RAG.

## 1.2. Mục tiêu của đề tài

Đề tài hướng đến các mục tiêu cụ thể sau:

- Xây dựng bộ dữ liệu tri thức về hệ đào tạo từ xa UIT từ các văn bản quy định chính thống, bao gồm pipeline thu thập, trích xuất (kể cả tài liệu scan bằng OCR), làm sạch và chuẩn hóa dữ liệu.
- Thiết kế và cài đặt hệ thống RAG hoàn chỉnh cho tiếng Việt, gồm các giai đoạn tiền xử lý truy vấn, truy xuất lai (hybrid retrieval), sàng lọc bằng mô hình rerank và tổng hợp câu trả lời bằng LLM, kết hợp các kỹ thuật prompt engineering phù hợp.
- Xây dựng tập kiểm thử và bộ khung đánh giá định lượng chất lượng hệ thống trên cả ba phương diện: chất lượng truy xuất, chất lượng tạo sinh và độ bền vững (robustness).
- Cài đặt chương trình minh họa gồm REST API và giao diện web trò chuyện, hỗ trợ hội thoại nhiều lượt, hiển thị nguồn trích dẫn và xuất báo cáo hội thoại.

## 1.3. Phát biểu bài toán

Bài toán của đề tài được phát biểu như sau:

- **Đầu vào:** câu hỏi của người dùng bằng ngôn ngữ tự nhiên tiếng Việt liên quan đến hệ đào tạo từ xa UIT (có thể là câu hỏi độc lập hoặc câu hỏi phụ thuộc ngữ cảnh các lượt hội thoại trước), cùng lịch sử hội thoại của phiên làm việc.
- **Đầu ra:** câu trả lời tiếng Việt chính xác, bám sát nội dung các văn bản quy định chính thống của nhà trường và Bộ Giáo dục và Đào tạo, kèm danh sách nguồn tài liệu được trích dẫn; trường hợp câu hỏi nằm ngoài phạm vi tri thức, hệ thống phải từ chối trả lời một cách lịch sự thay vì bịa đặt thông tin.

Về bản chất, đây là bài toán hỏi đáp miền đóng (closed-domain question answering) trên kho văn bản quy định tiếng Việt, đòi hỏi giải quyết đồng thời các bài toán con: trích xuất văn bản từ tài liệu không thuần nhất (PDF native, PDF scan, DOCX, JSON), biểu diễn và truy xuất ngữ nghĩa tiếng Việt, xử lý hội thoại nhiều lượt, và kiểm soát chất lượng tạo sinh của LLM.

## 1.4. Đối tượng và phạm vi nghiên cứu

### 1.4.1. Đối tượng nghiên cứu

- Kiến trúc RAG và các kỹ thuật thành phần: phân đoạn văn bản (chunking), mô hình nhúng ngữ nghĩa đa ngôn ngữ, cơ sở dữ liệu vector, truy xuất lai, mô hình rerank, prompt engineering cho LLM.
- Các phương pháp đánh giá hệ thống RAG: chỉ số xếp hạng truy xuất (MAP, MRR, Hit@K), chỉ số chất lượng tạo sinh với LLM-as-a-judge (Faithfulness, Answer Relevancy, Context Precision, Context Recall, Correctness) và đánh giá độ bền vững (Noise Robustness, Negative Rejection).
- Kho văn bản quy định về đào tạo từ xa của UIT và Bộ Giáo dục và Đào tạo.

### 1.4.2. Phạm vi nghiên cứu

- **Phạm vi tri thức:** các văn bản về tuyển sinh, chương trình đào tạo, học vụ, học phí và quy chế của hệ đào tạo từ xa UIT tính đến thời điểm thực hiện đề tài (19 nguồn tài liệu). Hệ thống không trả lời các câu hỏi ngoài phạm vi này.
- **Phạm vi ngôn ngữ:** tiếng Việt là ngôn ngữ chính của cả truy vấn và câu trả lời.
- **Phạm vi công nghệ:** sử dụng LLM thương mại qua API (Google Gemini), mô hình nhúng và rerank mã nguồn mở (BGE-M3, BGE-reranker-v2-m3), cơ sở dữ liệu vector Qdrant; triển khai ở mức minh họa trên môi trường cục bộ với Docker Compose, chưa đặt mục tiêu vận hành sản phẩm thương mại quy mô lớn.

## 1.5. Ý nghĩa đề tài

**Ý nghĩa khoa học:** đề tài cung cấp một nghiên cứu thực nghiệm hoàn chỉnh về việc áp dụng kiến trúc RAG cho bài toán hỏi đáp văn bản quy định tiếng Việt — một miền dữ liệu có nhiều đặc thù (văn bản hành chính, tài liệu scan chất lượng thấp, thuật ngữ chuyên ngành giáo dục), đồng thời xây dựng quy trình đánh giá định lượng đa chiều có thể tái sử dụng cho các hệ thống RAG tiếng Việt khác.

**Ý nghĩa thực tiễn:** sản phẩm chatbot giúp người học hệ đào tạo từ xa UIT tự tra cứu thông tin 24/7 với câu trả lời có trích dẫn nguồn, giảm tải cho bộ phận tư vấn của nhà trường, và là nền tảng để mở rộng cho các hệ đào tạo khác của trường.

## 1.6. Cấu trúc của báo cáo

Báo cáo gồm 6 chương:

- **Chương 1 — Giới thiệu đề tài:** trình bày lý do chọn đề tài, mục tiêu, phát biểu bài toán, đối tượng, phạm vi và ý nghĩa của đề tài.
- **Chương 2 — Tổng quan nghiên cứu:** khảo sát bài toán hỏi đáp trong giáo dục, các nghiên cứu liên quan trong và ngoài nước, từ đó chỉ ra khoảng trống nghiên cứu và ý tưởng đề xuất.
- **Chương 3 — Phương pháp xây dựng và triển khai:** mô tả quy trình thực hiện, kiến trúc hệ thống, quá trình xây dựng bộ dữ liệu và cơ sở dữ liệu vector, thiết kế các giai đoạn của hệ thống RAG, lựa chọn cấu hình mô hình và các kỹ thuật prompt engineering.
- **Chương 4 — Thực nghiệm và đánh giá:** trình bày tập kiểm thử, các phương pháp đánh giá và phân tích kết quả thực nghiệm về chất lượng truy xuất, tạo sinh, độ bền vững và độ trễ phản hồi.
- **Chương 5 — Cài đặt chương trình minh họa:** giới thiệu công nghệ sử dụng, thiết kế API, giao diện và các tính năng của hệ thống.
- **Chương 6 — Kết luận và hướng phát triển:** tổng kết kết quả đạt được, hạn chế và đề xuất hướng phát triển.
