# CHƯƠNG 2. TỔNG QUAN NGHIÊN CỨU

## 2.1. Tổng quan bài toán hỏi đáp tri thức học vụ

Hỏi đáp tự động (Question Answering — QA) là bài toán kinh điển của xử lý ngôn ngữ tự nhiên, trong đó hệ thống nhận câu hỏi ngôn ngữ tự nhiên và trả về câu trả lời thay vì danh sách tài liệu như tìm kiếm truyền thống. Trong môi trường giáo dục đại học, bài toán này xuất hiện dưới dạng hỏi đáp tri thức học vụ: người học hỏi về quy chế, tuyển sinh, học phí, chương trình đào tạo — những thông tin được quy định trong các văn bản hành chính có hiệu lực pháp lý, đòi hỏi câu trả lời phải chính xác tuyệt đối và có căn cứ.

Các thế hệ chatbot tư vấn học vụ trước đây chủ yếu dựa trên luật (rule-based) hoặc đối sánh mẫu câu hỏi — câu trả lời soạn sẵn (FAQ matching). Cách tiếp cận này dễ kiểm soát nhưng cứng nhắc: chỉ trả lời được các câu hỏi đã lường trước, không hiểu được cách diễn đạt đa dạng của người dùng và tốn nhiều công bảo trì khi quy định thay đổi.

Sự xuất hiện của các mô hình ngôn ngữ lớn như GPT, Gemini, LLaMA đã thay đổi căn bản chất lượng hội thoại tự nhiên. Tuy nhiên, LLM thuần túy không phù hợp trực tiếp cho miền tri thức học vụ vì: (1) tri thức của mô hình bị "đóng băng" tại thời điểm huấn luyện, không chứa quy định nội bộ của từng trường; (2) hiện tượng ảo giác khiến mô hình có thể tự tin đưa ra thông tin sai về học phí, thời hạn, điều kiện — những sai sót gây hậu quả thực tế cho người học. Kiến trúc RAG (Retrieval-Augmented Generation), do Lewis và cộng sự đề xuất năm 2020, giải quyết vấn đề này bằng cách ghép một bộ truy xuất (retriever) tìm các đoạn văn bản liên quan từ kho tri thức bên ngoài vào trước bộ tạo sinh (generator), buộc LLM trả lời dựa trên ngữ cảnh được cung cấp. RAG hiện là cách tiếp cận chủ đạo cho các hệ hỏi đáp miền đóng cần độ tin cậy cao.

## 2.2. Các nghiên cứu liên quan

### 2.2.1. Các nghiên cứu trên thế giới

Lewis và cộng sự (2020) đề xuất kiến trúc RAG nguyên bản, kết hợp truy xuất dense (DPR) với mô hình seq2seq, cho thấy hiệu quả vượt trội trên các tác vụ hỏi đáp tri thức mở. Từ đó, nhiều cải tiến cho từng khâu của pipeline đã được nghiên cứu: Karpukhin và cộng sự (2020) với Dense Passage Retrieval đặt nền móng cho truy xuất ngữ nghĩa bằng vector; Gao và cộng sự (2023) khảo sát toàn diện các kỹ thuật RAG nâng cao như viết lại truy vấn (query rewriting), mở rộng đa truy vấn và rerank hai giai đoạn.

Về biểu diễn ngữ nghĩa đa ngôn ngữ, Chen và cộng sự (2024) công bố BGE-M3 — mô hình nhúng hỗ trợ hơn 100 ngôn ngữ, đồng thời hỗ trợ truy xuất dense, sparse và multi-vector, đạt kết quả cao trên các benchmark truy xuất đa ngôn ngữ trong đó có tiếng Việt. Cùng nhóm tác giả, BGE-reranker-v2-m3 là mô hình cross-encoder dùng cho khâu sàng lọc tinh, cải thiện đáng kể độ chính xác xếp hạng so với chỉ dùng điểm tương đồng vector.

Về kỹ thuật truy vấn, RAG Fusion (Rackauckas, 2024) sinh nhiều biến thể truy vấn rồi hợp nhất kết quả truy xuất, giúp tăng độ phủ với các câu hỏi phức tạp hoặc mơ hồ. Về đánh giá, RAGAS (Es và cộng sự, 2023) đề xuất bộ chỉ số đánh giá tự động hệ RAG bằng LLM-as-a-judge (Faithfulness, Answer Relevancy, Context Precision, Context Recall) không cần nhãn vàng đầy đủ; Chen và cộng sự (2024) với benchmark RGB đưa ra các khía cạnh đánh giá độ bền vững như Noise Robustness và Negative Rejection; khung khảo sát Auepora (Yu và cộng sự, 2024) hệ thống hóa các tiêu chí đánh giá RAG theo cặp mục tiêu truy xuất — tạo sinh.

Trong lĩnh vực giáo dục, nhiều trường đại học trên thế giới đã thử nghiệm chatbot tư vấn sinh viên dựa trên RAG, cho thấy khả năng giảm tải bộ phận hỗ trợ, song các nghiên cứu chủ yếu thực hiện trên tiếng Anh với nguồn tài liệu sạch, ít đề cập đến thách thức của tài liệu scan và ngôn ngữ ít tài nguyên.

### 2.2.2. Các nghiên cứu trong nước

Tại Việt Nam, các nghiên cứu chatbot tư vấn tuyển sinh — học vụ đã xuất hiện ở nhiều trường đại học, giai đoạn đầu chủ yếu dựa trên các nền tảng đối thoại như Rasa, Dialogflow với ý định (intent) và thực thể được định nghĩa thủ công. Cách tiếp cận này đòi hỏi xây dựng và duy trì tập huấn luyện intent lớn, khó mở rộng khi phạm vi tri thức tăng.

Gần đây, cùng với sự phổ biến của LLM, một số nghiên cứu trong nước đã chuyển sang hướng RAG cho tiếng Việt: ứng dụng hỏi đáp văn bản pháp luật, hỏi đáp tài liệu doanh nghiệp, trợ lý tra cứu quy chế đào tạo. Các nghiên cứu này cho thấy hai thách thức đặc thù: (1) chất lượng mô hình nhúng cho tiếng Việt — các mô hình đa ngôn ngữ như multilingual-E5, BGE-M3 thường được lựa chọn do chưa có nhiều mô hình nhúng tiếng Việt chuyên biệt đủ mạnh; (2) chất lượng dữ liệu nguồn — văn bản hành chính Việt Nam tồn tại nhiều ở dạng PDF scan, đòi hỏi pipeline OCR và làm sạch công phu.

Riêng với hệ đào tạo từ xa của UIT, hiện kênh giải đáp chủ yếu vẫn là email, điện thoại và trang câu hỏi thường gặp tĩnh; chưa có hệ thống hỏi đáp tự động dựa trên chính các văn bản quy định hiện hành của trường.

## 2.3. Khoảng trống nghiên cứu và ý tưởng đề xuất

Từ khảo sát trên, có thể rút ra các khoảng trống sau:

- Các chatbot học vụ trong nước phần lớn vẫn theo hướng intent-based, hoặc nếu dùng RAG thì ít công bố quy trình đánh giá định lượng bài bản; thiếu các nghiên cứu đo lường đồng thời chất lượng truy xuất, chất lượng tạo sinh và độ bền vững trên dữ liệu quy định tiếng Việt.
- Thách thức xử lý tài liệu nguồn không thuần nhất (PDF scan chất lượng thấp, DOCX, dữ liệu web) trong pipeline RAG tiếng Việt chưa được giải quyết trọn vẹn trong các nghiên cứu hiện có.
- Chưa có hệ thống hỏi đáp tự động chuyên biệt cho hệ đào tạo từ xa UIT.

Trên cơ sở đó, đề tài đề xuất xây dựng hệ thống chatbot RAG hoàn chỉnh cho hệ đào tạo từ xa UIT với các điểm nhấn: (1) pipeline xử lý dữ liệu đa định dạng có OCR và làm sạch nhiễu; (2) truy xuất lai kết hợp vector — từ khóa cùng các kỹ thuật viết lại truy vấn, mở rộng đa truy vấn và rerank hai giai đoạn được tinh chỉnh cho tiếng Việt; (3) bộ khung đánh giá đa chiều kết hợp chỉ số xếp hạng truy xuất (MAP/MRR/Hit@K), chỉ số tạo sinh LLM-as-a-judge và đánh giá độ bền vững theo định hướng Auepora/RGB; (4) chương trình minh họa hoàn chỉnh sẵn sàng thử nghiệm thực tế.
