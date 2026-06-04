# CHƯƠNG 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 6.1. Kết luận

Đề tài đã hoàn thành mục tiêu xây dựng hệ thống chatbot hỏi đáp tự động cho hệ đào tạo từ xa của Trường Đại học Công nghệ Thông tin theo kiến trúc RAG, với các kết quả chính:

- **Về dữ liệu:** xây dựng được pipeline thu thập — trích xuất — làm sạch hoàn chỉnh cho 19 nguồn tài liệu không thuần nhất (PDF native, PDF scan, DOCX, dữ liệu web), trong đó giải quyết được thách thức OCR tài liệu scan tiếng Việt chất lượng thấp bằng tiền xử lý ảnh và lọc nhiễu; kết quả là kho tri thức 448 bản ghi, phân đoạn thành 513 chunks được đánh chỉ mục vector trong Qdrant kèm metadata đầy đủ.
- **Về phương pháp:** thiết kế và cài đặt hệ thống RAG bốn giai đoạn với nhiều kỹ thuật nâng cao: viết lại truy vấn theo ngữ cảnh hội thoại, phân loại ý định kèm lọc metadata, truy xuất lai vector — từ khóa, mở rộng đa truy vấn RAG Fusion, sàng lọc bằng cross-encoder và tổng hợp bằng prompt suy luận chuỗi có cơ chế chống ảo giác.
- **Về đánh giá:** xây dựng được tập kiểm thử 100 câu hỏi gắn nhãn và bộ khung đánh giá tự động đa chiều; kết quả thực nghiệm trên 100 câu (2 batch) cho thấy hệ thống đạt chất lượng truy xuất rất tốt (MAP@10 0,90–0,93; MRR 0,99; Hit@10 1,0; Context Recall 0,895), khả năng kiểm soát phạm vi tuyệt đối (Negative Rejection 1,0) và độ bền vững với nhiễu tốt (Noise Robustness 0,8).
- **Về sản phẩm:** hoàn thành chương trình minh họa gồm REST API và giao diện trò chuyện tiếng Việt với đầy đủ tính năng hội thoại nhiều lượt, trích dẫn nguồn, so sánh dạng bảng và xuất hội thoại ra Word.

## 6.2. Hạn chế

Bên cạnh kết quả đạt được, hệ thống còn các hạn chế:

- **Độ đúng của câu trả lời chưa cao:** Correctness chỉ đạt 0,58–0,59 trên cả hai batch và Faithfulness 0,73 cho thấy hiện tượng ảo giác vẫn còn — một số câu trả lời không bám sát tài liệu được truy xuất dù truy xuất đã tìm đúng nguồn (MRR 0,99). Đặc biệt danh mục học phí ở batch 1 đạt điểm tạo sinh rất thấp (Faithfulness 0,40; Answer Relevancy 0,20), cho thấy dữ liệu học phí trong kho tri thức chưa đủ chi tiết.
- **Chiến lược phân đoạn chưa tối ưu:** chunking theo số từ (512 từ ≈ 700–900 token) vượt ngưỡng 512 token tối ưu của BGE-M3, khiến phần đuôi chunk bị cắt khi nhúng, ảnh hưởng chất lượng biểu diễn; đây là một nguyên nhân khiến Context Precision giảm ở nhóm câu hỏi khó (batch 2: 0,556).
- **Độ trễ phản hồi lớn:** trung bình 2,2–6,7 phút/câu, nguyên nhân chính là mô hình reranker chạy trên CPU và chuỗi gọi LLM tuần tự (viết lại, sinh đa truy vấn, tổng hợp), chưa đáp ứng trải nghiệm thời gian thực.
- **Chất lượng OCR:** một số trang scan mờ, nghiêng vẫn còn nhiễu sau làm sạch, ảnh hưởng cục bộ đến truy xuất.
- **Phạm vi thực nghiệm:** quy mô tập kiểm thử độ bền vững (11 câu) còn nhỏ; chưa có đánh giá bởi người dùng thực để đối chiếu với kết quả chấm tự động bằng LLM.

## 6.3. Hướng phát triển

Từ các hạn chế trên, đề tài đề xuất các hướng phát triển:

- **Cải thiện độ đúng:** tinh chỉnh prompt tổng hợp để tăng Correctness (bổ sung few-shot examples, tự kiểm chứng câu trả lời, trích dẫn bắt buộc theo câu); chuyển sang chunking theo token với ranh giới ngữ nghĩa; bổ sung dữ liệu học phí chi tiết hơn để khắc phục danh mục yếu nhất.
- **Giảm độ trễ:** triển khai reranker trên GPU (mục tiêu dưới 10 giây/câu), áp dụng streaming phản hồi, cache kết quả nhúng và truy xuất cho câu hỏi phổ biến, hợp nhất các lần gọi LLM tiền xử lý.
- **Mở rộng dữ liệu và phạm vi:** bổ sung tự động hóa cập nhật khi nhà trường ban hành văn bản mới; mở rộng sang các hệ đào tạo khác của trường; bổ sung kênh tiếp cận (tích hợp website trường, ứng dụng di động, nền tảng nhắn tin).
- **Hoàn thiện đánh giá:** mở rộng tập kiểm thử độ bền vững, bổ sung đánh giá bởi người dùng thực (human evaluation) và thử nghiệm A/B các cấu hình truy xuất.
- **Hướng nghiên cứu xa hơn:** thử nghiệm các kiến trúc RAG nâng cao như Agentic RAG (tự quyết định khi nào cần truy xuất thêm), GraphRAG (khai thác quan hệ giữa các điều khoản văn bản) cho miền văn bản quy định.
