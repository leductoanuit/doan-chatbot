# CHƯƠNG 5. CÀI ĐẶT CHƯƠNG TRÌNH MINH HỌA

## 5.1. Các công nghệ sử dụng

Chương trình minh họa được xây dựng hoàn toàn bằng Python 3.12 với các công nghệ chính:

| Tầng | Công nghệ | Vai trò |
|---|---|---|
| Giao diện | Streamlit | Giao diện trò chuyện web, quản lý phiên |
| API | FastAPI 0.115 | REST API, kiểm tra dữ liệu vào bằng Pydantic |
| Lõi RAG | google-genai, sentence-transformers | Gọi Gemini; nhúng BGE-M3, rerank |
| CSDL vector | Qdrant | Lưu trữ và tìm kiếm vector |
| CSDL quan hệ | PostgreSQL 16 | Metadata tài liệu, lịch sử hội thoại |
| Xử lý tài liệu | PyMuPDF, Tesseract, OpenCV, python-docx | Trích xuất, OCR, tiền xử lý ảnh |
| Hạ tầng | Docker Compose | Khởi chạy Qdrant, PostgreSQL, MongoDB |

## 5.2. Thiết kế API

REST API được thiết kế gọn với bốn nhóm endpoint:

| Endpoint | Phương thức | Chức năng |
|---|---|---|
| `/chat` | POST | Hỏi đáp chính: nhận `{message, history}`, trả `{answer, sources[], context_used}` |
| `/search` | GET | Truy xuất trực tiếp không qua LLM: tham số `query`, `top_k` (1–20, mặc định 5), trả danh sách đoạn văn bản kèm nguồn và điểm liên quan |
| `/export` | POST | Xuất nội dung hội thoại ra tệp Word (hai định dạng: biên bản hội thoại hoặc báo cáo kỹ thuật) |
| `/health`, `/` | GET | Kiểm tra trạng thái dịch vụ và thông tin API |

Endpoint `/chat` đóng gói toàn bộ pipeline RAG bốn giai đoạn của Chương 3; trường `sources` trong phản hồi liệt kê các đoạn tài liệu được sử dụng kèm tên tệp nguồn và điểm liên quan, bảo đảm tính minh bạch và khả năng kiểm chứng của câu trả lời. API bật CORS phục vụ tích hợp với các giao diện web khác trong tương lai.

[Hình 5.1: Sơ đồ thiết kế API và luồng dữ liệu — sẽ bổ sung]

## 5.3. Mô tả giao diện của hệ thống

Giao diện người dùng xây dựng bằng Streamlit theo bố cục hai vùng: thanh bên trái quản lý các phiên hội thoại (tạo mới, tải lại, xóa phiên — lưu bền vững trong PostgreSQL qua hai bảng `chat_sessions` và `chat_messages`), vùng chính là khung trò chuyện hiển thị lịch sử hỏi đáp theo từng lượt với nhãn người hỏi/chatbot. Dưới mỗi câu trả lời, hệ thống hiển thị mục "Nguồn tham khảo" liệt kê các đoạn tài liệu được trích dẫn. Giao diện hoàn toàn bằng tiếng Việt, kèm nút xuất hội thoại ra tệp Word.

[Hình 5.2: Giao diện trò chuyện của hệ thống — sẽ bổ sung]

## 5.4. Các tính năng của hệ thống

### 5.4.1. Giới thiệu giao diện và tương tác với chatbot

Người dùng nhập câu hỏi tự nhiên vào khung trò chuyện; hệ thống nhận diện câu chào hỏi xã giao và phản hồi thân thiện không cần truy xuất, còn câu hỏi nghiệp vụ được chuyển qua pipeline RAG đầy đủ. Mỗi câu trả lời kèm danh sách nguồn giúp người dùng tự kiểm chứng.

[Hình 5.3: Minh họa tương tác hỏi đáp cơ bản — sẽ bổ sung]

### 5.4.2. Truy xuất thông tin theo thực thể và duy trì hội thoại nhiều lượt

Hệ thống duy trì ngữ cảnh hội thoại (6 lượt gần nhất): khi người dùng hỏi nối tiếp bằng đại từ ("vậy học phí của chương trình đó?"), cơ chế viết lại truy vấn tự động khôi phục thực thể được nhắc đến ở các lượt trước, cho phép hội thoại tự nhiên như trao đổi với tư vấn viên.

[Hình 5.4: Minh họa hội thoại nhiều lượt với viết lại truy vấn — sẽ bổ sung]

### 5.4.3. Tổng hợp tri thức từ nhiều nguồn và sinh câu trả lời hoàn chỉnh

Với câu hỏi đòi hỏi thông tin nằm ở nhiều văn bản (ví dụ điều kiện tốt nghiệp vừa theo quy chế trường vừa theo thông tư của Bộ), cơ chế mở rộng đa truy vấn và truy xuất lai gom đủ các đoạn liên quan, prompt tổng hợp ưu tiên nguồn theo thứ bậc pháp lý và hợp nhất thành câu trả lời mạch lạc có trích dẫn từng nguồn.

[Hình 5.5: Minh họa câu trả lời tổng hợp từ nhiều nguồn — sẽ bổ sung]

### 5.4.4. So sánh và phân tích giữa các đối tượng

Khi phát hiện câu hỏi so sánh (ví dụ so sánh chương trình văn bằng 1 và liên thông), hệ thống tự động tăng gấp đôi lượng ngữ cảnh truy xuất và yêu cầu mô hình trình bày kết quả dạng bảng so sánh theo tiêu chí, giúp người học nắm thông tin trực quan.

[Hình 5.6: Minh họa câu trả lời dạng bảng so sánh — sẽ bổ sung]

### 5.4.5. Kiểm soát nội dung và xử lý truy vấn ngoài phạm vi tri thức

Câu hỏi nằm ngoài miền đào tạo từ xa được từ chối lịch sự kèm gợi ý phạm vi hỗ trợ, thay vì trả lời bịa đặt — năng lực đã được kiểm chứng định lượng với Negative Rejection đạt 1,0 ở Chương 4. Cơ chế này đặc biệt quan trọng với chatbot đại diện cho đơn vị giáo dục, nơi một câu trả lời sai có thể gây hiểu nhầm về quy định.

[Hình 5.7: Minh họa từ chối câu hỏi ngoài phạm vi — sẽ bổ sung]

## 5.5. Nhận xét

Chương trình minh họa vận hành ổn định trên môi trường cục bộ, thể hiện đầy đủ các năng lực thiết kế: hỏi đáp có trích dẫn nguồn, hội thoại nhiều lượt, tổng hợp đa nguồn, so sánh dạng bảng và kiểm soát phạm vi. Kiến trúc tách lớp API — giao diện cho phép thay thế giao diện Streamlit bằng ứng dụng web/mobile chính thức của trường mà không thay đổi lõi RAG. Hạn chế chính ở trải nghiệm là độ trễ phản hồi (2,2–6,7 phút/câu) do reranker chạy trên CPU và chuỗi gọi LLM tuần tự, sẽ được thảo luận trong hướng phát triển.
