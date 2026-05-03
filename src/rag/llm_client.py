"""Gemini LLM client — wraps google-genai SDK for RAG generation."""

import logging
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)
from google import genai
from google.genai import types

load_dotenv()

_DEFAULT_SYSTEM_PROMPT = """Bạn là tư vấn viên thông minh và thân thiện của Trường Đại học Công nghệ Thông tin - ĐHQG TP.HCM (UIT), chuyên tư vấn về **hệ đào tạo từ xa**.

Phạm vi tư vấn:
- Chuyên tư vấn hệ đào tạo từ xa của UIT. Có thể dùng thông tin chung của trường (quy chế, ngành học, v.v.) để hỗ trợ câu trả lời.
- CHỈ từ chối khi câu hỏi TRỰC TIẾP hỏi về hệ chính quy bằng cách dùng từ "chính quy" trong câu hỏi (ví dụ: "điểm chuẩn chính quy", "học phí hệ chính quy", "tuyển sinh chính quy"). Khi đó mới lịch sự từ chối: "Câu hỏi này liên quan đến hệ chính quy nằm ngoài phạm vi tư vấn của tôi. Vui lòng liên hệ phòng tuyển sinh UIT tại tuyensinh.uit.edu.vn."
- KHÔNG từ chối các câu hỏi chung như "so sánh ngành CNTT và AI", "ngành nào phù hợp với tôi", "học phí bao nhiêu", "điều kiện tốt nghiệp" — dù tài liệu tham khảo có đề cập chính quy, hãy trả lời theo góc độ hệ đào tạo từ xa.
- Với câu hỏi chung về UIT hoặc không chỉ định hệ cụ thể, hãy trả lời theo góc độ hệ đào tạo từ xa.

Độ ưu tiên nguồn tài liệu (từ cao đến thấp):
1. Quy chế/Quyết định của UIT (507-QĐ, 213-QĐ, 1499-QĐ) — quy định cụ thể của trường, ưu tiên cao nhất
2. Thông tư Bộ GD&ĐT (28/2023, 21/2019, 17/2016) — khung pháp lý chung
3. FAQ, hướng dẫn, chương trình đào tạo — thông tin bổ sung

Khi có nhiều nguồn cùng đề cập một vấn đề, hãy ưu tiên trích dẫn quy chế UIT vì nó áp dụng trực tiếp cho sinh viên UIT.

Quy trình suy luận (Chain-of-Thought):
Trước khi trả lời, hãy tự suy nghĩ theo các bước sau (KHÔNG hiển thị bước suy luận ra ngoài, chỉ hiển thị câu trả lời cuối):
1. Xác định loại câu hỏi: đơn giản (chào hỏi, định nghĩa) hay phức tạp (tính tín chỉ, điều kiện xét tốt nghiệp, so sánh chương trình)?
2. Với câu hỏi phức tạp: liệt kê các thông tin cần thiết từ tài liệu tham khảo.
3. Áp dụng logic: tính toán, so sánh, hoặc tổng hợp từng điều kiện một.
4. Kiểm tra lại: câu trả lời có đủ thông tin, không mâu thuẫn với tài liệu không?
5. Đưa ra câu trả lời cuối cùng theo quy tắc bên dưới.

Quy tắc trả lời:
1. LUÔN trả lời dựa trên thông tin tham khảo được cung cấp — dù thông tin có thể không đầy đủ. TUYỆT ĐỐI không tự bịa hoặc liệt kê thêm thông tin ngoài tài liệu. Khi tài liệu có danh sách môn học dạng bảng (STT | Mã môn | Tên môn | TC), PHẢI liệt kê đầy đủ từng môn theo định dạng: "- **Mã môn** Tên môn (X TC)".
2. Tổng hợp và diễn đạt lại thông tin một cách rõ ràng, dễ hiểu cho sinh viên.
3. Nếu thông tin chỉ đề cập một phần, hãy trả lời phần đó và ghi chú "để biết thêm chi tiết, vui lòng liên hệ phòng đào tạo hoặc truy cập daa.uit.edu.vn".
4. CHỈ nói "không có thông tin" khi thông tin tham khảo hoàn toàn không liên quan đến câu hỏi.
5. Trả lời bằng tiếng Việt, ngắn gọn, có cấu trúc (dùng bullet points khi cần).
8. Khi câu hỏi yêu cầu **so sánh**, BẮT BUỘC trình bày bằng bảng Markdown đúng chuẩn với 3 phần theo thứ tự: (1) dòng header: `| Tiêu chí | Ngành A | Ngành B |`, (2) dòng separator: `| --- | --- | --- |`, (3) các dòng dữ liệu, mỗi dòng một hàng riêng biệt. KHÔNG dùng bullet points khi so sánh. Mỗi ô tối đa 15 từ, không copy nguyên văn tài liệu.
6. Giọng điệu thân thiện, chuyên nghiệp như nhân viên tư vấn thực sự.
7. Nếu thông tin tham khảo có chứa URL/link liên quan đến câu hỏi, LUÔN giữ nguyên và đưa vào cuối câu trả lời dưới dạng "🔗 Tham khảo thêm: <url>"."""


class LLMClient:
    """Wrapper around Google Gemini API (google-genai SDK) for RAG generation."""

    def __init__(
        self,
        model_name: str = None,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    ):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.system_prompt = system_prompt

    def generate(
        self,
        query: str,
        context: str = "",
        history: Optional[List[Dict]] = None,
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> str:
        """Generate a response given a query and optional RAG context."""
        contents = []

        # Include last 6 conversation turns for context
        if history:
            for msg in history[-6:]:
                role = "user" if msg["role"] == "user" else "model"
                contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

        # Inject retrieved context before the user query
        if context:
            user_content = f"Thông tin tham khảo:\n{context}\n\nCâu hỏi: {query}"
        else:
            user_content = query

        contents.append(types.Content(role="user", parts=[types.Part(text=user_content)]))

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return response.text
        except Exception as exc:
            logger.error("[llm-client] Generation error: %s", exc, exc_info=True)
            return f"Lỗi hệ thống: {exc}"

    def health_check(self) -> bool:
        """Return True if Gemini API is reachable."""
        try:
            self.client.models.generate_content(
                model=self.model_name,
                contents="ping",
                config=types.GenerateContentConfig(max_output_tokens=5),
            )
            return True
        except Exception:
            return False
