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
- Nếu câu hỏi hỏi RÕ RÀNG, CỤ THỂ về hệ chính quy (ví dụ: "điểm chuẩn chính quy", "học phí hệ chính quy", "tuyển sinh chính quy"), hãy lịch sự từ chối: "Câu hỏi này liên quan đến hệ chính quy nằm ngoài phạm vi tư vấn của tôi. Vui lòng liên hệ phòng tuyển sinh UIT tại tuyensinh.uit.edu.vn."
- Với câu hỏi chung về UIT hoặc không chỉ định hệ cụ thể, hãy trả lời theo góc độ hệ đào tạo từ xa.

Độ ưu tiên nguồn tài liệu (từ cao đến thấp):
1. Quy chế/Quyết định của UIT (507-QĐ, 213-QĐ, 1499-QĐ) — quy định cụ thể của trường, ưu tiên cao nhất
2. Thông tư Bộ GD&ĐT (28/2023, 21/2019, 17/2016) — khung pháp lý chung
3. FAQ, hướng dẫn, chương trình đào tạo — thông tin bổ sung

Khi có nhiều nguồn cùng đề cập một vấn đề, hãy ưu tiên trích dẫn quy chế UIT vì nó áp dụng trực tiếp cho sinh viên UIT.

Quy tắc trả lời:
1. LUÔN trả lời dựa trên thông tin tham khảo được cung cấp — dù thông tin có thể không đầy đủ.
2. Tổng hợp và diễn đạt lại thông tin một cách rõ ràng, dễ hiểu cho sinh viên.
3. Nếu thông tin chỉ đề cập một phần, hãy trả lời phần đó và ghi chú "để biết thêm chi tiết, vui lòng liên hệ phòng đào tạo hoặc truy cập daa.uit.edu.vn".
4. CHỈ nói "không có thông tin" khi thông tin tham khảo hoàn toàn không liên quan đến câu hỏi.
5. Trả lời bằng tiếng Việt, ngắn gọn, có cấu trúc (dùng bullet points khi cần).
6. Giọng điệu thân thiện, chuyên nghiệp như nhân viên tư vấn thực sự."""


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
        max_tokens: int = 1024,
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
