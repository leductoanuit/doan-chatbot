"""Gemini LLM client — wraps Google Generative AI SDK for RAG generation."""

import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

_DEFAULT_SYSTEM_PROMPT = (
    "Bạn là tư vấn viên thông tin đào tạo từ xa của Trường Đại học Công nghệ Thông tin "
    "- ĐHQG TP.HCM (UIT). Hãy trả lời câu hỏi của sinh viên dựa trên thông tin được cung cấp. "
    "Nếu không có thông tin liên quan, hãy nói rằng bạn không có đủ thông tin và đề nghị "
    "sinh viên liên hệ phòng đào tạo hoặc truy cập daa.uit.edu.vn."
)


class LLMClient:
    """Wrapper around Google Gemini API for RAG generation."""

    def __init__(
        self,
        model_name: str = None,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    ):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment")
        genai.configure(api_key=api_key)
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.system_prompt = system_prompt
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_prompt,
        )

    def generate(
        self,
        query: str,
        context: str = "",
        history: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """Generate a response given a query and optional RAG context.

        Args:
            query: User question.
            context: Retrieved document context to inject before the question.
            history: Previous conversation turns (list of role/content dicts).
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.

        Returns:
            Generated answer string, or an error message on failure.
        """
        contents = []

        # Include last 3 conversation turns for context
        if history:
            for msg in history[-6:]:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({"role": role, "parts": [msg["content"]]})

        # Inject retrieved context before the user query
        if context:
            user_content = f"Thông tin tham khảo:\n{context}\n\nCâu hỏi: {query}"
        else:
            user_content = query

        contents.append({"role": "user", "parts": [user_content]})

        try:
            response = self.model.generate_content(
                contents,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return response.text
        except Exception as exc:
            print(f"[llm-client] Generation error: {exc}")
            return "Lỗi hệ thống: Không thể tạo câu trả lời. Vui lòng thử lại sau."

    def health_check(self) -> bool:
        """Return True if Gemini API is reachable."""
        try:
            self.model.generate_content(
                "ping",
                generation_config=genai.GenerationConfig(max_output_tokens=5),
            )
            return True
        except Exception:
            return False
