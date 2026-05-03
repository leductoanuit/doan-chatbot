"""LLM-as-judge retrieval metrics for RAG evaluation.

Metrics:
  - score_context_precision:  fraction of retrieved contexts that are relevant (0-1)
  - score_context_recall:     contexts contain enough info to produce ground truth (0-1)
"""

from typing import List

from google import genai

from ragas_base import _score_prompt


def score_context_precision(client: genai.Client, question: str, contexts: List[str], ground_truth: str) -> float:
    """Fraction of retrieved contexts that are relevant and useful for answering."""
    ctx_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    prompt = f"""Bạn đánh giá độ chính xác của các context được truy xuất trong hệ thống RAG.

CÂU HỎI: {question}
ĐÁP ÁN CHUẨN: {ground_truth}

CÁC CONTEXT ĐƯỢC TRUY XUẤT:
{ctx_text}

Đánh giá CONTEXT PRECISION: tỷ lệ context liên quan và hữu ích để trả lời câu hỏi.
- 1.0: tất cả contexts đều liên quan
- 0.5: khoảng một nửa contexts liên quan
- 0.0: không có context nào liên quan

Chỉ trả lời một số thập phân từ 0.0 đến 1.0."""
    return _score_prompt(client, prompt)


def score_context_recall(client: genai.Client, contexts: List[str], ground_truth: str) -> float:
    """Whether contexts contain enough information to produce the ground truth answer."""
    ctx_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    prompt = f"""Bạn đánh giá mức độ đầy đủ của contexts để tạo ra đáp án chuẩn.

ĐÁP ÁN CHUẨN: {ground_truth}

CÁC CONTEXT ĐƯỢC TRUY XUẤT:
{ctx_text}

Đánh giá CONTEXT RECALL: contexts có chứa đủ thông tin để suy ra đáp án chuẩn không?
- 1.0: contexts chứa đầy đủ thông tin cần thiết
- 0.5: contexts chứa một phần thông tin cần thiết
- 0.0: contexts không chứa thông tin cần thiết

Chỉ trả lời một số thập phân từ 0.0 đến 1.0."""
    return _score_prompt(client, prompt)
