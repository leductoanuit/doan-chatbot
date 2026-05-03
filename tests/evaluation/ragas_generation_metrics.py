"""LLM-as-judge generation metrics for RAG evaluation.

Metrics:
  - score_faithfulness:        answer claims are grounded in contexts (0-1)
  - score_answer_relevancy:    answer addresses the question (0-1)
  - score_correctness:         answer matches ground truth semantically (0-1)
  - score_noise_robustness:    answer correct despite injected noisy contexts (0-1)
  - score_negative_rejection:  system properly refuses unanswerable questions (0-1)
"""

import re
from typing import List

from google import genai

from ragas_base import _score_prompt


_REFUSAL_PATTERN = re.compile(
    r"không (có |tìm thấy |đủ )?thông tin|không biết|không đủ dữ liệu|"
    r"ngoài phạm vi|không thuộc|không thể trả lời|xin lỗi.*không",
    re.IGNORECASE,
)


def score_faithfulness(client: genai.Client, question: str, answer: str, contexts: List[str]) -> float:
    """Fraction of answer claims directly supported by retrieved contexts."""
    ctx_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    prompt = f"""Bạn là người kiểm tra tính trung thực của câu trả lời RAG.

CONTEXTS:
{ctx_text}

CÂU HỎI: {question}

CÂU TRẢ LỜI: {answer}

Đánh giá FAITHFULNESS: tỷ lệ các luận điểm trong CÂU TRẢ LỜI được hỗ trợ trực tiếp bởi CONTEXTS.
- 1.0: mọi luận điểm đều có trong contexts
- 0.5: khoảng một nửa luận điểm có trong contexts
- 0.0: câu trả lời chứa thông tin không có trong contexts

Chỉ trả lời một số thập phân từ 0.0 đến 1.0."""
    return _score_prompt(client, prompt)


def score_answer_relevancy(client: genai.Client, question: str, answer: str) -> float:
    """How well the answer addresses the question."""
    prompt = f"""Bạn đánh giá mức độ liên quan của câu trả lời RAG.

CÂU HỎI: {question}

CÂU TRẢ LỜI: {answer}

Đánh giá ANSWER RELEVANCY: câu trả lời có trực tiếp giải quyết câu hỏi không?
- 1.0: trả lời trực tiếp và đầy đủ
- 0.5: trả lời một phần hoặc lạc đề
- 0.0: hoàn toàn không liên quan

Chỉ trả lời một số thập phân từ 0.0 đến 1.0."""
    return _score_prompt(client, prompt)


def score_correctness(client: genai.Client, question: str, answer: str, ground_truth: str) -> float:
    """Semantic similarity between answer and ground truth (independent of contexts)."""
    prompt = f"""Bạn đánh giá tính chính xác của câu trả lời so với đáp án chuẩn.

CÂU HỎI: {question}

ĐÁP ÁN CHUẨN: {ground_truth}

CÂU TRẢ LỜI CỦA HỆ THỐNG: {answer}

Đánh giá CORRECTNESS: mức độ tương đồng về nội dung/ý nghĩa giữa câu trả lời và đáp án chuẩn.
- 1.0: đúng hoàn toàn, thông tin khớp với đáp án chuẩn
- 0.5: đúng một phần, còn thiếu hoặc sai một số chi tiết quan trọng
- 0.0: sai hoàn toàn hoặc mâu thuẫn với đáp án chuẩn

Chỉ trả lời một số thập phân từ 0.0 đến 1.0."""
    return _score_prompt(client, prompt)


def score_noise_robustness(
    client: genai.Client,
    question: str,
    answer: str,
    ground_truth: str,
    noisy_contexts: List[str],
) -> float:
    """Whether answer is correct despite noisy (irrelevant) contexts being injected."""
    ctx_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(noisy_contexts))
    prompt = f"""Bạn đánh giá khả năng chống nhiễu của hệ thống RAG.

CÁC CONTEXT (CÓ LẪN THÔNG TIN KHÔNG LIÊN QUAN):
{ctx_text}

CÂU HỎI: {question}
ĐÁP ÁN CHUẨN: {ground_truth}

CÂU TRẢ LỜI CỦA HỆ THỐNG: {answer}

Đánh giá NOISE ROBUSTNESS: câu trả lời có vẫn đúng mặc dù contexts có lẫn thông tin nhiễu không?
- 1.0: câu trả lời chính xác, không bị ảnh hưởng bởi nhiễu
- 0.5: câu trả lời một phần đúng nhưng có bị ảnh hưởng bởi nhiễu
- 0.0: câu trả lời sai hoặc bị đánh lừa bởi thông tin nhiễu

Chỉ trả lời một số thập phân từ 0.0 đến 1.0."""
    return _score_prompt(client, prompt)


def score_negative_rejection(client: genai.Client, question: str, answer: str) -> float:
    """1.0 if system correctly refuses an unanswerable/out-of-domain question, 0.0 if it hallucinates."""
    # Fast heuristic: check for Vietnamese refusal phrases
    if _REFUSAL_PATTERN.search(answer):
        return 1.0
    # LLM fallback for ambiguous cases
    prompt = f"""Bạn đánh giá xem hệ thống RAG có từ chối đúng đắn một câu hỏi ngoài phạm vi không.

CÂU HỎI: {question}

CÂU TRẢ LỜI: {answer}

Đánh giá NEGATIVE REJECTION: hệ thống có nhận ra câu hỏi ngoài phạm vi và từ chối trả lời không?
- 1.0: hệ thống từ chối rõ ràng, không bịa đặt thông tin
- 0.0: hệ thống bịa đặt câu trả lời hoặc trả lời như thể câu hỏi nằm trong phạm vi

Chỉ trả lời một số thập phân từ 0.0 đến 1.0."""
    return _score_prompt(client, prompt)
