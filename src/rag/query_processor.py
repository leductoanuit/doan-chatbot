"""Query processing — context rewriting, multi-query generation, and intent routing."""

import re
from typing import Dict, List, Optional

# Patterns that signal the query depends on prior conversation context
_CONTEXT_DEPENDENCY_SIGNALS = [
    r"\b(nó|họ|đó|này|kia|vậy|thế)\b",       # demonstrative pronouns
    r"\b(còn|thêm|nữa|khác|tiếp theo)\b",     # continuity signals
    r"^(vậy|thế thì|thế còn|còn)\b",          # sentence starters
]

# Mapping: keyword signals → system_type filter value (must match Qdrant payload)
_INTENT_RULES: List[tuple] = [
    (
        ["tuyển sinh", "xét tuyển", "điều kiện vào", "hồ sơ đăng ký",
         "nộp hồ sơ", "thời hạn đăng ký", "kết quả trúng tuyển",
         "học phí", "chi phí", "tiền học", "đóng tiền",
         "miễn giảm học phí", "hoàn học phí",
         "ngành", "chương trình", "bằng cử nhân", "hệ từ xa",
         "đào tạo từ xa", "liên thông", "văn bằng 2"],
        "tuyen_sinh",
    ),
    (
        ["chứng chỉ", "ứng dụng cntt", "chứng chỉ tin học",
         "thi chứng chỉ", "cấp chứng chỉ", "sát hạch", "kỹ năng cntt"],
        "chung_chi",
    ),
    (
        ["quy chế", "điều khoản", "quy định học vụ", "kỷ luật", "học vụ",
         "bảo lưu", "thôi học", "cảnh báo học vụ", "tốt nghiệp",
         "chương trình đào tạo", "môn học", "tín chỉ",
         "chương trình", "học gì", "học những gì", "môn gì"],
        "dao_tao",
    ),
]


def detect_context_dependency(query: str) -> bool:
    """Return True if query likely depends on prior conversation context."""
    q = query.lower().strip()
    for pattern in _CONTEXT_DEPENDENCY_SIGNALS:
        if re.search(pattern, q):
            return True
    # Very short queries without domain keywords likely need context to be useful
    if len(q.split()) <= 4:
        domain_signals = ["học phí", "tuyển sinh", "quy chế", "môn học", "tín chỉ", "ngành"]
        if not any(kw in q for kw in domain_signals):
            return True
    return False


def rewrite_standalone_query(
    query: str,
    history: List[Dict],
    llm_client,
) -> str:
    """Rewrite a context-dependent query into a standalone version using history.

    Only calls LLM when history is non-empty and dependency is detected.
    Falls back to original query on error or when no rewrite is needed.
    """
    if not history or not detect_context_dependency(query):
        return query

    recent = history[-3:]
    history_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Bot'}: {m['content'][:200]}"
        for m in recent
    )
    prompt = (
        f"Lịch sử hội thoại:\n{history_text}\n\n"
        f"Câu hỏi mới: {query}\n\n"
        "Hãy viết lại câu hỏi trên thành một câu hỏi độc lập, đầy đủ context, "
        "không cần đọc lịch sử. Chỉ trả về câu hỏi đã viết lại, không giải thích."
    )
    try:
        rewritten = llm_client.generate(query=prompt, temperature=0.1, max_tokens=128)
        return rewritten.strip() if rewritten else query
    except Exception:
        return query


def generate_query_variants(
    query: str,
    llm_client,
    n: int = 2,
) -> List[str]:
    """Generate n alternative phrasings of the query for RAG Fusion.

    Returns a list of variants (not including original).
    Falls back to empty list on error.
    """
    prompt = (
        f"Viết {n} cách diễn đạt khác nhau cho câu hỏi sau, "
        f"giữ nguyên ý nghĩa, mỗi câu một dòng, không đánh số:\n\n{query}"
    )
    try:
        raw = llm_client.generate(query=prompt, temperature=0.5, max_tokens=200)
        variants = [line.strip() for line in raw.strip().splitlines() if line.strip()]
        return variants[:n]
    except Exception:
        return []


def reciprocal_rank_fusion(
    results_lists: List[List[Dict]],
    k: int = 60,
) -> List[Dict]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion (RRF).

    Uses rank position instead of raw scores for stable cross-query merging.
    Formula: RRF_score(d) = sum(1 / (k + rank_i(d))) across all query lists.
    """
    scores: Dict[int, float] = {}
    chunks: Dict[int, Dict] = {}

    for results in results_lists:
        for rank, chunk in enumerate(results):
            key = hash(chunk["content"][:100])
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in chunks:
                chunks[key] = chunk

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    merged = []
    for key in sorted_keys:
        chunk = chunks[key].copy()
        chunk["rrf_score"] = scores[key]
        chunk["final_score"] = scores[key]
        merged.append(chunk)

    return merged


def classify_query_intent(query: str) -> Optional[str]:
    """Return system_type filter string or None when intent is ambiguous.

    Returns None when query spans multiple intents to avoid over-filtering.
    """
    q = query.lower()
    matched: set = set()

    for keywords, system_type in _INTENT_RULES:
        if any(kw in q for kw in keywords):
            matched.add(system_type)

    # Only filter when exactly one intent matched — avoid false precision
    return matched.pop() if len(matched) == 1 else None
