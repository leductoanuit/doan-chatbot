"""Vietnamese text cleaning, stop word removal, and deduplication utilities."""

import re
import unicodedata


# ---------------------------------------------------------------------------
# Vietnamese stop words (common function words that add no semantic value)
# ---------------------------------------------------------------------------

_VIETNAMESE_STOP_WORDS = {
    # Pronouns / articles
    "tôi", "tao", "mình", "chúng", "ta", "chúng ta", "họ", "nó",
    "anh", "chị", "em", "ông", "bà", "cô", "chú", "bác",
    # Conjunctions / prepositions
    "và", "hoặc", "hay", "nhưng", "mà", "vì", "nên", "do",
    "để", "với", "của", "cho", "từ", "đến", "trong", "ngoài",
    "trên", "dưới", "về", "theo", "bằng", "qua", "tại",
    # Adverbs / particles
    "là", "có", "được", "bị", "đã", "đang", "sẽ", "vẫn",
    "cũng", "rất", "quá", "lắm", "thì", "mới", "còn",
    "không", "chưa", "chẳng", "đều", "cả", "những", "các",
    "này", "đó", "kia", "ấy", "nào", "gì", "sao", "thế",
    "như", "nếu", "khi", "lúc", "bao giờ", "vậy", "thôi",
    "rồi", "lại", "ra", "vào", "lên", "xuống",
    # Filler words
    "ạ", "à", "ơi", "nhé", "nha", "nhỉ", "hả", "ừ",
}


# ---------------------------------------------------------------------------
# Boilerplate patterns common on daa.uit.edu.vn
# ---------------------------------------------------------------------------

_BOILERPLATE_PATTERNS = [
    r"Trang chủ\s*[»›|/].*",
    r"Copyright\s*©.*",
    r"Liên hệ:.*",
    r"Đăng nhập.*",
    r"Đăng ký.*",
    r"Bản quyền.*",
    r"Số lượt truy cập.*",
    r"Website:.*",
    r"Email:.*(?:daa|uit).*",
    r"Điện thoại:.*",
    r"Phòng Đào tạo Đại học.*liên hệ.*",
    r"Menu\s*chính.*",
    r"Tìm kiếm.*",
    r"Breadcrumb.*",
]

_BOILERPLATE_RE = re.compile(
    "|".join(_BOILERPLATE_PATTERNS),
    flags=re.IGNORECASE,
)


def remove_special_characters(text: str) -> str:
    """Remove special characters while preserving Vietnamese diacritics and punctuation."""
    # Keep: Vietnamese letters (with diacritics), digits, basic punctuation, whitespace
    # Remove: symbols like ●★▶►◆■□▪▫→←↑↓©®™§¶†‡•…, etc.
    text = re.sub(
        r"[^\w\s.,;:!?()\"'\-–—/\n]",
        " ",
        text,
        flags=re.UNICODE,
    )
    return text


def remove_stop_words(text: str) -> str:
    """Remove Vietnamese stop words from text."""
    words = text.split()
    filtered = [w for w in words if w.lower() not in _VIETNAMESE_STOP_WORDS]
    return " ".join(filtered)


def clean_vietnamese_text(
    text: str,
    strip_stop_words: bool = False,
) -> str:
    """Normalise, strip boilerplate, remove special chars, and compress whitespace.

    Args:
        text: Raw text to clean.
        strip_stop_words: If True, also remove Vietnamese stop words.
            Use True for search index / embedding input.
            Use False for display text (keeps readability).
    """
    # Unicode NFC — essential for Vietnamese diacritics
    text = unicodedata.normalize("NFC", text)

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove common boilerplate lines
    text = _BOILERPLATE_RE.sub("", text)

    # Remove special characters
    text = remove_special_characters(text)

    # Optionally remove stop words (for embedding/search index)
    if strip_stop_words:
        text = remove_stop_words(text)

    # Compress whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _jaccard(a: str, b: str) -> float:
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def deduplicate_chunks(
    chunks: list[dict],
    threshold: float = 0.9,
) -> list[dict]:
    """Remove near-duplicate chunks using Jaccard word-level similarity."""
    unique: list[dict] = []
    seen_texts: list[str] = []

    for chunk in chunks:
        content = chunk["content"]
        if any(_jaccard(content, seen) >= threshold for seen in seen_texts):
            continue
        unique.append(chunk)
        seen_texts.append(content)

    removed = len(chunks) - len(unique)
    if removed:
        print(f"[text-cleaner] Removed {removed} near-duplicate chunks")
    return unique


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = (
        "Trang chủ » Đào tạo » Thông báo\n\n"
        "Sinh viên cần nộp học phí trước ngày 15 hàng tháng.\n\n"
        "Copyright © 2024 UIT. Liên hệ: daa@uit.edu.vn\n"
    )
    print(clean_vietnamese_text(sample))
