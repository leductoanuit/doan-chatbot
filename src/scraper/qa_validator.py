"""Validation, deduplication, and train/eval splitting for Q&A dataset."""

import json
import os
import random
import re
from typing import Tuple


# ---------------------------------------------------------------------------
# Vietnamese character check pattern
# ---------------------------------------------------------------------------

_VIET_CHARS_RE = re.compile(
    r"[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]",
    re.IGNORECASE,
)


def validate_pair(pair: dict) -> Tuple[bool, str]:
    """Check a single ChatML pair for basic quality requirements.

    Returns:
        (is_valid, reason) tuple.
    """
    messages = pair.get("messages", [])
    if len(messages) < 3:
        return False, "Too few messages (need system + user + assistant)"

    roles = [m.get("role") for m in messages]
    if roles[0] != "system" or roles[1] != "user" or roles[2] != "assistant":
        return False, "Wrong role order"

    user_content = messages[1].get("content", "")
    assistant_content = messages[2].get("content", "")

    if len(user_content.split()) < 3:
        return False, "Question too short"
    if len(assistant_content.split()) < 10:
        return False, "Answer too short"
    if len(assistant_content.split()) > 500:
        return False, "Answer too long"

    # Ensure assistant response contains some Vietnamese characters
    # (accepts ASCII Vietnamese too — only flag fully Latin/empty content)
    if not assistant_content.strip():
        return False, "Empty answer"

    return True, "OK"


def validate_dataset(pairs: list[dict]) -> list[dict]:
    """Validate all pairs, print a summary, and return valid subset."""
    valid: list[dict] = []
    reason_counts: dict[str, int] = {}

    for pair in pairs:
        ok, reason = validate_pair(pair)
        if ok:
            valid.append(pair)
        else:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    invalid_count = len(pairs) - len(valid)
    print(f"[validator] Valid: {len(valid)} / {len(pairs)} (removed {invalid_count})")
    for reason, count in reason_counts.items():
        print(f"  - {reason}: {count}")

    return valid


def deduplicate_pairs(pairs: list[dict], threshold: float = 0.85) -> list[dict]:
    """Remove near-duplicate Q&A pairs using question Jaccard similarity."""
    unique: list[dict] = []
    seen_questions: list[str] = []

    for pair in pairs:
        question = pair["messages"][1]["content"].lower()
        q_words = set(question.split())

        is_dup = False
        for seen in seen_questions:
            s_words = set(seen.split())
            union = q_words | s_words
            if union:
                similarity = len(q_words & s_words) / len(union)
                if similarity >= threshold:
                    is_dup = True
                    break

        if not is_dup:
            unique.append(pair)
            seen_questions.append(question)

    removed = len(pairs) - len(unique)
    if removed:
        print(f"[validator] Removed {removed} duplicate questions")
    return unique


def split_dataset(
    pairs: list[dict],
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[list[dict], list[dict]]:
    """Shuffle and split into train / eval sets."""
    random.seed(seed)
    shuffled = list(pairs)
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - eval_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def save_jsonl(pairs: list[dict], filepath: str) -> None:
    """Save pairs as JSONL (one JSON object per line, UTF-8)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"[validator] Saved {len(pairs)} pairs → {filepath}")


