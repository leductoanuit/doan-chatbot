"""
Generate evaluation questions from processed documents using Gemini.
Expands eval-dataset.json to ~100 questions while preserving existing ones.

Usage:
    python generate-eval-dataset.py [--target 100] [--dry-run]
"""

import argparse
import json
import logging
import os
import random
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DOCS_PATH = Path(__file__).parent.parent.parent / "data/processed/all_documents_final.json"
DATASET_PATH = Path(__file__).parent / "eval-dataset.json"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Map system_type + title keywords → eval category
CATEGORY_MAP = {
    "tuyen_sinh": "tuyen_sinh",
    "chung_chi": "ctdt",
}

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

# Target distribution per category
TARGET_DISTRIBUTION = {
    "tuyen_sinh": 20,
    "hoc_vu": 25,
    "hoc_phi": 15,
    "ctdt": 20,
    "quy_che": 20,
}

# Keywords to map document content to eval categories
CATEGORY_KEYWORDS = {
    "hoc_phi": ["học phí", "lệ phí", "thanh toán", "đóng tiền", "miễn giảm học phí"],
    "hoc_vu": ["học vụ", "đăng ký học", "tín chỉ", "điểm", "thi", "học kỳ", "bảo lưu", "thôi học"],
    "quy_che": ["quy chế", "quy định", "kỷ luật", "vi phạm", "xử lý"],
    "ctdt": ["chương trình đào tạo", "môn học", "chứng chỉ", "bằng cấp", "chuẩn đầu ra"],
    "tuyen_sinh": ["tuyển sinh", "xét tuyển", "nhập học", "hồ sơ", "điều kiện đầu vào"],
}

GENERATION_PROMPT = """Bạn là chuyên gia tạo bộ câu hỏi đánh giá chatbot tư vấn đại học.

Dựa vào đoạn văn bản sau từ tài liệu của trường đại học:

<document>
{content}
</document>

Hãy tạo {count} câu hỏi đánh giá với độ khó "{difficulty}" theo định dạng JSON sau:
[
  {{
    "question": "câu hỏi tự nhiên mà sinh viên/học viên thường hỏi",
    "ground_truth": "câu trả lời chính xác dựa hoàn toàn vào nội dung tài liệu",
    "reference_context": "đoạn văn bản gốc chứa thông tin để trả lời (copy nguyên văn từ tài liệu)"
  }}
]

Quy tắc:
- Câu hỏi phải tự nhiên, như sinh viên thật sự hỏi chatbot
- Ground truth phải có thể suy ra hoàn toàn từ tài liệu, không thêm thông tin ngoài
- Difficulty "{difficulty}": {difficulty_hint}
- Chỉ trả về JSON array, không thêm giải thích

JSON:"""


def make_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    return genai.Client(api_key=api_key)


def detect_category(doc: dict) -> str:
    """Detect eval category from document metadata and content."""
    system_type = doc.get("system_type", "")
    if system_type in CATEGORY_MAP:
        return CATEGORY_MAP[system_type]

    content = ((doc.get("content") or "") + " " + (doc.get("title") or "")).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in content for kw in keywords):
            return category

    return "hoc_vu"  # default


def generate_questions(
    client: genai.Client,
    doc: dict,
    category: str,
    difficulty: str,
    count: int = 2,
    retries: int = 3,
) -> list[dict]:
    """Call Gemini to generate questions from a document chunk."""
    content = doc.get("content", "").strip()
    if len(content) < 100:
        return []

    # Truncate very long documents
    if len(content) > 2000:
        content = content[:2000] + "..."

    difficulty_hint = (
        "câu hỏi trực tiếp, thông tin rõ ràng trong văn bản" if difficulty == "easy"
        else "câu hỏi cần tổng hợp 2-3 thông tin" if difficulty == "medium"
        else "câu hỏi cần suy luận hoặc so sánh, thông tin phân tán"
    )
    prompt = GENERATION_PROMPT.format(content=content, count=count, difficulty=difficulty, difficulty_hint=difficulty_hint)

    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=1024),
            )
            text = resp.text.strip()

            # Extract JSON array from response
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if not match:
                logger.warning("No JSON array in response for doc %s", doc.get("doc_id"))
                return []

            items = json.loads(match.group())
            results = []
            for item in items:
                if not item.get("question") or not item.get("ground_truth"):
                    continue
                results.append(
                    {
                        "question": item["question"].strip(),
                        "ground_truth": item["ground_truth"].strip(),
                        "reference_contexts": [item.get("reference_context", content[:500]).strip()],
                        "category": category,
                        "difficulty": difficulty,
                    }
                )
            return results

        except json.JSONDecodeError as exc:
            logger.warning("JSON parse failed (attempt %d): %s", attempt + 1, exc)
        except Exception as exc:
            logger.warning("Gemini call failed (attempt %d): %s", attempt + 1, exc)
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))

    return []


def load_existing_dataset() -> list[dict]:
    if DATASET_PATH.exists():
        return json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return []


def count_by_category(dataset: list[dict]) -> dict:
    counts = {cat: 0 for cat in TARGET_DISTRIBUTION}
    for item in dataset:
        cat = item.get("category", "hoc_vu")
        if cat in counts:
            counts[cat] += 1
    return counts


def select_docs_for_category(docs: list[dict], category: str, n: int) -> list[dict]:
    """Select diverse documents for a given category."""
    matching = [d for d in docs if detect_category(d) == category]
    if not matching:
        matching = docs  # fallback to all docs
    random.shuffle(matching)
    return matching[:n]


def main():
    parser = argparse.ArgumentParser(description="Generate eval dataset questions using Gemini")
    parser.add_argument("--target", type=int, default=100, help="Target total questions")
    parser.add_argument("--dry-run", action="store_true", help="Preview counts without calling API")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load existing data
    existing = load_existing_dataset()
    current_counts = count_by_category(existing)
    logger.info("Existing dataset: %d questions", len(existing))
    logger.info("Current distribution: %s", current_counts)

    # Calculate how many more needed per category
    needed = {}
    total_needed = args.target - len(existing)
    if total_needed <= 0:
        logger.info("Already have %d questions (target: %d). Nothing to do.", len(existing), args.target)
        return

    for cat, target in TARGET_DISTRIBUTION.items():
        gap = max(0, target - current_counts.get(cat, 0))
        needed[cat] = gap

    # Normalize if total needed < sum of gaps
    total_gaps = sum(needed.values())
    if total_gaps > total_needed:
        scale = total_needed / total_gaps
        needed = {cat: max(1, int(n * scale)) for cat, n in needed.items() if n > 0}

    logger.info("Questions to generate per category: %s", needed)

    if args.dry_run:
        print("\nDry run — no API calls made")
        print(f"Would generate {sum(needed.values())} questions to reach ~{args.target} total")
        return

    # Load documents
    docs = json.loads(DOCS_PATH.read_text(encoding="utf-8"))
    logger.info("Loaded %d documents", len(docs))

    client = make_client()
    new_questions = []

    for category, count in needed.items():
        if count == 0:
            continue

        logger.info("Generating %d questions for category: %s", count, category)
        selected_docs = select_docs_for_category(docs, category, n=count * 3)

        generated = 0
        for doc in selected_docs:
            if generated >= count:
                break

            remaining = count - generated
            per_call = min(2, remaining)

            # Vary difficulty: aim for ~50% easy, 35% medium, 15% hard
            difficulty = random.choices(
                DIFFICULTY_LEVELS, weights=[0.5, 0.35, 0.15], k=1
            )[0]

            questions = generate_questions(client, doc, category, difficulty, count=per_call)
            new_questions.extend(questions)
            generated += len(questions)

            if questions:
                logger.info("  +%d from doc %s (total new: %d)", len(questions), doc.get("doc_id", "?"), len(new_questions))

            time.sleep(0.5)  # rate limit

    # Merge and save
    combined = existing + new_questions
    DATASET_PATH.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")

    final_counts = count_by_category(combined)
    logger.info("Done. Total questions: %d", len(combined))
    logger.info("Final distribution: %s", final_counts)
    print(f"\nSaved {len(combined)} questions to {DATASET_PATH}")


if __name__ == "__main__":
    main()
