"""Q&A dataset generation pipeline — manual seeds + template + LLM augmentation.

Usage:
    python src/scraper/qa-generator.py --target 5000
    python src/scraper/qa-generator.py --target 5000 --skip-llm
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.scraper.qa_templates import (
    QUESTION_TEMPLATES,
    fill_template,
    create_chatml_pair,
    SYSTEM_PROMPT_VI,
)
from src.scraper.qa_validator import (
    validate_dataset,
    deduplicate_pairs,
    split_dataset,
    save_jsonl,
)

# ---------------------------------------------------------------------------
# Educational keywords for topic extraction
# ---------------------------------------------------------------------------

EDU_KEYWORDS = [
    "tuyển sinh", "đăng ký", "học phí", "lịch học",
    "điểm", "tốt nghiệp", "học bổng", "thực tập",
    "chương trình đào tạo", "quy chế", "môn học",
    "sinh viên", "giảng viên", "khoa", "ngành",
    "học kỳ", "tín chỉ", "cố vấn học tập", "bảng điểm",
    "xét duyệt", "miễn giảm", "hoãn thi", "phúc khảo",
]


def extract_topics(documents: list[dict]) -> dict[str, list[dict]]:
    """Group documents by educational keyword topics."""
    topics: dict[str, list[dict]] = {}
    for doc in documents:
        content_lower = doc["content"].lower()
        for kw in EDU_KEYWORDS:
            if kw in content_lower:
                if kw not in topics:
                    topics[kw] = []
                topics[kw].append(doc)
    return topics


def extract_relevant_paragraph(content: str, topic: str) -> str | None:
    """Return the first paragraph containing the topic keyword (min 50 chars)."""
    for para in content.split("\n\n"):
        if topic.lower() in para.lower() and len(para) > 50:
            return para.strip()
    # Fallback: first substantial paragraph
    for para in content.split("\n\n"):
        if len(para) > 50:
            return para.strip()
    return None


# ---------------------------------------------------------------------------
# Generation strategies
# ---------------------------------------------------------------------------

def generate_from_templates(
    topics: dict[str, list[dict]],
) -> list[dict]:
    """Generate Q&A pairs by combining topic keywords with question templates."""
    pairs: list[dict] = []
    for topic, docs in topics.items():
        for template in QUESTION_TEMPLATES:
            question = fill_template(template, topic)
            best_doc = docs[0] if docs else None
            if best_doc is None:
                continue
            answer = extract_relevant_paragraph(best_doc["content"], topic)
            if answer and len(answer.split()) >= 10:
                pairs.append(create_chatml_pair(question, answer))
    print(f"[qa-generator] Template-based: {len(pairs)} pairs")
    return pairs


def augment_with_gemini(
    documents: list[dict],
    seed_pairs: list[dict],
    target_count: int = 5000,
) -> list[dict]:
    """Use Google Gemini to generate additional Q&A pairs from source documents.

    Requires GEMINI_API_KEY in environment.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        print("[qa-generator] google-generativeai not installed — skipping LLM augmentation")
        return seed_pairs

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("[qa-generator] GEMINI_API_KEY not set — skipping LLM augmentation")
        return seed_pairs

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    generated = list(seed_pairs)
    print(f"[qa-generator] LLM augmentation: target={target_count}, starting from {len(generated)}")

    for doc in documents:
        if len(generated) >= target_count:
            break

        content_excerpt = doc["content"][:2000]
        prompt = (
            "Dựa trên nội dung sau đây, hãy tạo 3 cặp câu hỏi-trả lời bằng tiếng Việt.\n"
            "Câu hỏi phải là những gì sinh viên thường hỏi về thông tin đào tạo.\n"
            "Trả lời phải chính xác dựa trên nội dung đã cho.\n\n"
            f"Nội dung:\n{content_excerpt}\n\n"
            "Trả về đúng định dạng JSON:\n"
            '[{"question": "...", "answer": "..."}, ...]'
        )

        try:
            response = model.generate_content(prompt)
            # Strip possible markdown code fences
            text = response.text.strip().lstrip("```json").lstrip("```").rstrip("```")
            qa_list = json.loads(text)
            for qa in qa_list:
                q, a = qa.get("question", ""), qa.get("answer", "")
                if q and a:
                    generated.append(create_chatml_pair(q, a))
            # Respect free tier: 60 req/min
            time.sleep(1.1)
        except Exception as exc:
            print(f"[qa-generator] Gemini error: {exc}")
            continue

    print(f"[qa-generator] LLM augmentation complete: {len(generated)} pairs")
    return generated[:target_count]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    documents_path: str = "data/processed/all_documents.json",
    seeds_path: str = "data/training/seed_pairs.json",
    target: int = 5000,
    skip_llm: bool = False,
) -> None:
    # Load processed documents
    with open(documents_path, "r", encoding="utf-8") as fh:
        documents = json.load(fh)
    print(f"[qa-generator] Loaded {len(documents)} source documents")

    # Load manual seed pairs
    seed_pairs: list[dict] = []
    if os.path.exists(seeds_path):
        with open(seeds_path, "r", encoding="utf-8") as fh:
            seed_pairs = json.load(fh)
        print(f"[qa-generator] Loaded {len(seed_pairs)} seed pairs")

    # Template-based generation
    topics = extract_topics(documents)
    template_pairs = generate_from_templates(topics)

    all_pairs = seed_pairs + template_pairs

    # LLM augmentation
    if not skip_llm and len(all_pairs) < target:
        all_pairs = augment_with_gemini(documents, all_pairs, target_count=target)

    # Validate and deduplicate
    all_pairs = validate_dataset(all_pairs)
    all_pairs = deduplicate_pairs(all_pairs)

    # Split and save
    train, eval_set = split_dataset(all_pairs, eval_ratio=0.1)
    os.makedirs("data/training", exist_ok=True)
    save_jsonl(train, "data/training/train.jsonl")
    save_jsonl(eval_set, "data/training/eval.jsonl")

    print(f"\n[qa-generator] Done — train={len(train)}, eval={len(eval_set)}")
    print("Next: upload data/training/*.jsonl to Google Drive for Colab")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fine-tuning Q&A dataset")
    parser.add_argument("--data", default="data/processed/all_documents.json")
    parser.add_argument("--seeds", default="data/training/seed_pairs.json")
    parser.add_argument("--target", type=int, default=5000)
    parser.add_argument("--skip-llm", action="store_true", help="Skip Gemini augmentation")
    args = parser.parse_args()
    run(
        documents_path=args.data,
        seeds_path=args.seeds,
        target=args.target,
        skip_llm=args.skip_llm,
    )
