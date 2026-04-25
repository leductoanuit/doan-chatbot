"""Generate a markdown evaluation report from a JSON eval result file.

Usage:
  python tests/evaluation/generate-eval-report.py results/eval-report-*.json
  python tests/evaluation/generate-eval-report.py  # auto-picks latest report
"""

import json
import sys
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent / "results"

METRIC_LABELS = {
    "faithfulness": "Faithfulness (answer không bịa thông tin)",
    "answer_relevancy": "Answer Relevancy (answer trả lời đúng câu hỏi)",
    "context_precision": "Context Precision (chunks retrieved liên quan)",
    "context_recall": "Context Recall (chunks đủ để trả lời)",
}


def score_badge(score: float) -> str:
    if score >= 0.7:
        return "✅ Tốt"
    if score >= 0.5:
        return "⚠️ Trung bình"
    return "❌ Cần cải thiện"


def load_report(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def generate_markdown(report: dict) -> str:
    ts = report.get("timestamp", "N/A")
    total = report.get("total_questions", 0)
    metrics = report.get("metrics", {})
    by_cat = report.get("by_category", {})
    by_diff = report.get("by_difficulty", {})
    worst = report.get("worst_context_recall", [])
    judge_model = report.get("judge_model", "gemini-2.0-flash")

    lines = [
        "# Kết quả đánh giá RAG Chatbot UIT",
        "",
        f"**Thời gian:** {ts}  ",
        f"**Số câu hỏi:** {total}  ",
        f"**Mô hình LLM judge:** {judge_model}  ",
        "",
        "---",
        "",
        "## Tổng quan metrics",
        "",
        "| Metric | Score | Đánh giá |",
        "|--------|-------|----------|",
    ]
    for key, label in METRIC_LABELS.items():
        score = metrics.get(key, 0)
        lines.append(f"| {label} | **{score:.4f}** | {score_badge(score)} |")

    avg_all = sum(metrics.values()) / len(metrics) if metrics else 0
    lines += [
        "",
        f"**Điểm trung bình tổng:** {avg_all:.4f}",
        "",
        "---",
        "",
        "## Kết quả theo category",
        "",
        "| Category | Faithfulness | Answer Rel. | Ctx Precision | Ctx Recall | Avg |",
        "|----------|-------------|-------------|---------------|------------|-----|",
    ]
    for cat, scores in sorted(by_cat.items()):
        f = scores.get("faithfulness", 0)
        ar = scores.get("answer_relevancy", 0)
        cp = scores.get("context_precision", 0)
        cr = scores.get("context_recall", 0)
        avg = (f + ar + cp + cr) / 4
        lines.append(f"| {cat} | {f:.3f} | {ar:.3f} | {cp:.3f} | {cr:.3f} | **{avg:.3f}** |")

    lines += [
        "",
        "---",
        "",
        "## Kết quả theo độ khó",
        "",
        "| Độ khó | Faithfulness | Answer Rel. | Ctx Precision | Ctx Recall |",
        "|--------|-------------|-------------|---------------|------------|",
    ]
    for diff, scores in sorted(by_diff.items()):
        f = scores.get("faithfulness", 0)
        ar = scores.get("answer_relevancy", 0)
        cp = scores.get("context_precision", 0)
        cr = scores.get("context_recall", 0)
        lines.append(f"| {diff} | {f:.3f} | {ar:.3f} | {cp:.3f} | {cr:.3f} |")

    if worst:
        lines += [
            "",
            "---",
            "",
            "## Câu hỏi có Context Recall thấp nhất (retrieval miss)",
            "",
            "| Câu hỏi | Category | Context Recall | Faithfulness |",
            "|---------|----------|----------------|--------------|",
        ]
        for item in worst:
            q = item["question"][:60] + ("…" if len(item["question"]) > 60 else "")
            lines.append(
                f"| {q} | {item['category']} "
                f"| {item['context_recall']:.3f} | {item['faithfulness']:.3f} |"
            )

    lines += [
        "",
        "---",
        "",
        "## Phân tích & Hướng cải thiện",
        "",
    ]

    # Auto-generate analysis based on scores
    issues = []
    if metrics.get("faithfulness", 1) < 0.7:
        issues.append(
            "- **Faithfulness thấp** → Gemini thêm thông tin ngoài context. "
            "Giải pháp: tăng constraint trong system prompt, giảm `temperature`."
        )
    if metrics.get("context_precision", 1) < 0.7:
        issues.append(
            "- **Context Precision thấp** → Nhiều chunks retrieved không liên quan. "
            "Giải pháp: tăng `MIN_SCORE`, điều chỉnh `top_k`, cải thiện chunking."
        )
    if metrics.get("context_recall", 1) < 0.7:
        issues.append(
            "- **Context Recall thấp** → Hệ thống bỏ sót chunks quan trọng. "
            "Giải pháp: tăng `top_k`, cải thiện query expansion, kiểm tra OCR quality."
        )
    if metrics.get("answer_relevancy", 1) < 0.7:
        issues.append(
            "- **Answer Relevancy thấp** → Câu trả lời không focus vào câu hỏi. "
            "Giải pháp: cải thiện system prompt, thêm instruction về conciseness."
        )

    if issues:
        lines.append("### Vấn đề phát hiện")
        lines += issues
    else:
        lines.append("✅ Tất cả metrics đạt ngưỡng tốt (≥ 0.7). Hệ thống hoạt động ổn định.")

    lines += [
        "",
        "### Hướng cải thiện tiếp theo",
        "1. Semantic chunking theo điều khoản thay vì fixed-size",
        "2. Cải thiện OCR quality cho PDF scan (Điều 28, TT 21/2019)",
        "3. Tăng test coverage với câu hỏi multi-hop (cross-document)",
        "4. Thêm guardrails phát hiện câu hỏi ngoài phạm vi",
    ]

    return "\n".join(lines)


def main():
    # Find report file
    if len(sys.argv) > 1:
        report_path = Path(sys.argv[1])
    else:
        if not RESULTS_DIR.exists():
            print(f"Results directory not found: {RESULTS_DIR}")
            print("Run: python tests/evaluation/run-evaluation.py first")
            sys.exit(1)
        reports = sorted(RESULTS_DIR.glob("eval-summary-*.json"))
        if not reports:
            print("No eval-report-*.json found in", RESULTS_DIR)
            sys.exit(1)
        report_path = reports[-1]
        print(f"Using latest report: {report_path}")

    report = load_report(report_path)
    md = generate_markdown(report)

    out_path = report_path.with_name(
        report_path.stem.replace("eval-report", "eval-summary") + ".md"
    )
    out_path.write_text(md, encoding="utf-8")
    print(f"Markdown report saved to: {out_path}")
    print("\n" + md[:500] + "\n...")


if __name__ == "__main__":
    main()
