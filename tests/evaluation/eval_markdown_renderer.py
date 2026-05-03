"""Renders a RAG eval summary dict into a Markdown report string."""

METRIC_LABELS = {
    "faithfulness": "Faithfulness (answer không bịa thông tin)",
    "answer_relevancy": "Answer Relevancy (answer trả lời đúng câu hỏi)",
    "context_precision": "Context Precision (chunks retrieved liên quan)",
    "context_recall": "Context Recall (chunks đủ để trả lời)",
    "correctness": "Correctness (answer khớp với ground truth)",
    "noise_robustness": "Noise Robustness (chống nhiễu context)",
    "negative_rejection": "Negative Rejection (từ chối câu hỏi ngoài phạm vi)",
}


def score_badge(score: float) -> str:
    if score >= 0.7:
        return "✅ Tốt"
    if score >= 0.5:
        return "⚠️ Trung bình"
    return "❌ Cần cải thiện"


def generate_markdown(report: dict) -> str:
    ts = report.get("timestamp", "N/A")
    total = report.get("total_questions", 0)
    metrics = report.get("metrics", {})
    by_cat = report.get("by_category", {})
    by_diff = report.get("by_difficulty", {})
    worst = report.get("worst_context_recall", [])
    latency = report.get("latency", {})
    judge_model = report.get("judge_model", "gemini-2.0-flash")

    lines = [
        "# Kết quả đánh giá RAG Chatbot UIT", "",
        f"**Thời gian:** {ts}  ", f"**Số câu hỏi:** {total}  ",
        f"**Mô hình LLM judge:** {judge_model}  ", "", "---", "",
        "## Tổng quan metrics", "",
        "| Metric | Score | Đánh giá |", "|--------|-------|----------|",
    ]
    for key, label in METRIC_LABELS.items():
        if key not in metrics:
            continue
        score = metrics[key]
        lines.append(f"| {label} | **{score:.4f}** | {score_badge(score)} |")

    avg_all = sum(metrics.values()) / len(metrics) if metrics else 0
    lines += ["", f"**Điểm trung bình tổng:** {avg_all:.4f}", "", "---", "",
              "## Kết quả theo category", "",
              "| Category | Faithfulness | Answer Rel. | Ctx Precision | Ctx Recall | Avg |",
              "|----------|-------------|-------------|---------------|------------|-----|"]
    for cat, scores in sorted(by_cat.items()):
        f = scores.get("faithfulness", 0)
        ar = scores.get("answer_relevancy", 0)
        cp = scores.get("context_precision", 0)
        cr = scores.get("context_recall", 0)
        avg = (f + ar + cp + cr) / 4
        lines.append(f"| {cat} | {f:.3f} | {ar:.3f} | {cp:.3f} | {cr:.3f} | **{avg:.3f}** |")

    lines += ["", "---", "", "## Kết quả theo độ khó", "",
              "| Độ khó | Faithfulness | Answer Rel. | Ctx Precision | Ctx Recall |",
              "|--------|-------------|-------------|---------------|------------|"]
    for diff, scores in sorted(by_diff.items()):
        f = scores.get("faithfulness", 0); ar = scores.get("answer_relevancy", 0)
        cp = scores.get("context_precision", 0); cr = scores.get("context_recall", 0)
        lines.append(f"| {diff} | {f:.3f} | {ar:.3f} | {cp:.3f} | {cr:.3f} |")

    if worst:
        lines += ["", "---", "", "## Câu hỏi có Context Recall thấp nhất (retrieval miss)", "",
                  "| Câu hỏi | Category | Context Recall | Faithfulness |",
                  "|---------|----------|----------------|--------------|"]
        for item in worst:
            q = item["question"][:60] + ("…" if len(item["question"]) > 60 else "")
            lines.append(f"| {q} | {item['category']} | {item['context_recall']:.3f} | {item['faithfulness']:.3f} |")

    if latency:
        lines += ["", "---", "", "## Latency (RAG pipeline)", "",
                  "| Metric | Giá trị (ms) |", "|--------|-------------|",
                  f"| p50 | {latency.get('p50_ms', 'N/A')} |",
                  f"| p90 | {latency.get('p90_ms', 'N/A')} |",
                  f"| p99 | {latency.get('p99_ms', 'N/A')} |",
                  f"| avg | {latency.get('avg_ms', 'N/A')} |",
                  f"| min | {latency.get('min_ms', 'N/A')} |",
                  f"| max | {latency.get('max_ms', 'N/A')} |"]
        per_cat = latency.get("per_category_avg_ms", {})
        if per_cat:
            lines += ["", "**Latency trung bình theo category (ms):**", "",
                      "| Category | Avg (ms) |", "|----------|----------|"]
            for cat, avg_ms in sorted(per_cat.items()):
                lines.append(f"| {cat} | {avg_ms} |")

    lines += ["", "---", "", "## Phân tích & Hướng cải thiện", ""]
    issues = []
    if metrics.get("faithfulness", 1) < 0.7:
        issues.append("- **Faithfulness thấp** → Gemini thêm thông tin ngoài context. Giải pháp: tăng constraint trong system prompt, giảm `temperature`.")
    if metrics.get("context_precision", 1) < 0.7:
        issues.append("- **Context Precision thấp** → Nhiều chunks retrieved không liên quan. Giải pháp: tăng `MIN_SCORE`, điều chỉnh `top_k`, cải thiện chunking.")
    if metrics.get("context_recall", 1) < 0.7:
        issues.append("- **Context Recall thấp** → Hệ thống bỏ sót chunks quan trọng. Giải pháp: tăng `top_k`, cải thiện query expansion, kiểm tra OCR quality.")
    if metrics.get("answer_relevancy", 1) < 0.7:
        issues.append("- **Answer Relevancy thấp** → Câu trả lời không focus vào câu hỏi. Giải pháp: cải thiện system prompt, thêm instruction về conciseness.")

    if issues:
        lines.append("### Vấn đề phát hiện")
        lines += issues
    else:
        lines.append("✅ Tất cả metrics đạt ngưỡng tốt (≥ 0.7). Hệ thống hoạt động ổn định.")

    lines += ["", "### Hướng cải thiện tiếp theo",
              "1. Semantic chunking theo điều khoản thay vì fixed-size",
              "2. Cải thiện OCR quality cho PDF scan (Điều 28, TT 21/2019)",
              "3. Tăng test coverage với câu hỏi multi-hop (cross-document)",
              "4. Thêm guardrails phát hiện câu hỏi ngoài phạm vi"]
    return "\n".join(lines)
