"""Entry point for RAG system evaluation.

Usage:
  python tests/evaluation/run-evaluation.py
  python tests/evaluation/run-evaluation.py --sample 5   # quick test with 5 questions
  python tests/evaluation/run-evaluation.py --output tests/evaluation/results/

Requires:
  - Qdrant + PostgreSQL running (docker-compose up)
  - GEMINI_API_KEY set in .env
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root and eval dir to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).parent / "eval-dataset.json"
RESULTS_DIR = Path(__file__).parent / "results"


def load_dataset(sample: int = 0) -> list[dict]:
    with open(DATASET_PATH) as f:
        data = json.load(f)
    if sample and sample < len(data):
        logger.info("Using sample of %d questions (out of %d)", sample, len(data))
        return data[:sample]
    return data


def run_rag_pipeline(questions: list[dict]) -> list[dict]:
    """Run each question through the RAG pipeline and collect answers + contexts."""
    from src.rag.pipeline import RAGPipeline
    from src.rag.retriever import HybridRetriever
    from src.rag.reranker import BGEReranker

    logger.info("Initializing RAG pipeline...")
    retriever = HybridRetriever()
    reranker = BGEReranker()
    pipeline = RAGPipeline(retriever=retriever, llm_client=None)
    # inject reranker directly since RAGPipeline creates its own internally
    pipeline.reranker = reranker

    samples = []
    for item in tqdm(questions, desc="Running RAG queries"):
        try:
            # Call retriever directly to get full chunk content (not 150-char preview)
            expanded_q = pipeline._expand_query(item["question"])
            raw_results = retriever.hybrid_search(expanded_q, k=10, reranker=reranker)
            contexts = [r["content"] for r in raw_results]

            result = pipeline.query(item["question"], top_k=10)
            samples.append({
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "reference_contexts": item.get("reference_contexts", []),
                "answer": result["answer"],
                "contexts": contexts,
                "category": item.get("category", ""),
                "difficulty": item.get("difficulty", ""),
            })
        except Exception as exc:
            logger.warning("RAG query failed for '%s': %s", item["question"][:50], exc)
            samples.append({
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "reference_contexts": item.get("reference_contexts", []),
                "answer": "",
                "contexts": [],
                "category": item.get("category", ""),
                "difficulty": item.get("difficulty", ""),
            })

    return samples


def save_results(eval_result: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"eval-report-{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)
    logger.info("Results saved to %s", out_path)
    return out_path


def build_report(eval_result: dict) -> dict:
    """Aggregate scores by category and difficulty."""
    samples = eval_result["samples"]
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    def avg_scores(subset):
        if not subset:
            return {}
        return {m: round(sum(s["scores"][m] for s in subset) / len(subset), 4) for m in metrics}

    # By category
    categories = set(s["category"] for s in samples)
    by_category = {
        cat: avg_scores([s for s in samples if s["category"] == cat])
        for cat in sorted(categories)
    }

    # By difficulty
    difficulties = set(s["difficulty"] for s in samples)
    by_difficulty = {
        diff: avg_scores([s for s in samples if s["difficulty"] == diff])
        for diff in sorted(difficulties)
    }

    # Worst 5 by context_recall (retrieval misses)
    sorted_by_recall = sorted(samples, key=lambda s: s["scores"].get("context_recall", 1))
    worst_recall = [
        {
            "question": s["question"],
            "category": s["category"],
            "context_recall": s["scores"].get("context_recall"),
            "faithfulness": s["scores"].get("faithfulness"),
        }
        for s in sorted_by_recall[:5]
    ]

    return {
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(samples),
        "metrics": eval_result["aggregate"],
        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "worst_context_recall": worst_recall,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG chatbot quality")
    parser.add_argument("--sample", type=int, default=0, help="Use N questions (0 = all)")
    parser.add_argument("--output", default=str(RESULTS_DIR), help="Output directory")
    parser.add_argument("--skip-rag", action="store_true", help="Load existing RAG outputs (debug)")
    args = parser.parse_args()

    output_dir = Path(args.output)

    # Step 1: Load dataset
    questions = load_dataset(sample=args.sample)
    logger.info("Loaded %d questions from dataset", len(questions))

    # Step 2: Run RAG pipeline (or load cached results)
    if args.skip_rag:
        cached = sorted(output_dir.glob("eval-report-*.json"))
        if not cached:
            logger.error("--skip-rag: no cached eval-report-*.json found in %s", output_dir)
            sys.exit(1)
        logger.info("--skip-rag: loading cached samples from %s", cached[-1])
        samples = json.loads(cached[-1].read_text())["samples"]
    else:
        logger.info("Step 1/3: Running RAG pipeline...")
        samples = run_rag_pipeline(questions)

    # Step 3: Evaluate with LLM judge
    logger.info("Step 2/3: Running LLM-judge evaluation (this takes a few minutes)...")
    from ragas_evaluator import RagasEvaluator
    evaluator = RagasEvaluator()
    eval_result = evaluator.evaluate(samples)

    # Step 4: Build report and save
    logger.info("Step 3/3: Building report...")
    report = build_report(eval_result)

    # Save both raw + summary
    raw_path = save_results(eval_result, output_dir)
    summary_path = raw_path.with_name(raw_path.stem.replace("eval-report", "eval-summary") + ".json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Print summary to stdout
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Questions evaluated: {report['total_questions']}")
    print("\nMetrics:")
    for metric, score in report["metrics"].items():
        status = "✅" if score >= 0.7 else "⚠️ " if score >= 0.5 else "❌"
        print(f"  {status} {metric:25s}: {score:.4f}")
    print("\nBy category:")
    for cat, scores in report["by_category"].items():
        avg = sum(scores.values()) / len(scores) if scores else 0
        print(f"  {cat:20s}: avg={avg:.3f}")
    print(f"\nRaw results  : {raw_path}")
    print(f"Summary      : {summary_path}")
    print("="*60)


if __name__ == "__main__":
    main()
