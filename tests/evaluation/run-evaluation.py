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
import time
from datetime import datetime
from pathlib import Path

# Add project root and eval dir to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from tqdm import tqdm

from eval_report_builder import build_report, compute_latency_stats

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).parent / "eval-dataset.json"
ROBUSTNESS_DATASET_PATH = Path(__file__).parent / "eval-dataset-robustness.json"
RESULTS_DIR = Path(__file__).parent / "results"


def load_dataset(sample: int = 0, offset: int = 0, include_robustness: bool = False) -> list[dict]:
    with open(DATASET_PATH) as f:
        data = json.load(f)
    if include_robustness and ROBUSTNESS_DATASET_PATH.exists():
        with open(ROBUSTNESS_DATASET_PATH) as f:
            data += json.load(f)
        logger.info("Merged robustness dataset (%d total questions)", len(data))
    if offset:
        data = data[offset:]
    if sample and sample < len(data):
        logger.info("Using sample of %d questions (offset=%d)", sample, offset)
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
        t0 = time.perf_counter()
        try:
            result = pipeline.query(item["question"], top_k=10)
            # Extract contexts from sources returned by pipeline
            contexts = [r["content"] for r in retriever.hybrid_search(
                item["question"], k=10, reranker=reranker
            )]
            latency_ms = (time.perf_counter() - t0) * 1000
            samples.append({
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "reference_contexts": item.get("reference_contexts", []),
                "answer": result["answer"],
                "contexts": contexts,
                "category": item.get("category", ""),
                "difficulty": item.get("difficulty", ""),
                "eval_type": item.get("eval_type", "standard"),
                "latency_ms": round(latency_ms, 2),
            })
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.warning("RAG query failed for '%s': %s", item["question"][:50], exc)
            samples.append({
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "reference_contexts": item.get("reference_contexts", []),
                "answer": "",
                "contexts": [],
                "category": item.get("category", ""),
                "difficulty": item.get("difficulty", ""),
                "eval_type": item.get("eval_type", "standard"),
                "latency_ms": round(latency_ms, 2),
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



def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG chatbot quality")
    parser.add_argument("--sample", type=int, default=0, help="Use N questions (0 = all)")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N questions")
    parser.add_argument("--output", default=str(RESULTS_DIR), help="Output directory")
    parser.add_argument("--skip-rag", action="store_true", help="Load existing RAG outputs (debug)")
    parser.add_argument("--robustness", action="store_true", help="Include robustness dataset")
    args = parser.parse_args()

    output_dir = Path(args.output)

    # Step 1: Load dataset
    questions = load_dataset(sample=args.sample, offset=args.offset, include_robustness=args.robustness)
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
    latency_stats = compute_latency_stats(eval_result["samples"])
    eval_result["latency"] = latency_stats
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

    rr = report.get("retrieval_rank_metrics", {})
    if rr:
        k = rr.get("k", 10)
        print(f"\nRetrieval Rank Metrics (k={k}):")
        print(f"  MAP@{k}  : {rr.get(f'map_at_{k}', 0):.4f}")
        print(f"  MRR     : {rr.get('mrr', 0):.4f}")
        print(f"  Hit@{k}  : {rr.get(f'hit_at_{k}', 0):.4f}")

    print(f"\nRaw results  : {raw_path}")
    print(f"Summary      : {summary_path}")
    print("="*60)


if __name__ == "__main__":
    main()
