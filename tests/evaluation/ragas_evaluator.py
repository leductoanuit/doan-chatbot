"""RagasEvaluator orchestrator — delegates scoring to generation/retrieval metric modules.

Metrics computed:
  Standard samples:  faithfulness, answer_relevancy, context_precision, context_recall, correctness
  eval_type=noise:   + noise_robustness
  eval_type=negative: negative_rejection only (skip faithfulness/correctness)
"""

import logging
import time
from typing import Dict, List, Optional

from ragas_base import _make_client, _JUDGE_MODEL
from ragas_generation_metrics import (
    score_faithfulness,
    score_answer_relevancy,
    score_correctness,
    score_noise_robustness,
    score_negative_rejection,
)
from ragas_retrieval_metrics import score_context_precision, score_context_recall

logger = logging.getLogger(__name__)

_STANDARD_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "correctness"]
_ALL_METRICS = _STANDARD_METRICS + ["noise_robustness", "negative_rejection"]


def _avg(values: List[Optional[float]]) -> float:
    valid = [v for v in values if v is not None]
    return round(sum(valid) / len(valid), 4) if valid else 0.0


class RagasEvaluator:
    """Evaluates RAG quality using LLM-as-judge (RAGAS-equivalent + Auepora metrics)."""

    def __init__(self):
        self.client = _make_client()

    def evaluate_sample(self, sample: Dict) -> Dict:
        """Score one sample. Branches on sample['eval_type'] for robustness/rejection."""
        question = sample["question"]
        answer = sample.get("answer", "")
        contexts = sample.get("contexts", [])
        ground_truth = sample.get("ground_truth", "")
        eval_type = sample.get("eval_type", "standard")

        null_scores: Dict[str, Optional[float]] = {m: None for m in _ALL_METRICS}

        if not answer:
            return {m: 0.0 for m in _STANDARD_METRICS}

        if eval_type == "negative":
            # Only measure rejection; skip generation/retrieval quality metrics
            time.sleep(0.5)
            rejection = score_negative_rejection(self.client, question, answer)
            return {**null_scores, "negative_rejection": rejection}

        # Standard + noise path
        time.sleep(0.5)
        faith = score_faithfulness(self.client, question, answer, contexts)
        time.sleep(0.5)
        relevancy = score_answer_relevancy(self.client, question, answer)
        time.sleep(0.5)
        precision = score_context_precision(self.client, question, contexts, ground_truth)
        time.sleep(0.5)
        recall = score_context_recall(self.client, contexts, ground_truth)
        time.sleep(0.5)
        correctness = score_correctness(self.client, question, answer, ground_truth)

        scores: Dict[str, Optional[float]] = {
            **null_scores,
            "faithfulness": faith,
            "answer_relevancy": relevancy,
            "context_precision": precision,
            "context_recall": recall,
            "correctness": correctness,
        }

        if eval_type == "noise":
            time.sleep(0.5)
            noisy_contexts = sample.get("noisy_contexts", contexts)
            scores["noise_robustness"] = score_noise_robustness(
                self.client, question, answer, ground_truth, noisy_contexts
            )

        return scores

    def evaluate(self, samples: List[Dict]) -> Dict:
        """Evaluate a list of samples. Returns aggregate + per-sample scores."""
        if not samples:
            return {"aggregate": {m: 0.0 for m in _STANDARD_METRICS}, "samples": [], "judge_model": _JUDGE_MODEL}

        results = []
        for i, sample in enumerate(samples):
            logger.info("Evaluating sample %d/%d: %s", i + 1, len(samples), sample["question"][:60])
            scores = self.evaluate_sample(sample)
            results.append({**sample, "scores": scores})

        aggregate = {
            m: _avg([r["scores"].get(m) for r in results])
            for m in _ALL_METRICS
        }
        # Drop metrics that were never computed (all None)
        aggregate = {m: v for m, v in aggregate.items() if any(r["scores"].get(m) is not None for r in results)}

        return {"aggregate": aggregate, "samples": results, "judge_model": _JUDGE_MODEL}
