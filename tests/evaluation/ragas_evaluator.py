"""LLM-as-judge evaluator implementing RAGAS-equivalent metrics via Gemini.

Metrics computed:
  - faithfulness:       answer only uses info from contexts (0-1)
  - answer_relevancy:   answer addresses the question (0-1)
  - context_precision:  fraction of retrieved contexts that are relevant (0-1)
  - context_recall:     contexts cover enough to produce ground truth (0-1)
"""

import json
import logging
import os
import re
import time
from typing import Dict, List

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
logger = logging.getLogger(__name__)

_JUDGE_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
_RETRY_DELAY = 2  # seconds between retries on rate limit


def _make_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    return genai.Client(api_key=api_key)


def _score_prompt(client: genai.Client, prompt: str, retries: int = 3) -> float:
    """Call Gemini and parse a float score 0.0-1.0 from the response."""
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=_JUDGE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=64,
                ),
            )
            text = resp.text.strip()
            matches = re.findall(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", text)
            if matches:
                return round(float(matches[0]), 4)
            logger.warning("Could not parse score from: %s", text[:100])
            return 0.5
        except Exception as exc:
            if attempt < retries - 1:
                logger.warning("Judge call failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(_RETRY_DELAY * (attempt + 1))
            else:
                logger.error("Judge call failed after %d retries: %s", retries, exc)
                return 0.0


def score_faithfulness(client: genai.Client, question: str, answer: str, contexts: List[str]) -> float:
    """Does the answer only contain claims supported by the contexts?"""
    ctx_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    prompt = f"""You are a strict fact-checker evaluating a RAG system answer.

CONTEXTS:
{ctx_text}

QUESTION: {question}

ANSWER: {answer}

Rate FAITHFULNESS: what fraction of claims in the ANSWER are directly supported by the CONTEXTS?
- Score 1.0: every claim in answer is supported by contexts
- Score 0.5: about half the claims are supported
- Score 0.0: answer contains information not found in contexts at all

Reply with only a single decimal number between 0.0 and 1.0."""
    return _score_prompt(client, prompt)


def score_answer_relevancy(client: genai.Client, question: str, answer: str) -> float:
    """Does the answer actually address what was asked?"""
    prompt = f"""You are evaluating a RAG system answer for relevance.

QUESTION: {question}

ANSWER: {answer}

Rate ANSWER RELEVANCY: how well does the answer address the question?
- Score 1.0: answer directly and completely addresses the question
- Score 0.5: answer partially addresses the question or goes off-topic
- Score 0.0: answer is completely irrelevant to the question

Reply with only a single decimal number between 0.0 and 1.0."""
    return _score_prompt(client, prompt)


def score_context_precision(client: genai.Client, question: str, contexts: List[str], ground_truth: str) -> float:
    """What fraction of retrieved contexts are actually useful for answering?"""
    ctx_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    prompt = f"""You are evaluating the precision of retrieved contexts for a RAG system.

QUESTION: {question}
GROUND TRUTH ANSWER: {ground_truth}

RETRIEVED CONTEXTS:
{ctx_text}

Rate CONTEXT PRECISION: what fraction of the retrieved contexts are relevant and useful for answering the question?
- Score 1.0: all contexts are relevant
- Score 0.5: about half the contexts are relevant
- Score 0.0: none of the contexts are relevant

Reply with only a single decimal number between 0.0 and 1.0."""
    return _score_prompt(client, prompt)


def score_context_recall(client: genai.Client, contexts: List[str], ground_truth: str) -> float:
    """Do the contexts contain enough information to produce the ground truth answer?"""
    ctx_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    prompt = f"""You are evaluating whether retrieved contexts are sufficient to answer a question correctly.

GROUND TRUTH ANSWER: {ground_truth}

RETRIEVED CONTEXTS:
{ctx_text}

Rate CONTEXT RECALL: to what extent can the ground truth answer be inferred from the retrieved contexts?
- Score 1.0: the contexts contain all the information needed to produce the ground truth answer
- Score 0.5: contexts contain some but not all necessary information
- Score 0.0: contexts do not contain the information needed for the ground truth answer

Reply with only a single decimal number between 0.0 and 1.0."""
    return _score_prompt(client, prompt)


class RagasEvaluator:
    """Evaluates RAG quality using LLM-as-judge (RAGAS-equivalent metrics)."""

    def __init__(self):
        self.client = _make_client()

    def evaluate_sample(self, sample: Dict) -> Dict:
        """Score a single sample. Returns dict with all 4 metric scores."""
        question = sample["question"]
        answer = sample.get("answer", "")
        contexts = sample.get("contexts", [])
        ground_truth = sample.get("ground_truth", "")

        if not answer:
            return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                    "context_precision": 0.0, "context_recall": 0.0}

        time.sleep(0.5)  # avoid rate limit
        faithfulness = score_faithfulness(self.client, question, answer, contexts)
        time.sleep(0.5)
        relevancy = score_answer_relevancy(self.client, question, answer)
        time.sleep(0.5)
        precision = score_context_precision(self.client, question, contexts, ground_truth)
        time.sleep(0.5)
        recall = score_context_recall(self.client, contexts, ground_truth)

        return {
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "context_precision": precision,
            "context_recall": recall,
        }

    def evaluate(self, samples: List[Dict]) -> Dict:
        """Evaluate a list of samples. Returns aggregate + per-sample scores."""
        metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

        if not samples:
            return {"aggregate": {m: 0.0 for m in metrics}, "samples": [], "judge_model": _JUDGE_MODEL}

        results = []
        for i, sample in enumerate(samples):
            logger.info("Evaluating sample %d/%d: %s", i + 1, len(samples), sample["question"][:60])
            scores = self.evaluate_sample(sample)
            results.append({**sample, "scores": scores})

        aggregate = {
            m: round(sum(r["scores"][m] for r in results) / len(results), 4)
            for m in metrics
        }

        return {"aggregate": aggregate, "samples": results, "judge_model": _JUDGE_MODEL}
