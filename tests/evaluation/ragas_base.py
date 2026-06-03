"""Shared infrastructure for LLM-as-judge evaluation: client factory and scored prompt helper."""

import logging
import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
logger = logging.getLogger(__name__)

_JUDGE_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_RETRY_DELAY = 2


def _make_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    return genai.Client(api_key=api_key)


def _score_prompt(client: genai.Client, prompt: str, retries: int = 3) -> float:
    """Call Gemini and parse a float score 0.0–1.0 from the response."""
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=_JUDGE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=64),
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
