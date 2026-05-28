"""
utils/evaluation.py
──────────────────────────────────────────────────────────────────────────────
DeepEval-powered RAG evaluation for StudyPal.
Judge LLM priority (auto-detected from .env, no OpenAI needed):

  1. Gemini  — if GOOGLE_API_KEY is set  → uses DeepEval's native GeminiModel
  2. Groq    — if GROQ_API_KEY is set    → uses a DeepEvalBaseLLM wrapper
                                           around langchain_groq.ChatGroq

Set in .env:
  GOOGLE_API_KEY=AIza...               <- preferred (native DeepEval support)
  GEMINI_EVAL_MODEL=gemini-2.0-flash   <- optional, defaults to gemini-2.0-flash

  GROQ_API_KEY=gsk_...                 <- fallback
  GROQ_EVAL_MODEL=llama-3.1-8b-instant <- optional, defaults to llama-3.1-8b-instant

Runs 5 RAG metrics:
  1. AnswerRelevancy      - is the answer relevant to the question?
  2. Faithfulness         - is the answer grounded in the retrieved context?
  3. ContextualRelevancy  - are the retrieved chunks relevant to the question?
  4. ContextualPrecision  - are the most relevant chunks ranked highest?*
  5. ContextualRecall     - do the chunks cover the expected answer?*
  (* requires an expected/reference answer)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv
load_dotenv()

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

DEFAULT_THRESHOLD = 0.5


# -- Groq custom wrapper ------------------------------------------------------
def _build_groq_judge():
    """Return a DeepEvalBaseLLM-compatible Groq judge, or None on failure."""
    groq_key   = os.getenv("GROQ_API_KEY", "")
    groq_model = os.getenv("GROQ_EVAL_MODEL", "llama-3.3-70b-versatile")

    if not groq_key:
        return None

    try:
        from langchain_groq import ChatGroq
        from deepeval.models.base_model import DeepEvalBaseLLM

        class GroqJudge(DeepEvalBaseLLM):
            """Wraps ChatGroq so DeepEval can use Groq as the judge LLM."""

            def __init__(self):
                self._llm = ChatGroq(
                    api_key=groq_key,
                    model_name=groq_model,
                    temperature=0,
                )

            def load_model(self):
                return self._llm

            def generate(self, prompt: str) -> str:
                return self._llm.invoke(prompt).content

            async def a_generate(self, prompt: str) -> str:
                res = await self._llm.ainvoke(prompt)
                return res.content

            def get_model_name(self) -> str:
                return f"Groq/{groq_model}"

        return GroqJudge()

    except Exception:
        return None


# -- Judge model factory -------------------------------------------------------
def _get_judge():
    """
    Return the best available judge in priority order:
      1. Gemini (native DeepEval support, GOOGLE_API_KEY)
      2. Groq   (custom wrapper, GROQ_API_KEY)
    Raises RuntimeError if neither key is available.
    """
    google_key   = os.getenv("GOOGLE_API_KEY", "")
    gemini_model = os.getenv("GEMINI_EVAL_MODEL", "gemini-2.0-flash")

    if google_key:
        from deepeval.models import GeminiModel
        return GeminiModel(model=gemini_model, api_key=google_key)

    groq_judge = _build_groq_judge()
    if groq_judge:
        return groq_judge

    raise RuntimeError(
        "No judge LLM available.\n"
        "Add one of these to your .env file:\n"
        "  GOOGLE_API_KEY=AIza...    (recommended - Gemini)\n"
        "  GROQ_API_KEY=gsk_...      (fallback - Groq/Llama)\n"
    )


# -- Result dataclass ---------------------------------------------------------
@dataclass
class EvalResult:
    question: str
    answer: str
    judge_model: str           = ""
    scores: dict               = field(default_factory=dict)
    passed: dict               = field(default_factory=dict)
    reasons: dict              = field(default_factory=dict)
    errors: dict               = field(default_factory=dict)
    overall_pass: bool         = False
    skipped_metrics: List[str] = field(default_factory=list)


# -- Internal: run one metric safely ------------------------------------------
def _measure(metric, test_case: LLMTestCase, name: str, result: EvalResult) -> None:
    try:
        metric.measure(test_case)
        result.scores[name]  = round(metric.score, 3)
        result.passed[name]  = metric.is_successful()
        result.reasons[name] = getattr(metric, "reason", "") or ""
    except Exception as exc:
        result.errors[name]  = str(exc)
        result.scores[name]  = 0.0
        result.passed[name]  = False
        result.reasons[name] = f"Evaluation error: {exc}"


# -- Public API ---------------------------------------------------------------
def run_evaluation(
    question: str,
    answer: str,
    contexts: List[str],
    expected: str = "",
    threshold: float = DEFAULT_THRESHOLD,
) -> EvalResult:
    """
    Run DeepEval RAG metrics using Gemini or Groq as the judge LLM.

    Parameters
    ----------
    question  : user query (PII-scrubbed)
    answer    : LLM response
    contexts  : retrieved chunk texts  (result["context"] -> doc.page_content)
    expected  : optional reference answer - unlocks Contextual Precision + Recall
    threshold : pass/fail threshold (default 0.5)
    """
    judge  = _get_judge()
    result = EvalResult(
        question=question,
        answer=answer,
        judge_model=judge.get_model_name(),
    )

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=contexts if contexts else ["(no context retrieved)"],
        expected_output=expected if expected else None,
    )

    # 1. Answer Relevancy
    _measure(
        AnswerRelevancyMetric(threshold=threshold, model=judge),
        test_case, "Answer Relevancy", result,
    )

    # 2. Faithfulness
    _measure(
        FaithfulnessMetric(threshold=threshold, model=judge),
        test_case, "Faithfulness", result,
    )

    # 3. Contextual Relevancy
    _measure(
        ContextualRelevancyMetric(threshold=threshold, model=judge),
        test_case, "Contextual Relevancy", result,
    )

    # 4 & 5 -- only when a reference answer is provided
    if expected:
        _measure(
            ContextualPrecisionMetric(threshold=threshold, model=judge),
            test_case, "Contextual Precision", result,
        )
        _measure(
            ContextualRecallMetric(threshold=threshold, model=judge),
            test_case, "Contextual Recall", result,
        )
    else:
        result.skipped_metrics = ["Contextual Precision", "Contextual Recall"]

    result.overall_pass = all(result.passed.values()) if result.passed else False
    return result