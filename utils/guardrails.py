"""
utils/guardrails.py
──────────────────────────────────────────────────────────────────────────────
Two-layer guardrail pipeline powered by Microsoft Presidio:

  Layer 1 — PII Scrubbing (Presidio)
    Detects and redacts sensitive personal information from user input
    BEFORE it is sent to the LLM or stored in chat history.
    Entities covered: PERSON, EMAIL_ADDRESS, PHONE_NUMBER, LOCATION,
    CREDIT_CARD, IBAN_CODE, US_SSN, US_PASSPORT, IP_ADDRESS, URL, NRP

  Layer 2 — Off-topic / Prompt-injection Guard
    Blocks queries that are clearly not study-related (e.g. requests to
    generate harmful content, jailbreak attempts, or completely unrelated
    tasks) using a fast keyword + pattern check — no extra API call needed.

Usage
-----
    from utils.guardrails import GuardrailResult, run_guardrails

    result: GuardrailResult = run_guardrails(user_text)

    if not result.allowed:
        st.warning(result.reason)      # show the block reason
    else:
        query = result.clean_text      # PII-scrubbed, safe to send to LLM
        if result.pii_found:
            st.info(result.pii_summary)  # optional: tell user what was masked
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


# ── Presidio engine singletons (expensive to init, so cache them) ─────────────
_analyzer: AnalyzerEngine | None = None
_anonymizer: AnonymizerEngine | None = None

# PII entity types to detect and redact
PII_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "LOCATION",
    "CREDIT_CARD",
    "IBAN_CODE",
    "US_SSN",
    "US_PASSPORT",
    "IP_ADDRESS",
    "URL",
    "NRP",          # Nationality, Religious or Political group
    "MEDICAL_LICENSE",
    "US_BANK_NUMBER",
    "US_DRIVER_LICENSE",
]

# Off-topic / jailbreak patterns (case-insensitive)
# These are checked BEFORE the PII scan to fail fast.
_BLOCKED_PATTERNS: List[str] = [
    # Jailbreak / prompt injection
    r"ignore (previous|all|above|prior) instructions?",
    r"forget (everything|all|your|prior)",
    r"you are now",
    r"act as (if you are|a|an)",
    r"pretend (you are|to be)",
    r"jailbreak",
    r"DAN mode",
    r"override (safety|guidelines|restrictions)",
    # Harmful content
    r"\b(bomb|explosive|weapon|hack|malware|ransomware|phishing)\b",
    r"how to (kill|hurt|harm|attack|poison)",
    r"generate (fake|forged) (id|passport|document)",
    # Completely off-topic intent (not study-related at all)
    r"write me? (a |an )?(poem|song|lyrics|essay) about (love|dating|sex)",
    r"(buy|sell|invest|trade) (stocks?|crypto|bitcoin|shares)",
    r"(book|reserve) (a |)(flight|hotel|restaurant|table)",
    r"what('s| is) the weather",
    r"tell me a (dirty|adult|sexual|inappropriate) joke",
]

_COMPILED_BLOCKS = [re.compile(p, re.IGNORECASE) for p in _BLOCKED_PATTERNS]

# Minimum meaningful query length
MIN_QUERY_LEN = 3


# ── Data class returned to the caller ─────────────────────────────────────────
@dataclass
class GuardrailResult:
    allowed: bool                        # False → block the request
    clean_text: str                      # PII-scrubbed text (use this for LLM)
    original_text: str                   # original input (for logging only)
    pii_found: bool = False              # True if at least one PII entity was masked
    pii_summary: str = ""                # human-readable summary of what was masked
    reason: str = ""                     # populated when allowed=False
    detected_entities: List[str] = field(default_factory=list)


# ── Internal helpers ──────────────────────────────────────────────────────────
def _get_engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    global _analyzer, _anonymizer
    if _analyzer is None:
        _analyzer = AnalyzerEngine()
    if _anonymizer is None:
        _anonymizer = AnonymizerEngine()
    return _analyzer, _anonymizer


def _redact_pii(text: str) -> GuardrailResult:
    """Run Presidio analyzer + anonymizer on text; return scrubbed result."""
    analyzer, anonymizer = _get_engines()

    analysis = analyzer.analyze(
        text=text,
        entities=PII_ENTITIES,
        language="en",
    )

    if not analysis:
        return GuardrailResult(
            allowed=True,
            clean_text=text,
            original_text=text,
        )

    # Replace every detected PII entity with a <ENTITY_TYPE> placeholder
    operators = {
        entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
        for entity in PII_ENTITIES
    }

    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=analysis,
        operators=operators,
    )

    detected = list({r.entity_type for r in analysis})
    friendly_names = {
        "PERSON": "name",
        "EMAIL_ADDRESS": "email address",
        "PHONE_NUMBER": "phone number",
        "LOCATION": "location",
        "CREDIT_CARD": "credit card number",
        "IBAN_CODE": "bank account (IBAN)",
        "US_SSN": "Social Security Number",
        "US_PASSPORT": "passport number",
        "IP_ADDRESS": "IP address",
        "URL": "URL/link",
        "NRP": "nationality/political/religious group reference",
        "MEDICAL_LICENSE": "medical license number",
        "US_BANK_NUMBER": "bank account number",
        "US_DRIVER_LICENSE": "driver's license number",
    }
    masked_list = [friendly_names.get(e, e.lower()) for e in detected]
    summary = (
        f"🛡 For your privacy, the following were masked before sending to the AI: "
        + ", ".join(masked_list) + "."
    )

    return GuardrailResult(
        allowed=True,
        clean_text=anonymized.text,
        original_text=text,
        pii_found=True,
        pii_summary=summary,
        detected_entities=detected,
    )


def _check_blocked_patterns(text: str) -> str | None:
    """Return a block reason string if text matches any blocked pattern, else None."""
    for pattern in _COMPILED_BLOCKS:
        if pattern.search(text):
            matched = pattern.pattern
            if any(k in matched for k in ["ignore", "forget", "jailbreak", "DAN", "override", "act as", "pretend", "you are now"]):
                return "⚠️ This looks like a prompt injection or jailbreak attempt. Please ask a genuine study question."
            if any(k in matched for k in ["bomb", "weapon", "hack", "malware", "kill", "hurt", "harm", "poison", "forged"]):
                return "⚠️ This question contains content that can't be processed. Please keep questions study-related."
            return "⚠️ This question seems off-topic. StudyPal is designed for academic study questions only."
    return None


# ── Public API ─────────────────────────────────────────────────────────────────
def run_guardrails(text: str) -> GuardrailResult:
    """
    Run both guardrail layers on the user's input.

    Parameters
    ----------
    text : str
        Raw user input from the Streamlit text box.

    Returns
    -------
    GuardrailResult
        .allowed      – whether the query should proceed
        .clean_text   – PII-scrubbed text to pass to the LLM
        .pii_found    – True if PII was detected and masked
        .pii_summary  – friendly message describing what was masked
        .reason       – block reason (populated when allowed=False)
    """
    stripped = text.strip()

    # ── Guard: empty / too short ──────────────────────────────────────────────
    if len(stripped) < MIN_QUERY_LEN:
        return GuardrailResult(
            allowed=False,
            clean_text=stripped,
            original_text=text,
            reason="⚠️ Query is too short. Please type a meaningful question.",
        )

    # ── Guard: off-topic / injection patterns ─────────────────────────────────
    block_reason = _check_blocked_patterns(stripped)
    if block_reason:
        return GuardrailResult(
            allowed=False,
            clean_text=stripped,
            original_text=text,
            reason=block_reason,
        )

    # ── Guard: PII scrubbing via Presidio ─────────────────────────────────────
    result = _redact_pii(stripped)
    return result