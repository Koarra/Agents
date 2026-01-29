"""
State Definitions for SIAP Compliance Detection.

Uses TypedDict for type-safe state management in LangGraph.
"""

from typing import TypedDict, Optional, List
from enum import Enum


class Verdict(Enum):
    """Final compliance verdict."""
    HIT = "HIT"              # Risk detected with high confidence
    NO_HIT = "NO_HIT"        # Clean, no risks found
    PENDING = "PENDING"      # Processing not complete yet
    MISSING_INFO = "MISSING_INFO"  # Insufficient information to determine


class ConfidenceLevel(Enum):
    """Confidence classification for answers."""
    HIGH = "HIGH"        # >75% - answer is reliable
    MEDIUM = "MEDIUM"    # 50-75% - answer flagged as uncertain
    LOW = "LOW"          # <50% - cannot determine, treated as missing


class AnswerRecord(TypedDict):
    """Record of a single question's answer."""
    question: str
    answer: bool  # True = YES, False = NO
    evidence: str
    confidence: float
    confidence_level: str  # HIGH, MEDIUM, LOW
    is_missing: bool  # True if confidence too low to determine


class MissingFieldRecord(TypedDict):
    """Record of a field that couldn't be determined."""
    question_id: str
    question_text: str
    confidence: float
    suggested_documents: str  # What documents could help answer this


class ComplianceState(TypedDict):
    """
    State for compliance detection workflow.

    Tracks article processing through decision tree traversal.
    """
    # Article Information
    article_id: str
    article_text: str

    # Scenario Classification
    scenario_id: Optional[str]
    scenario_name: Optional[str]

    # Decision Tree State
    current_node: str  # Current question node (Q1, Q2, etc.)
    decision_tree: dict  # {Q1: {text, next_if_yes, next_if_no}, ...}

    # Answers Collected
    answers: dict[str, AnswerRecord]

    # Missing Information Tracking
    missing_fields: list[MissingFieldRecord]  # Questions that couldn't be answered
    uncertain_answers: list[str]  # Question IDs with MEDIUM confidence (flagged)
    early_termination: bool  # True if stopped due to LOW confidence answer

    # Final Verdict
    final_verdict: Verdict
    verdict_reason: str
    risk_score: float
    recommended_action: str  # What to do next (e.g., "Request income statements")


def create_initial_state(article_id: str, article_text: str) -> ComplianceState:
    """Create initial state for article processing."""
    return ComplianceState(
        article_id=article_id,
        article_text=article_text,
        scenario_id=None,
        scenario_name=None,
        current_node="",
        decision_tree={},
        answers={},
        missing_fields=[],
        uncertain_answers=[],
        early_termination=False,
        final_verdict=Verdict.PENDING,
        verdict_reason="",
        risk_score=0.0,
        recommended_action=""
    )
