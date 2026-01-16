"""
State Definitions for SIAP Compliance Detection.

Uses TypedDict for type-safe state management in LangGraph.
"""

from typing import TypedDict, Optional
from enum import Enum


class Verdict(Enum):
    """Final compliance verdict."""
    HIT = "HIT"
    NO_HIT = "NO_HIT"
    PENDING = "PENDING"


class AnswerRecord(TypedDict):
    """Record of a single question's answer."""
    question: str
    answer: bool  # True = YES, False = NO
    evidence: str
    confidence: float


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

    # Final Verdict
    final_verdict: Verdict
    verdict_reason: str
    risk_score: float


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
        final_verdict=Verdict.PENDING,
        verdict_reason="",
        risk_score=0.0
    )
