"""
State Management for SIAP Compliance Detection.

Defines ComplianceState TypedDict that tracks the article through
the decision tree to reach a HIT or NO_HIT verdict.
"""

from typing import TypedDict, Optional, Literal
from enum import Enum


class Verdict(str, Enum):
    """Final compliance verdict."""
    PENDING = "PENDING"
    HIT = "HIT"
    NO_HIT = "NO_HIT"


class AnswerRecord(TypedDict):
    """Record of an answered question."""
    question: str
    answer: bool  # True = YES, False = NO
    evidence: str
    confidence: float


class ComplianceState(TypedDict):
    """
    Main state for compliance detection workflow.

    Tracks the article through the decision tree until reaching
    a terminal state (HIT or NO_HIT) autonomously.
    """

    # Input
    article_id: str
    article_text: str

    # Routing
    scenario_id: Optional[str]  # e.g., "cannabis_business", "art_dealing"
    scenario_name: Optional[str]

    # Decision Tree Traversal
    current_node: str  # Current question node (Q1, Q2, etc.)
    decision_tree: dict  # {Q1: {text, next_if_yes, next_if_no}, ...}

    # Evidence Collection
    answers: dict[str, AnswerRecord]  # {node_id: AnswerRecord, ...}

    # Final Output
    final_verdict: Verdict
    verdict_reason: str
    risk_score: float


def create_initial_state(article_id: str, article_text: str) -> ComplianceState:
    """Create initial state for processing an article."""
    return ComplianceState(
        # Input
        article_id=article_id,
        article_text=article_text,

        # Routing (to be set by classifier)
        scenario_id=None,
        scenario_name=None,

        # Tree traversal
        current_node="",
        decision_tree={},

        # Evidence
        answers={},

        # Output
        final_verdict=Verdict.PENDING,
        verdict_reason="",
        risk_score=0.0
    )
