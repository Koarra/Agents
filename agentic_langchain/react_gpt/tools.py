"""
ReAct Tool Functions for SIAP Compliance Detection.

These tools are designed to work with create_react_agent from langgraph.prebuilt.
OpenAI models support native tool binding via bind_tools().
"""

import re
import os
import json
from glob import glob
from typing import Optional
from langchain_core.tools import tool


# =============================================================================
# CORE TOOL FUNCTIONS (Plain Python)
# =============================================================================

def extract_text_evidence(query: str, text: str, context_chars: int = 300) -> dict:
    """
    Extract supporting quotes from text that answer a compliance question.

    Args:
        query: The compliance question or search terms
        text: The article text to search
        context_chars: Characters of context around matches

    Returns:
        dict with: found (bool), evidence (list of quotes), confidence (float)
    """
    # Extract key terms from query
    stop_words = {"is", "the", "a", "an", "does", "are", "there", "any", "have",
                  "has", "been", "to", "of", "in", "for", "with", "this", "that",
                  "client", "involved", "activity", "business"}
    query_terms = [w.lower() for w in re.findall(r'\w+', query.lower())
                   if w.lower() not in stop_words and len(w) > 2]

    evidence_snippets = []
    matched_terms = set()

    # Search for each term and extract context
    for term in query_terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        for match in pattern.finditer(text):
            start = max(0, match.start() - context_chars)
            end = min(len(text), match.end() + context_chars)
            snippet = text[start:end].strip()

            # Clean up snippet boundaries
            if start > 0:
                first_space = snippet.find(" ")
                if first_space > 0:
                    snippet = "..." + snippet[first_space + 1:]
            if end < len(text):
                last_space = snippet.rfind(" ")
                if last_space > 0:
                    snippet = snippet[:last_space] + "..."

            if snippet and snippet not in evidence_snippets:
                evidence_snippets.append(snippet)
                matched_terms.add(term)

    # Calculate confidence based on term coverage
    confidence = len(matched_terms) / len(query_terms) if query_terms else 0.0

    return {
        "found": len(evidence_snippets) > 0,
        "query": query,
        "evidence": evidence_snippets[:5],  # Top 5 snippets
        "matched_terms": list(matched_terms),
        "confidence": round(min(confidence, 1.0), 2)
    }


def validate_threshold(value: float, limit: float, comparison: str = "greater_than") -> dict:
    """
    Check if a numerical value exceeds a compliance threshold.

    This uses pure Python logic to avoid LLM hallucination on numbers.

    Args:
        value: The extracted numerical value (e.g., ownership percentage)
        limit: The compliance threshold (e.g., 25%)
        comparison: "greater_than", "less_than", "greater_or_equal", "less_or_equal"

    Returns:
        dict with: value, limit, exceeds_threshold (bool), explanation
    """
    comparisons = {
        "greater_than": (lambda v, l: v > l, ">"),
        "less_than": (lambda v, l: v < l, "<"),
        "greater_or_equal": (lambda v, l: v >= l, ">="),
        "less_or_equal": (lambda v, l: v <= l, "<="),
    }

    if comparison not in comparisons:
        return {"error": f"Invalid comparison: {comparison}"}

    compare_fn, symbol = comparisons[comparison]
    exceeds = compare_fn(value, limit)

    return {
        "value": value,
        "limit": limit,
        "comparison": comparison,
        "exceeds_threshold": exceeds,
        "explanation": f"{value} {symbol} {limit} = {exceeds}"
    }


def classify_scenario(text: str, scenarios: dict) -> dict:
    """
    Classify text into one of the available scenarios based on keywords.

    Args:
        text: The article text
        scenarios: Dict of {scenario_id: {keywords: [...], name: "..."}}

    Returns:
        dict with: scenario_id, scenario_name, confidence, matched_keywords
    """
    text_lower = text.lower()
    best_match = None
    best_score = 0
    best_keywords = []

    for scenario_id, scenario_data in scenarios.items():
        keywords = scenario_data.get("keywords", [])
        matched = [kw for kw in keywords if kw.lower() in text_lower]
        score = len(matched)

        if score > best_score:
            best_score = score
            best_match = scenario_id
            best_keywords = matched

    if best_match:
        scenario_data = scenarios[best_match]
        confidence = min(best_score / 3, 1.0)  # 3+ keywords = full confidence
        return {
            "scenario_id": best_match,
            "scenario_name": scenario_data.get("name", best_match),
            "confidence": round(confidence, 2),
            "matched_keywords": best_keywords
        }

    return {
        "scenario_id": None,
        "scenario_name": None,
        "confidence": 0.0,
        "matched_keywords": []
    }


# =============================================================================
# SCENARIO LOADER
# =============================================================================

def load_scenarios_from_directory(scenarios_dir: str = None) -> dict:
    """
    Load scenario definitions from scenarios_1 directory.

    Returns:
        Dict mapping scenario_id to scenario data
    """
    if scenarios_dir is None:
        scenarios_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scenarios_1"
        )

    scenarios = {}
    json_files = glob(os.path.join(scenarios_dir, "*.json"))

    for json_file in sorted(json_files):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        scenario_id = os.path.basename(json_file).replace(".json", "")
        data["id"] = scenario_id

        # Extract keywords from name and description
        keywords = _extract_keywords(data)
        data["keywords"] = keywords

        scenarios[scenario_id] = data

    return scenarios


def _extract_keywords(scenario: dict) -> list:
    """Extract classification keywords from scenario."""
    text = f"{scenario.get('name', '')} {scenario.get('description', '')}".lower()

    keyword_patterns = {
        "cannabis": ["cannabis", "marijuana", "hemp", "thc", "cbd", "dispensary"],
        "art": ["art", "antique", "antiquity", "auction", "gallery", "fine art", "artefact"],
        "commodity": ["commodity", "trading", "energy", "metals", "agricultural", "oil", "gas"],
    }

    keywords = []
    for patterns in keyword_patterns.values():
        for pattern in patterns:
            if pattern in text:
                keywords.append(pattern)

    # Add words from the name
    name_words = scenario.get("name", "").lower().split()
    keywords.extend([w for w in name_words if len(w) > 3])

    return list(set(keywords))


def get_scenario(scenario_id: str) -> Optional[dict]:
    """Get a specific scenario by ID."""
    scenarios = load_scenarios_from_directory()
    return scenarios.get(scenario_id)


def get_all_scenarios() -> dict:
    """Get all available scenarios."""
    return load_scenarios_from_directory()


# =============================================================================
# REACT TOOLS FOR create_react_agent (OpenAI Compatible)
# =============================================================================

# Global variable to hold article text for tools
_current_article = ""


def set_current_article(article: str):
    """Set the current article for tools to access."""
    global _current_article
    _current_article = article


def get_current_article() -> str:
    """Get the current article."""
    global _current_article
    return _current_article


@tool
def search_evidence(query: str) -> str:
    """
    Search the article for evidence related to a compliance question.
    Use this to find specific quotes that support or refute a claim.

    Args:
        query: Keywords or question to search for in the article
    """
    global _current_article
    result = extract_text_evidence(query, _current_article)
    if result["found"]:
        snippets = "\n".join([f"- {e[:300]}" for e in result["evidence"][:3]])
        return f"EVIDENCE FOUND (confidence: {result['confidence']}):\n{snippets}"
    return "NO EVIDENCE FOUND for this query."


@tool("check_threshold", parse_docstring=True)
def check_threshold(value: float, threshold: float) -> str:
    """Check if a numerical value exceeds a compliance threshold. Use for ownership percentages, income thresholds, etc.

    Args:
        value: The percentage to check (e.g., 60.0 for 60%)
        threshold: The limit to compare against (e.g., 10.0 for 10%)
    """
    result = validate_threshold(value, threshold, "greater_than")
    if result["exceeds_threshold"]:
        return f"EXCEEDS THRESHOLD: {value}% > {threshold}%"
    return f"WITHIN THRESHOLD: {value}% <= {threshold}%"


# List of tools for create_react_agent
REACT_TOOLS = [search_evidence, check_threshold]


def get_react_tools():
    """Get list of tools for create_react_agent."""
    return REACT_TOOLS
