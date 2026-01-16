"""
Node Functions for SIAP Compliance Detection Graph.

Each node represents a stage in the compliance detection pipeline:
1. Router Node - Classifies article to a scenario
2. Question Nodes - Answer compliance questions using ReAct agent with tools
3. Verdict Node - Determines final HIT/NO_HIT status

Uses langgraph.prebuilt.create_react_agent for the ReAct pattern.
This requires an LLM that supports tool binding (e.g., OpenAI GPT-4).
"""

from state import ComplianceState, Verdict, AnswerRecord
from tools import (
    extract_text_evidence,
    validate_threshold,
    classify_scenario,
    get_all_scenarios,
    set_current_article,
    get_react_tools
)

# ReAct agent from langgraph.prebuilt
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# Confidence threshold for YES/NO decisions
CONFIDENCE_THRESHOLD = 0.75


# =============================================================================
# ROUTER NODE
# =============================================================================

def router_node(state: ComplianceState, llm) -> ComplianceState:
    """
    Classification node that routes article to appropriate scenario.

    Analyzes article content and maps to one of the available scenarios
    (Cannabis, Art Dealing, Commodity Trading, etc.)
    """
    article = state["article_text"]
    scenarios = get_all_scenarios()

    # First try keyword-based classification
    classification = classify_scenario(article, scenarios)

    if classification["scenario_id"]:
        scenario_id = classification["scenario_id"]
        scenario = scenarios[scenario_id]

        state["scenario_id"] = scenario_id
        state["scenario_name"] = scenario.get("name")
        state["decision_tree"] = scenario.get("questions", {})
        state["current_node"] = scenario.get("start", "Q1")

        print(f"[ROUTER] Classified as: {scenario.get('name')}")
        print(f"[ROUTER] Confidence: {classification['confidence']}")
        print(f"[ROUTER] Starting at: {state['current_node']}")
    else:
        # Use LLM for classification if keywords don't match
        scenario_list = "\n".join([
            f"- {sid}: {s.get('name')} - {s.get('description', '')[:100]}"
            for sid, s in scenarios.items()
        ])

        prompt = f"""Classify this article into one of the compliance scenarios.

AVAILABLE SCENARIOS:
{scenario_list}

ARTICLE:
{article[:2000]}

Respond with ONLY the scenario_id (e.g., "cannabis_business") or "none" if no match.

SCENARIO:"""

        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        response_text = response_text.strip().lower()

        # Extract scenario ID from response
        for sid in scenarios.keys():
            if sid in response_text:
                scenario = scenarios[sid]
                state["scenario_id"] = sid
                state["scenario_name"] = scenario.get("name")
                state["decision_tree"] = scenario.get("questions", {})
                state["current_node"] = scenario.get("start", "Q1")
                print(f"[ROUTER] LLM classified as: {scenario.get('name')}")
                break
        else:
            # No scenario matched
            state["scenario_id"] = None
            state["final_verdict"] = Verdict.NO_HIT
            state["verdict_reason"] = "No matching compliance scenario identified"
            print("[ROUTER] No scenario match - auto NO_HIT")

    return state


# =============================================================================
# REACT AGENT RUNNER (Using create_react_agent)
# =============================================================================

def run_react_agent(llm, question: str, article: str, max_iterations: int = 5) -> dict:
    """
    Run ReAct agent using langgraph.prebuilt.create_react_agent.

    This uses native tool binding which is supported by OpenAI models.

    Args:
        llm: Language model with tool binding support (e.g., ChatOpenAI)
        question: The compliance question to answer
        article: The article text to analyze
        max_iterations: Maximum reasoning iterations

    Returns:
        dict with: answer (bool), reason (str), confidence (float)
    """
    # Set the article for tools to access
    set_current_article(article)

    # Get tools for ReAct agent
    tools = get_react_tools()

    # Create ReAct agent with tools
    react_agent = create_react_agent(llm, tools)

    # Create the prompt for the agent
    system_prompt = f"""You are a compliance analyst. Your task is to answer a YES/NO question about an article.

QUESTION: {question}

ARTICLE (first 2000 chars):
{article[:2000]}

INSTRUCTIONS:
1. Use the search_evidence tool to find relevant quotes from the article
2. Use the check_threshold tool if you need to verify numerical values (percentages, amounts)
3. Based on the evidence, provide a final answer: YES or NO
4. You must be confident (>75%) to answer YES. If uncertain, answer NO.

After gathering evidence, respond with your final determination in this format:
FINAL ANSWER: YES or NO
REASON: Brief explanation with evidence
CONFIDENCE: A number between 0 and 1"""

    # Run the agent
    try:
        result = react_agent.invoke({
            "messages": [HumanMessage(content=system_prompt)]
        })

        # Extract the final message
        final_message = result["messages"][-1]
        response_text = final_message.content if hasattr(final_message, 'content') else str(final_message)

        print(f"[REACT] Agent response: {response_text[:200]}...")

        # Parse the response
        answer, reason, confidence = parse_react_response(response_text)

        # Apply confidence threshold
        if confidence <= CONFIDENCE_THRESHOLD:
            print(f"[REACT] Low confidence ({confidence:.2f}) - defaulting to NO")
            answer = False
            reason = f"Insufficient confidence ({confidence:.2f} <= {CONFIDENCE_THRESHOLD}): {reason}"

        return {
            "answer": answer,
            "reason": reason,
            "confidence": confidence
        }

    except Exception as e:
        print(f"[REACT] Agent error: {e}")
        # Fallback to direct evidence search
        return fallback_evidence_search(question, article)


def parse_react_response(response: str) -> tuple[bool, str, float]:
    """
    Parse the ReAct agent's response to extract answer, reason, and confidence.

    Returns:
        tuple of (answer: bool, reason: str, confidence: float)
    """
    import re

    response_upper = response.upper()

    # Extract answer
    answer = False
    if "FINAL ANSWER: YES" in response_upper or "ANSWER: YES" in response_upper:
        answer = True
    elif "FINAL ANSWER: NO" in response_upper or "ANSWER: NO" in response_upper:
        answer = False
    elif "YES" in response_upper and "NO" not in response_upper:
        answer = True

    # Extract reason
    reason_match = re.search(r"REASON:?\s*(.+?)(?:CONFIDENCE|$)", response, re.IGNORECASE | re.DOTALL)
    reason = reason_match.group(1).strip()[:300] if reason_match else response[:300]

    # Extract confidence
    confidence = 0.7  # Default
    confidence_match = re.search(r"CONFIDENCE:?\s*([\d.]+)", response, re.IGNORECASE)
    if confidence_match:
        try:
            confidence = float(confidence_match.group(1))
            if confidence > 1:
                confidence = confidence / 100  # Convert percentage to decimal
        except ValueError:
            pass

    # Boost confidence if evidence was found
    if "EVIDENCE FOUND" in response.upper():
        confidence = max(confidence, 0.8)

    return answer, reason, confidence


def fallback_evidence_search(question: str, article: str) -> dict:
    """
    Fallback when ReAct agent fails - use direct evidence extraction.
    """
    print("[REACT] Falling back to direct evidence search...")
    evidence_result = extract_text_evidence(question, article)

    if evidence_result["found"] and evidence_result["confidence"] > CONFIDENCE_THRESHOLD:
        return {
            "answer": True,
            "reason": evidence_result["evidence"][0][:300] if evidence_result["evidence"] else "Evidence found",
            "confidence": evidence_result["confidence"]
        }
    else:
        return {
            "answer": False,
            "reason": f"Insufficient evidence (confidence: {evidence_result['confidence']:.2f})",
            "confidence": evidence_result["confidence"]
        }


# =============================================================================
# QUESTION NODE (with ReAct)
# =============================================================================

def question_node(state: ComplianceState, llm) -> ComplianceState:
    """
    Process the current question using ReAct pattern with create_react_agent.

    The ReAct agent:
    1. Thinks about what information is needed
    2. Calls tools (search_evidence, check_threshold)
    3. Observes results
    4. Repeats until ready to answer YES/NO
    """
    current_node = state["current_node"]
    decision_tree = state["decision_tree"]

    if not current_node or current_node not in decision_tree:
        return state

    question_data = decision_tree[current_node]
    question_text = question_data.get("text", "")
    article = state["article_text"]

    print(f"\n[{current_node}] {question_text}")
    print("[REACT] Starting ReAct agent with create_react_agent...")

    # Run ReAct agent
    result = run_react_agent(llm, question_text, article)

    answer = result["answer"]
    reason = result["reason"]
    confidence = result["confidence"]

    # Record answer
    state["answers"][current_node] = AnswerRecord(
        question=question_text,
        answer=answer,
        evidence=reason,
        confidence=confidence
    )

    answer_str = "YES" if answer else "NO"
    print(f"[{current_node}] Answer: {answer_str} (confidence: {confidence:.2f})")

    # Determine next node (SKIP LOGIC via decision tree)
    next_node = question_data.get("next_if_yes") if answer else question_data.get("next_if_no")

    if next_node is None:
        state["current_node"] = ""
        print(f"[{current_node}] Terminal node reached")
    else:
        state["current_node"] = next_node
        print(f"[{current_node}] Next: {next_node}")

    return state


# =============================================================================
# VERDICT NODE
# =============================================================================

def verdict_node(state: ComplianceState) -> ComplianceState:
    """
    Final scoring node that determines HIT or NO_HIT verdict.

    Analyzes all answers and applies risk scoring to reach
    autonomous determination.
    """
    answers = state["answers"]

    print("\n" + "=" * 50)
    print("VERDICT DETERMINATION")
    print("=" * 50)

    # Calculate risk indicators
    hit_reasons = []
    risk_score = 0.0

    # Check for high-risk answers
    for node_id, answer_record in answers.items():
        question_lower = answer_record["question"].lower()

        # Red flag indicators
        red_flags = ["illegal", "more than 10%", "more than 25%",
                     "executive", "board of directors", "direct"]

        if answer_record["answer"]:  # YES answer
            # Check if this is a red flag question
            is_red_flag = any(flag in question_lower for flag in red_flags)

            if is_red_flag:
                hit_reasons.append(f"{node_id}: {answer_record['question'][:60]}...")
                risk_score += 0.3
            else:
                risk_score += 0.1

    # Normalize risk score
    risk_score = min(risk_score, 1.0)

    # Determine verdict
    if hit_reasons:
        state["final_verdict"] = Verdict.HIT
        state["verdict_reason"] = "; ".join(hit_reasons)
        print(f"[VERDICT] HIT - Risk triggers found")
        for reason in hit_reasons:
            print(f"  - {reason}")
    else:
        state["final_verdict"] = Verdict.NO_HIT
        state["verdict_reason"] = "No compliance risks identified"
        print(f"[VERDICT] NO_HIT - Clean")

    state["risk_score"] = risk_score
    print(f"[VERDICT] Risk Score: {risk_score:.2f}")

    return state


# =============================================================================
# ROUTING FUNCTIONS (For Conditional Edges)
# =============================================================================

def should_continue_questions(state: ComplianceState) -> str:
    """
    Determines if we should continue asking questions or move to verdict.

    Returns:
        "question" - Continue to next question
        "verdict" - Move to verdict determination
        "end" - Early termination (no scenario matched)
    """
    # Check if scenario was matched
    if state["scenario_id"] is None:
        return "end"

    # Check if we've reached a terminal node (empty current_node)
    if not state["current_node"]:
        return "verdict"

    # Continue to next question
    return "question"


def route_after_router(state: ComplianceState) -> str:
    """Route after classification."""
    if state["scenario_id"] is None:
        return "end"
    return "question"
