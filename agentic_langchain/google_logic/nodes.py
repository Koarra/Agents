"""
Node Functions for SIAP Compliance Detection Graph.

Each node represents a stage in the compliance detection pipeline:
1. Router Node - Classifies article to a scenario
2. Question Nodes - Answer compliance questions using ReAct pattern with tools
3. Verdict Node - Determines final HIT/NO_HIT status

ReAct Pattern: Reasoning + Acting
The LLM reasons about what tool to use, executes it, observes the result,
and continues until it has enough information to answer.
"""

import re
from state import ComplianceState, Verdict, AnswerRecord
from tools import (
    extract_text_evidence,
    validate_threshold,
    classify_scenario,
    get_all_scenarios
)


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

        response = llm.invoke(prompt).strip().lower()

        # Extract scenario ID from response
        for sid in scenarios.keys():
            if sid in response:
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
# REACT TOOLS
# =============================================================================

def tool_search_evidence(query: str, article: str) -> str:
    """Tool: Search article for evidence."""
    result = extract_text_evidence(query, article)
    if result["found"]:
        snippets = "\n".join([f"- {e[:300]}" for e in result["evidence"][:3]])
        return f"EVIDENCE FOUND (confidence: {result['confidence']}):\n{snippets}"
    return "NO EVIDENCE FOUND for this query."


def tool_check_threshold(value: float, threshold: float) -> str:
    """Tool: Check if value exceeds threshold."""
    result = validate_threshold(value, threshold, "greater_than")
    if result["exceeds_threshold"]:
        return f"EXCEEDS: {value}% > {threshold}%"
    return f"OK: {value}% <= {threshold}%"


# =============================================================================
# REACT LOOP (Manual Implementation)
# =============================================================================

def run_react_loop(llm, question: str, article: str, max_iterations: int = 3) -> dict:
    """
    Run a ReAct (Reasoning + Acting) loop to answer a compliance question.

    The LLM:
    1. Thinks about what information it needs
    2. Decides which tool to use (Action)
    3. Gets the tool result (Observation)
    4. Repeats until ready to answer

    Returns:
        dict with: answer (bool), reason (str), confidence (float)
    """
    # Available tools description
    tools_desc = """Available Tools:
1. search_evidence(query) - Search the article for evidence related to keywords
2. check_threshold(value, limit) - Check if a percentage exceeds a limit

To use a tool, respond with:
Action: tool_name
Input: your input"""

    # Initial prompt
    react_prompt = f"""You are a compliance analyst. Answer this question about the article.

QUESTION: {question}

ARTICLE (excerpt):
{article[:2000]}

{tools_desc}

Think step-by-step:
1. What evidence do I need to find?
2. Use search_evidence to find relevant quotes
3. If the question mentions a percentage threshold, use check_threshold
4. Based on evidence, answer YES or NO

Format:
Thought: [your reasoning]
Action: [tool name or "answer"]
Input: [tool input or your YES/NO answer with explanation]

Begin:
Thought:"""

    scratchpad = ""
    final_answer = None
    confidence = 0.5

    for i in range(max_iterations):
        # Get LLM response
        full_prompt = react_prompt + scratchpad
        response = llm.invoke(full_prompt)

        print(f"[REACT iteration {i+1}] {response[:150]}...")

        # Parse response for Action
        action_match = re.search(r"Action:\s*(\w+)", response, re.IGNORECASE)
        input_match = re.search(r"Input:\s*(.+?)(?:\n|$)", response, re.IGNORECASE | re.DOTALL)

        if not action_match:
            # No action found, try to extract answer directly
            if "YES" in response.upper():
                final_answer = True
                confidence = 0.7
            elif "NO" in response.upper():
                final_answer = False
                confidence = 0.7
            break

        action = action_match.group(1).lower()
        action_input = input_match.group(1).strip() if input_match else ""

        # Execute tool or finalize answer
        if action == "answer" or action == "final":
            if "YES" in action_input.upper():
                final_answer = True
            else:
                final_answer = False
            confidence = 0.85
            scratchpad += f"\n{response}\nFinal Answer recorded."
            break

        elif action == "search_evidence" or action == "search":
            observation = tool_search_evidence(action_input, article)
            if "EVIDENCE FOUND" in observation:
                confidence = 0.8

        elif action == "check_threshold" or action == "threshold":
            # Parse value and limit from input
            nums = re.findall(r"[\d.]+", action_input)
            if len(nums) >= 2:
                observation = tool_check_threshold(float(nums[0]), float(nums[1]))
                if "EXCEEDS" in observation:
                    confidence = 0.9
            else:
                observation = "Error: Need two numbers (value, threshold)"

        else:
            observation = f"Unknown tool: {action}"

        # Add to scratchpad
        scratchpad += f"\n{response}\nObservation: {observation}\nThought:"

    # If no answer determined, use evidence-based fallback
    if final_answer is None:
        evidence = tool_search_evidence(question, article)
        if "EVIDENCE FOUND" in evidence:
            final_answer = True  # Conservative bias
            confidence = 0.6
        else:
            final_answer = False
            confidence = 0.4

    return {
        "answer": final_answer,
        "reason": scratchpad[-300:] if scratchpad else "Direct evidence check",
        "confidence": confidence
    }


# =============================================================================
# QUESTION NODE (with ReAct)
# =============================================================================

def question_node(state: ComplianceState, llm) -> ComplianceState:
    """
    Process the current question using ReAct pattern.

    ReAct = Reasoning + Acting:
    - LLM reasons about what tool to use
    - Executes tool (search_evidence or check_threshold)
    - Observes result
    - Continues until ready to answer YES/NO
    """
    current_node = state["current_node"]
    decision_tree = state["decision_tree"]

    if not current_node or current_node not in decision_tree:
        return state

    question_data = decision_tree[current_node]
    question_text = question_data.get("text", "")
    article = state["article_text"]

    print(f"\n[{current_node}] {question_text}")
    print("[REACT] Starting ReAct reasoning loop...")

    # Run ReAct loop
    result = run_react_loop(llm, question_text, article)

    answer = result["answer"]
    reason = result["reason"]
    confidence = result["confidence"]

    # Record answer
    state["answers"][current_node] = AnswerRecord(
        question=question_text,
        answer=answer,
        evidence=reason[:500],
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
