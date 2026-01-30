"""
Node Functions for SIAP Compliance Detection Graph.

Each node represents a stage in the compliance detection pipeline:
1. Router Node - Classifies article to a scenario
2. Question Nodes - Answer compliance questions using ReAct agent with tools
3. Verdict Node - Determines final HIT/NO_HIT status

Uses langgraph.prebuilt.create_react_agent for the ReAct pattern.
This requires an LLM that supports tool binding (e.g., OpenAI GPT-4).
"""

from state import ComplianceState, Verdict, AnswerRecord, ConfidenceLevel, MissingFieldRecord
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

# Confidence thresholds for routing decisions
CONFIDENCE_HIGH = 0.75      # >75%: Reliable answer, continue normally
CONFIDENCE_MEDIUM = 0.50    # 50-75%: Answer accepted but flagged as uncertain
CONFIDENCE_LOW = 0.50       # <50%: Cannot determine, mark as MISSING


def get_confidence_level(confidence: float) -> ConfidenceLevel:
    """Classify confidence into HIGH, MEDIUM, or LOW."""
    if confidence > CONFIDENCE_HIGH:
        return ConfidenceLevel.HIGH
    elif confidence >= CONFIDENCE_MEDIUM:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW


def get_suggested_documents(question: str) -> str:
    """
    Suggest documents that could help answer the question.
    Maps question types to relevant document requests.
    """
    question_lower = question.lower()

    suggestions = {
        "income": "income statements, tax returns, financial records",
        "percentage": "ownership documents, shareholder registry, tax returns",
        "executive": "corporate filings, org charts, employment contracts",
        "board": "corporate filings, board minutes, annual reports",
        "ownership": "shareholder registry, ownership certificates, cap table",
        "direct": "business structure documents, operational agreements",
        "illegal": "business licenses, regulatory filings, compliance certificates",
        "hemp": "product documentation, regulatory permits, lab certificates",
        "cannabis": "business licenses, state permits, operational documents",
        "shipping": "vessel registration, maritime licenses, charter agreements",
        "maritime": "maritime certifications, port registrations, shipping contracts",
        "tanker": "vessel documentation, cargo manifests, shipping licenses",
        "cargo": "cargo manifests, shipping contracts, port documentation",
        "vessel": "vessel registration, ownership certificates, flag state documentation",
        "inland": "waterway permits, navigation licenses, route documentation",
        "passenger": "passenger vessel licenses, cruise line registrations",
        "dredging": "dredging permits, maritime operation licenses",
        "intermediary": "service agreements, consultancy contracts, engagement letters",
        "broker": "brokerage licenses, service contracts, commission agreements",
        "consultant": "consultancy agreements, scope of work documents, fee arrangements",
        "lobbyist": "lobbying registration, government relations disclosures, activity reports",
        "government": "government contracts, public sector engagement records, SNE agreements",
        "corruption": "anti-corruption certifications, compliance policies, due diligence reports",
        "pep": "PEP screening results, relationship disclosures, beneficial ownership records",
        "state": "government contract records, public entity agreements, official correspondence",
        "reputable": "regulatory licenses, professional certifications, compliance attestations",
        "charity": "charity registration, tax-exempt status certificates, annual reports",
        "charitable": "charitable purpose documentation, mission statements, program descriptions",
        "donor": "donor records, funding source documentation, grant agreements",
        "fundraising": "fundraising licenses, donor lists, campaign documentation",
        "foundation": "foundation registration, governance documents, financial statements",
        "nonprofit": "501(c)(3) status, nonprofit registration, board minutes",
        "sanction": "sanctions screening results, OFAC compliance records, country risk assessments",
        "religious": "religious organization registration, faith-based entity documentation, clergy credentials",
        "church": "church registration, religious tax exemption, congregation records",
        "faith": "faith-based organization documents, religious affiliation records",
        "construction": "construction contracts, project agreements, government procurement records",
        "infrastructure": "infrastructure project documents, government tender records, public works contracts",
        "public": "public sector contracts, government agency agreements, procurement documentation",
        "federal": "federal contract records, national government project documentation",
        "local": "local government contracts, municipal project records, regional agreements",
        "private": "private financing records, funding source documentation, investment agreements",
        "gambling": "gambling licenses, gaming permits, regulatory approvals, compliance certificates",
        "casino": "casino operating license, gaming commission approvals, regulatory filings",
        "betting": "bookmaker license, sports betting permits, wagering authorizations",
        "online": "online gambling license, iGaming permits, jurisdiction approvals",
        "regulated": "regulatory licenses, compliance certifications, jurisdiction authorizations",
        "illegal": "legal status documentation, country operation permits, regulatory clearances",
        "sports": "sports federation membership, athlete contracts, competition records",
        "agent": "sports agent license, representation agreements, athlete contracts",
        "broadcasting": "broadcasting rights agreements, media contracts, licensing documentation",
        "federation": "federation membership records, organizational registration, governance documents",
        "club": "club registration, membership records, financial statements",
        "athlete": "athlete contracts, sponsorship agreements, income documentation",
        "professional": "professional athlete status, career earnings, contract documentation"
    }

    for keyword, docs in suggestions.items():
        if keyword in question_lower:
            return docs

    return "additional documentation related to this question"


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
4. Report your actual confidence level. If very uncertain (<50%), say so clearly.

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

        # Classify confidence level
        confidence_level = get_confidence_level(confidence)

        # Log confidence routing
        if confidence_level == ConfidenceLevel.HIGH:
            print(f"[REACT] High confidence ({confidence:.2f}) - answer reliable")
        elif confidence_level == ConfidenceLevel.MEDIUM:
            print(f"[REACT] Medium confidence ({confidence:.2f}) - answer accepted but flagged")
        else:
            print(f"[REACT] Low confidence ({confidence:.2f}) - marking as MISSING")
            # For low confidence, we don't force an answer - we mark it as missing
            reason = f"Insufficient confidence ({confidence:.2f}): {reason}"

        return {
            "answer": answer,
            "reason": reason,
            "confidence": confidence,
            "confidence_level": confidence_level.value,
            "is_missing": confidence_level == ConfidenceLevel.LOW
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

    confidence = evidence_result["confidence"]
    confidence_level = get_confidence_level(confidence)

    if evidence_result["found"] and confidence_level == ConfidenceLevel.HIGH:
        return {
            "answer": True,
            "reason": evidence_result["evidence"][0][:300] if evidence_result["evidence"] else "Evidence found",
            "confidence": confidence,
            "confidence_level": confidence_level.value,
            "is_missing": False
        }
    else:
        is_missing = confidence_level == ConfidenceLevel.LOW
        return {
            "answer": False,
            "reason": f"Insufficient evidence (confidence: {confidence:.2f})",
            "confidence": confidence,
            "confidence_level": confidence_level.value,
            "is_missing": is_missing
        }


# =============================================================================
# QUESTION NODE (with ReAct)
# =============================================================================

def question_node(state: ComplianceState, llm) -> ComplianceState:
    """
    Process the current question using ReAct pattern with create_react_agent.

    Enhanced with confidence-based routing:
    - HIGH confidence (>75%): Answer YES/NO, continue normally
    - MEDIUM confidence (50-75%): Answer YES/NO, flag as uncertain
    - LOW confidence (<50%): Mark as MISSING, trigger early termination

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
    confidence_level = result.get("confidence_level", ConfidenceLevel.MEDIUM.value)
    is_missing = result.get("is_missing", False)

    # Record answer with confidence tracking
    state["answers"][current_node] = AnswerRecord(
        question=question_text,
        answer=answer,
        evidence=reason,
        confidence=confidence,
        confidence_level=confidence_level,
        is_missing=is_missing
    )

    answer_str = "YES" if answer else "NO"
    print(f"[{current_node}] Answer: {answer_str} (confidence: {confidence:.2f}, level: {confidence_level})")

    # Handle confidence-based routing
    if is_missing:
        # LOW confidence: Mark as missing and trigger early termination
        missing_record = MissingFieldRecord(
            question_id=current_node,
            question_text=question_text,
            confidence=confidence,
            suggested_documents=get_suggested_documents(question_text)
        )
        state["missing_fields"].append(missing_record)
        state["early_termination"] = True
        state["current_node"] = ""  # Stop processing

        print(f"[{current_node}] LOW CONFIDENCE - Marked as MISSING")
        print(f"[{current_node}] Suggested documents: {missing_record['suggested_documents']}")
        print(f"[{current_node}] Early termination triggered")

    elif confidence_level == ConfidenceLevel.MEDIUM.value:
        # MEDIUM confidence: Flag as uncertain but continue
        state["uncertain_answers"].append(current_node)
        print(f"[{current_node}] MEDIUM CONFIDENCE - Flagged as uncertain")

        # Determine next node normally
        next_node = question_data.get("next_if_yes") if answer else question_data.get("next_if_no")
        if next_node is None:
            state["current_node"] = ""
            print(f"[{current_node}] Terminal node reached")
        else:
            state["current_node"] = next_node
            print(f"[{current_node}] Next: {next_node}")

    else:
        # HIGH confidence: Continue normally
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
    Final scoring node that determines verdict.

    Verdict types:
    - HIT: Risk detected with high confidence → escalate/reject client
    - NO_HIT: Clean, no risks found → approve client
    - MISSING_INFO: Article lacks critical information → request more documents

    Handles uncertainty through:
    - Early termination tracking (low confidence answers)
    - Missing fields tracking with suggested documents
    - Uncertain answer flagging
    """
    answers = state["answers"]
    missing_fields = state.get("missing_fields", [])
    uncertain_answers = state.get("uncertain_answers", [])
    early_termination = state.get("early_termination", False)

    print("\n" + "=" * 50)
    print("VERDICT DETERMINATION")
    print("=" * 50)

    # Check for MISSING_INFO verdict first
    if early_termination or missing_fields:
        state["final_verdict"] = Verdict.MISSING_INFO

        # Build missing fields description
        missing_descriptions = []
        suggested_docs = set()
        for mf in missing_fields:
            missing_descriptions.append(f"{mf['question_id']}: {mf['question_text'][:50]}...")
            suggested_docs.add(mf['suggested_documents'])

        state["verdict_reason"] = f"Cannot determine compliance status. Missing: {'; '.join(missing_descriptions)}"
        state["recommended_action"] = f"Please provide: {', '.join(suggested_docs)}"
        state["risk_score"] = 0.0  # Cannot score without full information

        print(f"[VERDICT] MISSING_INFO - Insufficient data for determination")
        print(f"[VERDICT] Missing fields:")
        for mf in missing_fields:
            print(f"  - {mf['question_id']}: {mf['question_text'][:60]}... (conf: {mf['confidence']:.2f})")
        print(f"[VERDICT] Suggested documents: {', '.join(suggested_docs)}")

        return state

    # Calculate risk indicators for HIT/NO_HIT determination
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
        state["recommended_action"] = "Escalate to compliance team for manual review"
        print(f"[VERDICT] HIT - Risk triggers found")
        for reason in hit_reasons:
            print(f"  - {reason}")
    else:
        state["final_verdict"] = Verdict.NO_HIT
        state["verdict_reason"] = "No compliance risks identified"
        state["recommended_action"] = "Approve - proceed with onboarding"
        print(f"[VERDICT] NO_HIT - Clean")

    # Flag if there were uncertain answers (MEDIUM confidence)
    if uncertain_answers:
        state["verdict_reason"] += f" [Note: {len(uncertain_answers)} answer(s) had medium confidence]"
        state["recommended_action"] += f" [Review flagged questions: {', '.join(uncertain_answers)}]"
        print(f"[VERDICT] Note: {len(uncertain_answers)} uncertain answer(s) flagged for review")
        for ua in uncertain_answers:
            print(f"  - {ua}")

    state["risk_score"] = risk_score
    print(f"[VERDICT] Risk Score: {risk_score:.2f}")

    return state


# =============================================================================
# ROUTING FUNCTIONS (For Conditional Edges)
# =============================================================================

def should_continue_questions(state: ComplianceState) -> str:
    """
    Determines if we should continue asking questions or move to verdict.

    Handles early termination when confidence is too low (MISSING_INFO case).

    Returns:
        "question" - Continue to next question
        "verdict" - Move to verdict determination
        "end" - Early termination (no scenario matched)
    """
    # Check if scenario was matched
    if state["scenario_id"] is None:
        return "end"

    # Check for early termination due to low confidence (MISSING_INFO)
    if state.get("early_termination", False):
        print("[ROUTING] Early termination triggered - moving to verdict (MISSING_INFO)")
        return "verdict"

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
