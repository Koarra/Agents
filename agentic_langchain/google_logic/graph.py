"""
LangGraph Workflow for SIAP Compliance Detection.

Implements decision tree traversal with conditional edges for skip logic.
Processes articles autonomously to reach HIT or NO_HIT verdict.

Graph Structure:
    START → router → question (loop with skip logic) → verdict → END
"""

from langgraph.graph import StateGraph, END
from state import ComplianceState, Verdict, create_initial_state
from nodes import (
    router_node,
    question_node,
    verdict_node,
    should_continue_questions,
    route_after_router
)


def create_compliance_graph(llm):
    """
    Create the compliance detection graph with conditional edges.

    The graph implements skip logic through conditional_edges:
    - If Q1 returns NO, it may skip to Q3 (defined in scenario JSON)
    - Each question's next_if_yes/next_if_no controls the flow

    Args:
        llm: Language model instance (e.g., OllamaLLM)

    Returns:
        Compiled LangGraph
    """

    # Initialize graph with ComplianceState
    graph = StateGraph(ComplianceState)

    # ==========================================================================
    # DEFINE NODES
    # ==========================================================================

    # Router Node - Classifies article to scenario
    def router(state: ComplianceState) -> ComplianceState:
        return router_node(state, llm)

    # Question Node - Processes current question with ReAct tools
    def question(state: ComplianceState) -> ComplianceState:
        return question_node(state, llm)

    # Verdict Node - Determines final HIT/NO_HIT
    def verdict(state: ComplianceState) -> ComplianceState:
        return verdict_node(state)

    # Add nodes to graph
    graph.add_node("router", router)
    graph.add_node("question", question)
    graph.add_node("verdict", verdict)

    # ==========================================================================
    # DEFINE EDGES (Including Conditional Skip Logic)
    # ==========================================================================

    # Entry point
    graph.set_entry_point("router")

    # After router: go to question or end (if no scenario matched)
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "question": "question",
            "end": END
        }
    )

    # After question: loop back, go to verdict, or end
    # This implements the SKIP LOGIC - the question node updates current_node
    # based on the answer (next_if_yes or next_if_no from scenario JSON)
    graph.add_conditional_edges(
        "question",
        should_continue_questions,
        {
            "question": "question",  # Loop to next question
            "verdict": "verdict",     # Tree traversal complete
            "end": END                # Early termination
        }
    )

    # After verdict: end
    graph.add_edge("verdict", END)

    return graph.compile()


def process_article(article_id: str, article_text: str, llm) -> dict:
    """
    Process a single article through the compliance detection pipeline.

    Args:
        article_id: Unique identifier for the article
        article_text: The article content to analyze
        llm: Language model instance

    Returns:
        Final state dict with verdict and evidence
    """
    print("\n" + "=" * 60)
    print(f"PROCESSING: {article_id}")
    print("=" * 60)

    # Create initial state
    initial_state = create_initial_state(article_id, article_text)

    # Build and run graph
    graph = create_compliance_graph(llm)
    final_state = graph.invoke(initial_state)

    # Print summary
    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(f"Article: {article_id}")
    print(f"Scenario: {final_state.get('scenario_name', 'None')}")
    print(f"Verdict: {final_state['final_verdict']}")
    print(f"Risk Score: {final_state['risk_score']:.2f}")
    print(f"Reason: {final_state['verdict_reason']}")

    return dict(final_state)


def process_batch(articles: list[dict], llm) -> list[dict]:
    """
    Process multiple articles in parallel.

    Args:
        articles: List of dicts with keys: id, text
        llm: Language model instance

    Returns:
        List of final states
    """
    import concurrent.futures

    print("\n" + "=" * 60)
    print(f"BATCH PROCESSING: {len(articles)} articles")
    print("=" * 60)

    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_article, art["id"], art["text"], llm): art["id"]
            for art in articles
        }

        for future in concurrent.futures.as_completed(futures):
            article_id = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"[ERROR] {article_id}: {e}")
                results.append({
                    "article_id": article_id,
                    "final_verdict": Verdict.PENDING,
                    "error": str(e)
                })

    # Summary
    hits = sum(1 for r in results if r.get("final_verdict") == Verdict.HIT)
    no_hits = sum(1 for r in results if r.get("final_verdict") == Verdict.NO_HIT)

    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    print(f"Total: {len(results)}")
    print(f"HITs: {hits}")
    print(f"NO_HITs: {no_hits}")

    return results


