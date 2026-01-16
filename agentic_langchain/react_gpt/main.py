"""
Main Entry Point for SIAP Compliance Detection System (OpenAI Version).

Demonstrates autonomous HIT/NO_HIT determination using LangGraph
decision trees with conditional edges for skip logic.

Uses OpenAI GPT models with native tool binding via create_react_agent.
Uses documents from documents_test/ folder for testing.
"""

import os
import sys
import json
from glob import glob

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from graph import process_article, process_batch
from state import Verdict


# =============================================================================
# DOCUMENT LOADER
# =============================================================================

def load_documents(documents_dir: str = None) -> list[dict]:
    """
    Load test documents from documents_test directory.

    Returns:
        List of dicts with keys: id, text, path
    """
    if documents_dir is None:
        documents_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "documents_test"
        )

    documents = []
    txt_files = glob(os.path.join(documents_dir, "*.txt"))

    for filepath in sorted(txt_files):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        doc_id = os.path.basename(filepath).replace(".txt", "")
        documents.append({
            "id": doc_id,
            "text": content,
            "path": filepath
        })

    return documents


# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

def demo_single_document():
    """Process a single document from documents_test."""
    print("\n" + "=" * 70)
    print("PROCESSING DOCUMENT FROM documents_test/ (OpenAI GPT-4)")
    print("=" * 70)

    documents = load_documents()

    if not documents:
        print("No documents found in documents_test/")
        return None

    # Process first document
    doc = documents[0]
    print(f"\nLoaded: {doc['path']}")
    print(f"Document ID: {doc['id']}")
    print(f"Length: {len(doc['text'])} characters")

    # Initialize LLM - OpenAI with tool binding support
    print("\nInitializing OpenAI (gpt-4)...")
    print("Note: Ensure OPENAI_API_KEY environment variable is set")

    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.3
    )

    # Process
    result = process_article(doc["id"], doc["text"], llm)

    # Print detailed results
    print("\n" + "-" * 70)
    print("ANSWERS COLLECTED:")
    print("-" * 70)
    for node_id, answer in sorted(result.get("answers", {}).items()):
        ans_str = "YES" if answer["answer"] else "NO"
        print(f"\n{node_id}: {ans_str}")
        print(f"  Q: {answer['question'][:70]}...")
        print(f"  Evidence: {answer['evidence'][:100]}...")

    # Save result
    output_file = os.path.join(os.path.dirname(__file__), "result.json")
    with open(output_file, "w") as f:
        # Convert Verdict enum to string for JSON
        output_data = {
            "article_id": result.get("article_id"),
            "scenario": result.get("scenario_name"),
            "verdict": str(result.get("final_verdict")),
            "risk_score": result.get("risk_score", 0),
            "reason": result.get("verdict_reason", ""),
            "answers": {
                k: {
                    "question": v["question"],
                    "answer": "YES" if v["answer"] else "NO",
                    "evidence": v["evidence"],
                    "confidence": v["confidence"]
                }
                for k, v in result.get("answers", {}).items()
            }
        }
        json.dump(output_data, f, indent=2)

    print(f"\nResult saved to: {output_file}")

    return result


def demo_all_documents():
    """Process all documents from documents_test."""
    print("\n" + "=" * 70)
    print("BATCH PROCESSING ALL DOCUMENTS FROM documents_test/ (OpenAI GPT-4)")
    print("=" * 70)

    documents = load_documents()

    if not documents:
        print("No documents found in documents_test/")
        return None

    print(f"\nFound {len(documents)} document(s):")
    for doc in documents:
        print(f"  - {doc['id']}")

    # Initialize LLM
    print("\nInitializing OpenAI (gpt-4)...")
    print("Note: Ensure OPENAI_API_KEY environment variable is set")

    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.3
    )

    # Process all documents
    articles = [{"id": d["id"], "text": d["text"]} for d in documents]
    results = process_batch(articles, llm)

    # Save results
    output_file = os.path.join(os.path.dirname(__file__), "batch_results.json")
    output_data = []
    for result in results:
        output_data.append({
            "article_id": result.get("article_id"),
            "scenario": result.get("scenario_name"),
            "verdict": str(result.get("final_verdict")),
            "risk_score": result.get("risk_score", 0),
            "reason": result.get("verdict_reason", "")
        })

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)

    # Check for command line argument
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        # Run single document by default
        choice = "single"

    if choice == "single" or choice == "1":
        demo_single_document()
    elif choice == "batch" or choice == "all" or choice == "2":
        demo_all_documents()
    else:
        demo_single_document()


if __name__ == "__main__":
    main()
