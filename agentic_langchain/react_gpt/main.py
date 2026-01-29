import os
import sys
import json
from glob import glob

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_openai import AzureChatOpenAI
from graph import process_article, process_batch
from state import Verdict

# Azure AD token provider for authentication
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"
)


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
# ===========================================================================

def demo_single_document():
    """Process a single document from documents_test."""
    print("\n" + "=" * 70)
    print("PROCESSING DOCUMENT FROM documents_test/ (Azure OpenAI)")
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

    # Initialize LLM - Azure OpenAI with tool binding support
    print("\nInitializing Azure OpenAI with Azure AD authentication...")

    llm = AzureChatOpenAI(
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        azure_ad_token_provider=token_provider,
        temperature=0.3
    )

    # Test LLM connection
    print("Testing LLM connection...")
    test_response = llm.invoke([{"role": "user", "content": "Say hello"}])
    print(f"LLM test successful: {test_response.content[:50]}...")

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
    print("BATCH PROCESSING ALL DOCUMENTS FROM documents_test/ (Azure OpenAI)")
    print("=" * 70)

    documents = load_documents()

    if not documents:
        print("No documents found in documents_test/")
        return None

    print(f"\nFound {len(documents)} document(s):")
    for doc in documents:
        print(f"  - {doc['id']}")

    # Initialize LLM
    print("\nInitializing Azure OpenAI with Azure AD authentication...")

    llm = AzureChatOpenAI(
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        azure_ad_token_provider=token_provider,
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
    # Check for Azure OpenAI endpoint (using Azure AD authentication, no API key needed)
    if not os.environ.get("AZURE_OPENAI_ENDPOINT"):
        print("ERROR: AZURE_OPENAI_ENDPOINT environment variable not set")
        print("Please set it with: export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'")
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
