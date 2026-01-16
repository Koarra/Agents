# SIAP Compliance Detection System (OpenAI + create_react_agent)

LangGraph application that processes articles to detect compliance "Hits" using OpenAI GPT models with native tool binding.

## How It Works

This system uses **LangGraph** for workflow orchestration and **`create_react_agent`** from `langgraph.prebuilt` for intelligent question answering with native tool binding.

### Key Difference from `google_logic`

| Feature | `google_logic` (Ollama) | `react_gpt` (OpenAI) |
|---------|-------------------------|----------------------|
| LLM | ChatOllama (llama3.1) | ChatOpenAI (GPT-4) |
| ReAct Implementation | Manual loop (text parsing) | `create_react_agent` (native) |
| Tool Binding | Not supported | Native via `bind_tools()` |
| Cost | Free (local) | Paid (API) |

### Libraries Used

- **LangGraph** (`langgraph`) - Graph-based workflow with conditional edges
- **LangGraph Prebuilt** (`langgraph.prebuilt.create_react_agent`) - Native ReAct agent
- **LangChain OpenAI** (`langchain_openai.ChatOpenAI`) - OpenAI GPT integration
- **LangChain Core** (`langchain_core.tools`) - Tool decorator for ReAct agent

### create_react_agent Usage

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def search_evidence(query: str) -> str:
    """Search the article for evidence."""
    ...

@tool
def check_threshold(value: float, threshold: float) -> str:
    """Check if value exceeds compliance threshold."""
    ...

# Create LLM with tool binding support
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# Create ReAct agent - uses native tool binding
react_agent = create_react_agent(llm, [search_evidence, check_threshold])

# Run the agent
result = react_agent.invoke({
    "messages": [HumanMessage(content="Is the client involved in cannabis?")]
})
```

The agent autonomously decides which tools to call:

```
[Agent] I need to search for cannabis-related evidence
[Tool Call] search_evidence("cannabis business cultivation")
[Observation] EVIDENCE FOUND: "vertically integrated cannabis business..."

[Agent] Found evidence. Checking for executive role...
[Tool Call] search_evidence("executive CEO management")
[Observation] EVIDENCE FOUND: "CEO holds executive position"

[Agent] Both conditions met.
FINAL ANSWER: YES
REASON: Direct cannabis involvement with executive position
CONFIDENCE: 0.92
```

### Confidence Threshold

Answers require **>75% confidence** to be accepted:
- If confidence > 0.75: Use the determined YES/NO answer
- If confidence <= 0.75: Default to NO (conservative approach)

```python
CONFIDENCE_THRESHOLD = 0.75

if confidence <= CONFIDENCE_THRESHOLD:
    answer = False  # Default to NO
    reason = f"Insufficient confidence ({confidence:.2f})"
```

### Pipeline Flow

```
Article -> Router -> Question Node (create_react_agent) -> ... -> Verdict -> HIT/NO_HIT
                          ^_________|
                         (decision tree traversal)
```

1. **Router**: Classifies article to a scenario (Cannabis, Art Dealing, etc.)
2. **Question Node**: Uses `create_react_agent` with native tool binding
3. **Conditional Edges**: Skip logic based on YES/NO answers
4. **Verdict**: Aggregates answers to determine final HIT or NO_HIT

## Installation

```bash
pip install langgraph langchain-openai langchain-core

# Set OpenAI API key
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

### Command Line

```bash
cd react_gpt

# Single document
python main.py

# Batch processing
python main.py batch
```

### Programmatic

```python
from langchain_openai import ChatOpenAI
from graph import process_article

llm = ChatOpenAI(model="gpt-4", temperature=0.3)

result = process_article(
    article_id="test_001",
    article_text="Article about cannabis dispensary...",
    llm=llm
)

print(f"Verdict: {result['final_verdict']}")  # HIT or NO_HIT
print(f"Reason: {result['verdict_reason']}")
```

## Files

| File | Purpose |
|------|---------|
| `state.py` | ComplianceState TypedDict definition |
| `tools.py` | ReAct tools with @tool decorator |
| `nodes.py` | Graph nodes using create_react_agent |
| `graph.py` | LangGraph workflow with conditional edges |
| `main.py` | Entry point with OpenAI initialization |
| `../scenarios_1/` | Scenario JSON files (decision trees) |

## Example Output

```
======================================================================
PROCESSING DOCUMENT FROM documents_test/ (OpenAI GPT-4)
======================================================================

Loaded: ../documents_test/client_doc_12.txt
Document ID: client_doc_12
Length: 2847 characters

Initializing OpenAI (gpt-4)...

[ROUTER] Classified as: Cannabis Business
[ROUTER] Confidence: 1.0
[ROUTER] Starting at: Q1

[Q1] Is the client involved in a cannabis-related business?
[REACT] Starting ReAct agent with create_react_agent...
[REACT] Agent response: I'll search for cannabis-related evidence...
[Q1] Answer: YES (confidence: 0.92)
[Q1] Next: Q2

[Q2] Is the cannabis-related activity illegal?
[REACT] Starting ReAct agent with create_react_agent...
[Q2] Answer: NO (confidence: 0.88)
[Q2] Next: Q3

...

==================================================
VERDICT DETERMINATION
==================================================
[VERDICT] HIT - Risk triggers found
  - Q5: Does the client hold an executive management or board of...
[VERDICT] Risk Score: 0.30
```

## Comparison: Manual ReAct vs create_react_agent

### Manual ReAct (`google_logic`)
```python
# Parse LLM text output for actions
action_match = re.search(r"Action:\s*(\w+)", response)
input_match = re.search(r"Action Input:\s*(.+)", response)

# Execute tool manually
if action == "search_evidence":
    observation = search_evidence.invoke(action_input)
```

### Native ReAct (`react_gpt`)
```python
# LLM natively calls tools via bind_tools()
react_agent = create_react_agent(llm, tools)
result = react_agent.invoke({"messages": messages})
# Tools are called automatically by the agent
```

The native approach is more reliable because:
1. No text parsing required
2. LLM understands tool schemas natively
3. Better error handling
4. More consistent tool invocations
