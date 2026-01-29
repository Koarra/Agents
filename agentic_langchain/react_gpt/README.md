# SIAP Compliance Detection System

An autonomous compliance detection pipeline using **LangGraph**, **ReAct agents**, and **Azure OpenAI** for document analysis and risk assessment.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Components Deep Dive](#components-deep-dive)
- [The ReAct Pattern](#the-react-pattern)
- [Decision Tree & Skip Logic](#decision-tree--skip-logic)
- [Tools Explained](#tools-explained)
- [Why Keyword Functions?](#why-keyword-functions)
- [Multilingual Support](#multilingual-support)
- [Extending the System](#extending-the-system)
- [Troubleshooting](#troubleshooting)

---

## Overview

This system automatically analyzes documents to detect compliance risks, producing a **HIT** (risk detected) or **NO_HIT** (clean) verdict with supporting evidence.

### Key Features

| Feature | Description |
|---------|-------------|
| **Autonomous** | No human intervention for standard cases |
| **Evidence-Based** | Every decision backed by actual document quotes |
| **Auditable** | Full reasoning chain for compliance review |
| **Configurable** | Scenarios defined in JSON, no code changes needed |
| **Secure** | Azure AD authentication (no API keys in code) |
| **Grounded** | Tools prevent LLM hallucination |

### Use Cases

- Anti-Money Laundering (AML) screening
- Know Your Customer (KYC) document review
- Regulatory compliance checks
- Risk assessment automation

### Technology Stack

| Component | Technology |
|-----------|------------|
| Orchestration | LangGraph (StateGraph) |
| LLM | Azure OpenAI (GPT-4o-mini) |
| Agent Pattern | ReAct (`create_react_agent`) |
| Authentication | Azure AD (`DefaultAzureCredential`) |
| State Management | TypedDict |
| Tools | LangChain `@tool` decorator |

---

## Architecture

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           COMPLIANCE DETECTION PIPELINE                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────┐    ┌──────────┐    ┌──────────────────┐    ┌──────────┐     │
│   │          │    │          │    │                  │    │          │     │
│   │ DOCUMENT │───▶│  ROUTER  │───▶│  QUESTION LOOP   │───▶│ VERDICT  │     │
│   │          │    │          │    │  (ReAct Agent)   │    │          │     │
│   └──────────┘    └────┬─────┘    └────────┬─────────┘    └────┬─────┘     │
│                        │                   │                    │           │
│                        ▼                   ▼                    ▼           │
│                   ┌─────────┐        ┌───────────┐        ┌─────────┐      │
│                   │Scenario │        │  TOOLS    │        │HIT /    │      │
│                   │  JSON   │        │           │        │NO_HIT   │      │
│                   │         │        │• search   │        │+ reason │      │
│                   └─────────┘        │• threshold│        │+ score  │      │
│                                      └───────────┘        └─────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### LangGraph State Machine

```
                    ┌─────────┐
                    │  START  │
                    └────┬────┘
                         │
                         ▼
                    ┌─────────┐
            ┌──────▶│ ROUTER  │──────┐
            │       └────┬────┘      │
            │            │           │ (no scenario matched)
            │            ▼           ▼
            │       ┌─────────┐    ┌─────┐
            │       │QUESTION │    │ END │
            │       │  NODE   │    └─────┘
            │       └────┬────┘
            │            │
            │    ┌───────┴───────┐
            │    │               │
            │    ▼               ▼
            │ (more Qs)    (tree complete)
            │    │               │
            │    │          ┌────┴────┐
            └────┘          │ VERDICT │
                            └────┬────┘
                                 │
                                 ▼
                            ┌─────────┐
                            │   END   │
                            └─────────┘
```

### Component Flow

1. **Document Input** → Raw text enters the pipeline
2. **Router Node** → Classifies document to a scenario (Cannabis, Art, Commodity)
3. **Question Loop** → ReAct agent answers YES/NO questions using tools
4. **Verdict Node** → Aggregates answers, calculates risk score, determines HIT/NO_HIT

---

## How It Works

### Step-by-Step Example

```
1. DOCUMENT LOADED
   "GreenLeaf Inc. operates dispensaries in Colorado, generating
    60% of revenue from cannabis sales..."

2. ROUTER CLASSIFIES → "cannabis_business" scenario
   Matched keywords: ["cannabis", "dispensary"]

3. DECISION TREE LOADED FROM JSON
   Q1: "Is the client involved in cannabis business?"
   Q2: "Does cannabis generate more than 10% of revenue?"
   Q3: "Is the cannabis business legal in jurisdiction?"

4. REACT AGENT PROCESSES Q1
   ┌─────────────────────────────────────────────────────────┐
   │ THINK: I need to find evidence of cannabis involvement  │
   │ ACT:   search_evidence("cannabis business dispensary")  │
   │ OBSERVE: "EVIDENCE FOUND: ...operates dispensaries..."  │
   │ ANSWER: YES (confidence: 0.92)                          │
   └─────────────────────────────────────────────────────────┘

5. SKIP LOGIC → Q1=YES → proceed to Q2

6. REACT AGENT PROCESSES Q2
   ┌─────────────────────────────────────────────────────────┐
   │ THINK: Need to find revenue percentage                  │
   │ ACT:   search_evidence("cannabis revenue percentage")   │
   │ OBSERVE: "...60% of revenue from cannabis..."           │
   │ ACT:   check_threshold(60.0, 10.0)                      │
   │ OBSERVE: "EXCEEDS THRESHOLD: 60% > 10%"                 │
   │ ANSWER: YES (confidence: 0.95)                          │
   └─────────────────────────────────────────────────────────┘

7. VERDICT DETERMINATION
   ┌─────────────────────────────────────────────────────────┐
   │ Q1: YES (cannabis involvement)                          │
   │ Q2: YES (exceeds 10% threshold) ← RED FLAG              │
   │                                                         │
   │ Result: HIT | Risk Score: 0.6                           │
   │ Reason: "Q2: Revenue exceeds 10% threshold"             │
   └─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
react_gpt/
├── main.py              # Entry point, Azure OpenAI setup, demo functions
├── graph.py             # LangGraph workflow definition
├── nodes.py             # Node functions (router, question, verdict)
├── state.py             # State definitions (ComplianceState, Verdict)
├── tools.py             # ReAct tools (search_evidence, check_threshold)
├── requirements.txt     # Python dependencies
├── explanations.txt     # Detailed technical explanation
└── README.md            # This file

../scenarios_1/          # Scenario JSON definitions
├── cannabis_business.json
├── art_dealing.json
└── commodity_trading.json

../documents_test/       # Test documents
├── doc_001.txt
├── doc_002.txt
└── ...
```

---

## Installation

### Prerequisites

- Python 3.10+
- Azure OpenAI resource with deployed model
- Azure CLI (for authentication)

### Setup

```bash
# Navigate to project
cd react_gpt

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Packages

```
langchain-openai>=0.1.0
langgraph>=0.0.50
azure-identity>=1.15.0
```

---

## Configuration

### Environment Variables

```bash
# Required
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"

# Optional (with defaults)
export AZURE_OPENAI_DEPLOYMENT="gpt-4o-mini"
export AZURE_OPENAI_API_VERSION="2024-02-01"
```

### Azure AD Authentication

The system uses `DefaultAzureCredential` which tries these methods in order:

| Priority | Method | Use Case |
|----------|--------|----------|
| 1 | Environment Variables | CI/CD pipelines |
| 2 | Managed Identity | Azure VMs, App Service |
| 3 | Azure CLI | Local development |
| 4 | VS Code credentials | IDE integration |

**For local development:**
```bash
az login
```

**Code implementation:**
```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"
)

llm = AzureChatOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
    api_version="2024-02-01",
    azure_ad_token_provider=token_provider,
    temperature=0.3
)
```

---

## Usage

### Single Document Analysis

```bash
python main.py single
```

Processes the first document in `documents_test/` and saves to `result.json`.

### Batch Processing

```bash
python main.py batch
```

Processes all documents in parallel, saves to `batch_results.json`.

### Programmatic Usage

```python
from langchain_openai import AzureChatOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from graph import process_article

# Setup Azure AD auth
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"
)

llm = AzureChatOpenAI(
    azure_endpoint="https://your-resource.openai.azure.com/",
    azure_deployment="gpt-4o-mini",
    azure_ad_token_provider=token_provider
)

# Process document
result = process_article(
    article_id="test_001",
    article_text="Article about cannabis dispensary...",
    llm=llm
)

print(f"Verdict: {result['final_verdict']}")  # HIT or NO_HIT
print(f"Reason: {result['verdict_reason']}")
```

### Output Format

```json
{
  "article_id": "doc_001",
  "scenario": "Cannabis Business",
  "verdict": "Verdict.HIT",
  "risk_score": 0.6,
  "reason": "Q2: Does cannabis generate more than 10% of revenue?...",
  "answers": {
    "Q1": {
      "question": "Is the client involved in cannabis business?",
      "answer": "YES",
      "evidence": "operates dispensaries in Colorado...",
      "confidence": 0.92
    },
    "Q2": {
      "question": "Does cannabis generate more than 10% of revenue?",
      "answer": "YES",
      "evidence": "60% of revenue from cannabis sales...",
      "confidence": 0.95
    }
  }
}
```

---

## Components Deep Dive

### state.py - State Management

```python
class ComplianceState(TypedDict):
    article_id: str           # Document identifier
    article_text: str         # Full document content
    scenario_id: str          # Matched scenario (e.g., "cannabis_business")
    scenario_name: str        # Human-readable name
    current_node: str         # Current question (Q1, Q2, etc.)
    decision_tree: dict       # Questions loaded from JSON
    answers: dict             # Collected YES/NO answers with evidence
    final_verdict: Verdict    # HIT / NO_HIT / PENDING
    verdict_reason: str       # Explanation for verdict
    risk_score: float         # 0.0 to 1.0
```

### graph.py - LangGraph Workflow

```python
graph = StateGraph(ComplianceState)

# Add nodes
graph.add_node("router", router_node)
graph.add_node("question", question_node)
graph.add_node("verdict", verdict_node)

# Set entry point
graph.set_entry_point("router")

# Conditional edges for routing
graph.add_conditional_edges("router", route_after_router,
    {"question": "question", "end": END})

graph.add_conditional_edges("question", should_continue_questions,
    {"question": "question", "verdict": "verdict", "end": END})

graph.add_edge("verdict", END)
```

### nodes.py - Node Functions

**Router Node** - Classifies documents:
```python
def router_node(state, llm):
    # 1. Try keyword matching first (fast)
    classification = classify_scenario(article, scenarios)

    # 2. Fall back to LLM if keywords don't match
    if not classification["scenario_id"]:
        response = llm.invoke(classification_prompt)
```

**Question Node** - Uses ReAct agent:
```python
def question_node(state, llm):
    # Create agent with tools
    react_agent = create_react_agent(llm, tools)

    # Run agent
    result = react_agent.invoke({"messages": [prompt]})

    # Update state with answer
    state["answers"][current_node] = answer_record
    state["current_node"] = next_node  # Skip logic
```

**Verdict Node** - Calculates risk:
```python
def verdict_node(state):
    for answer in state["answers"]:
        if is_yes and is_red_flag_question:
            risk_score += 0.3

    state["final_verdict"] = Verdict.HIT if risk_found else Verdict.NO_HIT
```

---

## The ReAct Pattern

**ReAct** = **Rea**soning + **Act**ing

The LLM iteratively:
1. **Thinks** about what information is needed
2. **Acts** by calling a tool
3. **Observes** the tool's output
4. **Repeats** until confident in the answer

### Why ReAct?

| Approach | Problem |
|----------|---------|
| Pure LLM | Can hallucinate evidence that doesn't exist |
| Pure Rules | Can't understand context or nuance |
| **ReAct** | LLM reasons, tools provide grounded evidence |

### Example Trace

```
[THINK]   I need to find if the client is involved in cannabis.
          Let me search for evidence in the document.

[ACT]     search_evidence("cannabis business involvement")

[OBSERVE] EVIDENCE FOUND (confidence: 0.85):
          - "...operates three dispensaries across Colorado..."
          - "...licensed cannabis retailer since 2019..."

[THINK]   The evidence clearly shows cannabis involvement.
          Multiple dispensaries and explicit licensing mentioned.

[ANSWER]  FINAL ANSWER: YES
          REASON: Document states client operates dispensaries
          CONFIDENCE: 0.92
```

### Confidence Threshold

Answers require **>75% confidence** to be YES:
```python
CONFIDENCE_THRESHOLD = 0.75

if confidence <= CONFIDENCE_THRESHOLD:
    answer = False  # Conservative: default to NO
```

---

## Decision Tree & Skip Logic

Scenarios define decision trees with conditional branching:

```json
{
  "name": "Cannabis Business",
  "start": "Q1",
  "questions": {
    "Q1": {
      "text": "Is the client involved in cannabis business?",
      "next_if_yes": "Q2",
      "next_if_no": null
    },
    "Q2": {
      "text": "Does cannabis generate more than 10% of revenue?",
      "next_if_yes": "Q3",
      "next_if_no": null
    },
    "Q3": {
      "text": "Is the cannabis activity legal in the jurisdiction?",
      "next_if_yes": null,
      "next_if_no": null
    }
  }
}
```

### Skip Logic Diagram

```
Q1: Is client in cannabis?
    │
    ├── NO ──────────────────────▶ STOP (NO_HIT)
    │                              No further questions needed
    │
    └── YES ─▶ Q2: Revenue > 10%?
                   │
                   ├── NO ────────▶ STOP (low risk, NO_HIT)
                   │
                   └── YES ─▶ Q3: Is it legal?
                                  │
                                  └── ... continue
```

**Benefits:**
- Reduces unnecessary LLM calls
- Faster processing
- Logical flow matching real compliance procedures

---

## Tools Explained

### search_evidence(query: str) → str

Searches the document for text matching query keywords.

```python
# Input
search_evidence("cannabis revenue percentage")

# Output
"EVIDENCE FOUND (confidence: 0.85):
- ...generating 60% of revenue from cannabis sales...
- ...dispensary operations contribute majority of income..."
```

**How it works:**
1. Extracts keywords from query (removes stop words)
2. Searches document using regex
3. Extracts surrounding context (300 chars)
4. Returns top 3 snippets with confidence score

### check_threshold(value: float, threshold: float) → str

Deterministic numerical comparison - **no LLM math**.

```python
# Input
check_threshold(60.0, 10.0)

# Output
"EXCEEDS THRESHOLD: 60% > 10%"
```

**Why this exists:** LLMs can make arithmetic errors. This tool guarantees correct numerical comparison.

---

## Why Keyword Functions?

### The Problem

If you ask an LLM: *"Does this document mention cannabis revenue over 10%?"*

It might respond: *"Yes, the document states 45% of revenue comes from cannabis."*

**But what if the document actually says 25%? Or doesn't mention a percentage at all?**

LLMs can **hallucinate** - confidently state things that aren't in the text.

### The Solution: Grounding Tools

Keyword functions serve as **grounding mechanisms**:

| Benefit | Description |
|---------|-------------|
| **Proof** | Returns ACTUAL text snippets from document |
| **No invention** | Can only return text that EXISTS |
| **Audit trail** | Compliance officers can verify sources |
| **Deterministic** | Same input = same output |

### Analogy

Think of it like a courtroom:
- **LLM** = The lawyer making arguments
- **Keyword tool** = Evidence clerk retrieving actual documents
- The lawyer can interpret, but **cannot fabricate evidence**

### Alternatives

| Option | Pros | Cons | Best For |
|--------|------|------|----------|
| **Pure LLM** | Simple, any language | Hallucination risk | Prototyping |
| **Keyword Search** | Fast, deterministic | English only, exact match | Current implementation |
| **Semantic Search** | Language agnostic, finds synonyms | More complex, added cost | **Recommended upgrade** |
| **Hybrid** | Best coverage | Most complex | Critical decisions |

See `explanations.txt` for detailed analysis of each approach.

---

## Multilingual Support

### Current Limitations

| Component | Limitation |
|-----------|------------|
| `extract_text_evidence()` | English stop words only |
| `classify_scenario()` | English keywords only |
| ReAct prompts | English (LLM understands other languages) |

### What Works Without Changes

- LLM can read/understand documents in **any language**
- `check_threshold()` is language-independent
- Decision tree logic is language-independent

### Recommended Upgrade: Semantic Search

```python
from langchain_openai import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small")

# Embeddings work across languages:
# "cannabis" ≈ "chanvre" (FR) ≈ "Hanf" (DE)
```

---

## Extending the System

### Adding a New Scenario

1. Create `scenarios_1/new_scenario.json`:
```json
{
  "name": "New Compliance Scenario",
  "description": "Description for classification",
  "start": "Q1",
  "questions": {
    "Q1": {
      "text": "First compliance question?",
      "next_if_yes": "Q2",
      "next_if_no": null
    }
  }
}
```

2. (Optional) Add keywords to `tools.py`:
```python
keyword_patterns = {
    "new_scenario": ["keyword1", "keyword2"]
}
```

### Adding a New Tool

1. Define in `tools.py`:
```python
@tool
def my_new_tool(param: str) -> str:
    """Tool description for the LLM.

    Args:
        param: Parameter description
    """
    return "Result"
```

2. Add to tool list:
```python
REACT_TOOLS = [search_evidence, check_threshold, my_new_tool]
```

---

## Troubleshooting

### Common Errors

| Error | Solution |
|-------|----------|
| `AZURE_OPENAI_ENDPOINT not set` | `export AZURE_OPENAI_ENDPOINT="https://..."` |
| `DefaultAzureCredential failed` | Run `az login` |
| `Model deployment not found` | Check `AZURE_OPENAI_DEPLOYMENT` value |
| `ValidationError: must provide azure_endpoint` | Ensure env var is set correctly |

### Debug Output

The console shows detailed progress:
```
[ROUTER]  Scenario classification
[Q1]      Question processing
[REACT]   Agent reasoning steps
[VERDICT] Final determination
```

### Testing LLM Connection

```python
# Add this test in main.py
print("Testing LLM connection...")
response = llm.invoke([{"role": "user", "content": "Say hello"}])
print(f"LLM test: {response.content}")
```

---

## Comparison: Manual ReAct vs create_react_agent

### Manual ReAct (older approach)
```python
# Parse LLM text output for actions
action_match = re.search(r"Action:\s*(\w+)", response)
# Execute tool manually
if action == "search_evidence":
    observation = search_evidence.invoke(action_input)
```

### Native ReAct (this implementation)
```python
# LLM natively calls tools via bind_tools()
react_agent = create_react_agent(llm, tools)
result = react_agent.invoke({"messages": messages})
# Tools called automatically
```

**Benefits of native approach:**
- No text parsing required
- LLM understands tool schemas natively
- Better error handling
- More consistent tool invocations

---

## License

Internal use only. Contact compliance team for distribution.
