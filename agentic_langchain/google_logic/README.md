# SIAP Compliance Detection System

LangGraph application that processes articles to detect compliance "Hits" based on decision tree scenarios.

## How It Works

This system uses **LangGraph** for workflow orchestration and implements the **ReAct (Reasoning + Acting)** pattern for intelligent question answering.

### Libraries Used
- **LangGraph** (`langgraph`) - Graph-based workflow with conditional edges for decision tree traversal
- **LangChain Ollama** (`langchain_ollama`) - LLM integration with local Ollama models
- **LangChain Core** (`langchain_core`) - Base tools and prompts

### ReAct Pattern

ReAct combines reasoning and acting in an iterative loop. For each compliance question, the LLM:

```
Thought: I need to find evidence about cannabis involvement
Action: search_evidence
Input: "cannabis business cultivation"
Observation: EVIDENCE FOUND: "vertically integrated cannabis business..."

Thought: Found cannabis evidence. Now checking executive role...
Action: search_evidence
Input: "executive management CEO"
Observation: EVIDENCE FOUND: "CEO holds executive management position"

Thought: Evidence confirms both. Ready to answer.
Action: answer
Input: YES - Direct cannabis involvement with executive position
```

The ReAct loop is implemented manually in `nodes.py` (`run_react_loop` function) with two tools:
- `search_evidence(query)` - Extract relevant quotes from the article
- `check_threshold(value, limit)` - Validate numerical compliance (e.g., ownership %)

### Pipeline Flow

```
Article → Router → Question Node (ReAct loop) → ... → Verdict → HIT/NO_HIT
                         ↑_________|
                        (decision tree traversal)
```

1. **Router**: Classifies article to a scenario (Cannabis, Art Dealing, etc.)
2. **Question Node**: Uses ReAct to answer each question with evidence
3. **Conditional Edges**: Skip logic based on YES/NO answers (`next_if_yes`/`next_if_no`)
4. **Verdict**: Aggregates answers to determine final HIT or NO_HIT

## Overview

This system autonomously determines HIT or NO_HIT verdicts by:
1. **Classifying** articles to compliance scenarios (Cannabis, Art Dealing, etc.)
2. **Traversing** decision trees with conditional skip logic
3. **Extracting** evidence using ReAct pattern with tools
4. **Validating** numerical thresholds with Python logic (no LLM hallucination)
5. **Reaching** a terminal verdict without human intervention

## Architecture

### State Management (`state.py`)

```python
class ComplianceState(TypedDict):
    article_id: str
    article_text: str
    scenario_id: Optional[str]      # e.g., "cannabis_business"
    scenario_name: Optional[str]
    current_node: str               # Current question (Q1, Q2, etc.)
    decision_tree: dict             # {Q1: {text, next_if_yes, next_if_no}, ...}
    answers: dict[str, AnswerRecord]
    final_verdict: Verdict          # HIT or NO_HIT
    verdict_reason: str
    risk_score: float
```

### Graph Structure (`graph.py`)

```
START → ROUTER → QUESTION (loop) → VERDICT → END
              ↓         ↑
         [no match]     │
              ↓         │
            END    (skip logic)
```

**Key Feature: Conditional Edges for Skip Logic**

```python
graph.add_conditional_edges(
    "question",
    should_continue_questions,
    {
        "question": "question",  # Loop to next question
        "verdict": "verdict",    # Tree complete
        "end": END               # Early termination
    }
)
```

The `question` node updates `current_node` based on the answer:
- YES → follow `next_if_yes` (may skip questions)
- NO → follow `next_if_no` (may skip questions)

### ReAct Tools (`tools.py`)

| Function | Purpose |
|----------|---------|
| `extract_text_evidence(query, text)` | Find supporting quotes from article |
| `validate_threshold(value, limit)` | Check numerical compliance (e.g., ownership %) |

```python
# Example: Extract evidence
result = extract_text_evidence("Is client in cannabis business?", article)
# Returns: {found: True, evidence: [...], confidence: 0.85}

# Example: Validate threshold
result = validate_threshold(60.0, 25.0)  # 60% ownership vs 25% limit
# Returns: {exceeds_threshold: True, explanation: "60.0 >= 25.0 = True"}
```

### Nodes (`nodes.py`)

| Node | Function |
|------|----------|
| `router_node` | Classify article to scenario |
| `question_node` | Answer current question with ReAct tools |
| `verdict_node` | Determine final HIT/NO_HIT |

## Scenarios (from `scenarios_1/`)

Decision trees are loaded from JSON files:

| Scenario | File | Questions |
|----------|------|-----------|
| Cannabis Business | `cannabis_business.json` | 8 questions |
| Professional Art Dealing | `art_dealing_professional.json` | 11 questions |
| Commodity Trading | `commodity_trading.json` | 10 questions |

### Cannabis Business Example

```
Q1: Is client involved in cannabis business?
    YES → Q2
    NO  → END (NO_HIT)

Q2: Is activity illegal in country of operation?
    YES → END (HIT - illegal)
    NO  → Q3

Q3: Is it solely hemp business?
    YES → END (NO_HIT - exempt)
    NO  → Q4

Q4: Direct cannabis operations?
    YES → Q5 (check executive role)
    NO  → Q7 (check indirect involvement)

Q5: Executive/board position?
    YES → END (HIT - high risk)
    NO  → Q6

Q6: Income > 10%?
    YES → END (HIT - threshold)
    NO  → END (NO_HIT)
```

## Installation

```bash
pip install langgraph langchain-ollama langchain-core

# Ensure Ollama is running
ollama pull llama3.1
```

## Usage

### Command Line

```bash
cd google_logic
python main.py
```

### Programmatic

```python
from langchain_ollama import OllamaLLM
from graph import process_article

llm = OllamaLLM(model="llama3.1", temperature=0.3)

result = process_article(
    article_id="test_001",
    article_text="Article about cannabis dispensary...",
    llm=llm
)

print(f"Verdict: {result['final_verdict']}")  # HIT or NO_HIT
print(f"Reason: {result['verdict_reason']}")
```

### Batch Processing

```python
from graph import process_batch

articles = [
    {"id": "art_1", "text": "Cannabis article..."},
    {"id": "art_2", "text": "Art dealer article..."},
]

results = process_batch(articles, llm)
```

## Example Output

```
============================================================
PROCESSING: cannabis_demo
============================================================

[ROUTER] Classified as: Cannabis Business
[ROUTER] Starting at: Q1

[Q1] Is the client involved in a cannabis-related business?
[Q1] Answer: YES (confidence: 0.85)
[Q1] Next: Q2

[Q2] Is the cannabis-related activity illegal?
[Q2] Answer: NO (confidence: 0.90)
[Q2] Next: Q3

[Q3] Does the client activity solely involve hemp business?
[Q3] Answer: NO (confidence: 0.75)
[Q3] Next: Q4

[Q4] Does the client activity involve direct cannabis operations?
[Q4] Answer: YES (confidence: 0.88)
[Q4] Next: Q5

[Q5] Does the client hold executive/board position?
[Q5] Answer: YES (confidence: 0.92)
[Q5] Terminal node reached

==================================================
VERDICT DETERMINATION
==================================================
[VERDICT] HIT - Risk triggers found
  - Q5: Does the client hold an executive management or board of...
[VERDICT] Risk Score: 0.30

------------------------------------------------------------
SUMMARY
------------------------------------------------------------
Article: cannabis_demo
Scenario: Cannabis Business
Verdict: Verdict.HIT
Risk Score: 0.30
Reason: Q5: Does the client hold an executive management or board of...
```

## Files

| File | Purpose |
|------|---------|
| `state.py` | ComplianceState TypedDict definition |
| `tools.py` | ReAct tools (extract_text_evidence, validate_threshold) |
| `nodes.py` | Graph node functions (router, question, verdict) |
| `graph.py` | LangGraph workflow with conditional edges |
| `main.py` | Entry point with demo articles |
| `../scenarios_1/` | Scenario JSON files (decision trees) |

## Key Design Principles

1. **Autonomous Execution**: No human intervention required
2. **Skip Logic**: Conditional edges based on yes/no answers
3. **ReAct Tools**: Evidence extraction and threshold validation
4. **Conservative Bias**: Unclear answers default to YES (flag risk)
5. **Auditability**: All answers and evidence recorded

## Adding New Scenarios

Create a JSON file in `scenarios_1/`:

```json
{
  "name": "New Scenario",
  "description": "Description...",
  "start": "Q1",
  "questions": {
    "Q1": {
      "text": "First question?",
      "next_if_yes": "Q2",
      "next_if_no": null
    },
    "Q2": {
      "text": "Second question?",
      "next_if_yes": null,
      "next_if_no": "Q3"
    },
    "Q3": {
      "text": "Third question?",
      "next_if_yes": null,
      "next_if_no": null
    }
  }
}
```

The system will automatically load and use the new scenario.
