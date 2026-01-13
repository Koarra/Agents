# SIAP Compliance Analysis - System Flow

## Architecture Flow Diagram

```
MAIN.PY - SIAP Analysis System
    ↓
Load Document (client_doc_X.txt)
    ↓
    ↓
═══════════════════════════════════════════
STEP 1: ROUTER AGENT (Pre-Screening)
1 LLM Call
═══════════════════════════════════════════
    ↓
Analyze first 2000 chars of document
    ↓
    ↓
    ├─────────────────────────────────┐
    ↓                                 ↓
NO SCENARIOS                    SCENARIOS FOUND
DETECTED                        (e.g., Cannabis + Art)
    ↓                                 ↓
LOW RISK                        Skip Irrelevant Scenarios
Return Early                    (e.g., Commodity Trading)
(No more calls)                       ↓
                                      ↓
                      ═══════════════════════════════════
                      STEP 2: DEEP INVESTIGATION
                      (Only Flagged Scenarios)
                      ═══════════════════════════════════
                                      ↓
                      ┌───────────────┴───────────────┐
                      ↓                               ↓
                  AGENT 1                         AGENT 2
                  Cannabis Business               Art Dealing Professional
                      ↓                               ↓
                  Uses 5 Tools:                   Uses 5 Tools:
                  → SearchDocument                → SearchDocument
                  → ExtractEntities               → ExtractEntities
                  → AnalyzeTransactions           → AnalyzeTransactions
                  → ConsultGuidelines             → ConsultGuidelines
                  → CalculateRiskScore            → CalculateRiskScore
                      ↓                               ↓
                  MEDIUM RISK                     HIGH RISK
                  Licensed cannabis               Professional dealer
                  operations, EDD required        high-value objects, SIAP
                      ↓                               ↓
                      └───────────────┬───────────────┘
                                      ↓
                          Aggregate Results
                          Save to JSON
                          Display Summary
```

---

## Complete Execution Flow

```
START
  ↓
python main.py
  ↓
INITIALIZATION
  ↓
  → Load 3 scenarios from scenarios_1/*.json
  → Initialize Ollama (llama3, temp=0.3)
  → Find all .txt files in documents/
  ↓
FOR EACH DOCUMENT
  ↓
  ├─→ STEP 1: ROUTER SCREENING (1 LLM call)
  │     ↓
  │     → Read first 2000 chars
  │     → Send to router agent
  │     → Get relevant scenarios list
  │     → Display what was flagged
  │     ↓
  ├─→ STEP 2: DEEP INVESTIGATION (per flagged scenario)
  │     ↓
  │     ├─→ IF no scenarios flagged
  │     │     ↓
  │     │     → Mark as LOW RISK
  │     │     → Move to next document
  │     │
  │     └─→ FOR EACH flagged scenario
  │           ↓
  │           → Create ReAct agent with 5 tools
  │           → Agent investigates autonomously
  │           →   ConsultGuidelines
  │           →   SearchDocument
  │           →   ExtractEntities
  │           →   AnalyzeTransactions
  │           →   CalculateRiskScore
  │           → Collect agent's final assessment
  │           ↓
  └─→ AGGREGATE RESULTS for this document
      ↓
FINALIZATION
  ↓
  → Save all results to siap_analysis_results.json
  → Calculate efficiency metrics
  → Display summary
  ↓
END
```

---

## ReAct Agent Investigation Flow

```
Agent Investigation: Cannabis Business Scenario

START
  ↓
THINK → "I need to investigate Cannabis Business risks"
  ↓
ACT → ConsultGuidelines("Cannabis Business")
  ↓
OBSERVE → "Key indicators: direct operations, state licensing, income >10%"
  ↓
THINK → "Let me search for cannabis mentions"
  ↓
ACT → SearchDocument("cannabis")
  ↓
OBSERVE → "Found 8 matches: ...cultivation facilities...retail dispensary..."
  ↓
THINK → "Cannabis confirmed. Check if it's legal"
  ↓
ACT → ExtractEntities("locations")
  ↓
OBSERVE → "Colorado"
  ↓
THINK → "Colorado has legal cannabis. Need to verify licensing"
  ↓
ACT → SearchDocument("license")
  ↓
OBSERVE → "...licensed by Colorado Department of Revenue..."
  ↓
THINK → "Licensed operation. Check if direct or indirect"
  ↓
ACT → SearchDocument("cultivation OR distribution")
  ↓
OBSERVE → "...operates cultivation facilities and retail locations..."
  ↓
THINK → "Direct operations confirmed. This is HIGH involvement"
  ↓
ACT → CalculateRiskScore("cannabis + direct operations + licensed")
  ↓
OBSERVE → "Risk Score: 6/10 - Level: MEDIUM"
  ↓
THINK → "I have enough evidence for assessment"
  ↓
FINAL ANSWER
  ↓
"MEDIUM RISK - Cannabis Business SIAP
Client operates licensed cannabis cultivation and retail (direct operations).
Requires Enhanced Due Diligence (EDD) per SIAP guidelines."
  ↓
END
```

---

## Strategy Comparison

### Naive Approach (NO ROUTER)
```
Document
  ↓
Investigate ALL 3 scenarios
  ↓
  → Art Dealing Investigation (5-10 LLM calls)
  → Cannabis Business Investigation (5-10 LLM calls)
  → Commodity Trading Investigation (5-10 LLM calls)
  ↓
15+ LLM calls per document
Expensive & Slow
```

### Router Agent Pattern (OPTIMIZED)
```
Document
  ↓
Router screens (1 call)
  ↓
Investigate ONLY relevant scenarios
  ↓
  → Cannabis Business Investigation (5-10 LLM calls)
  → Art Dealing Investigation (5-10 LLM calls)
  ↓
5-7 LLM calls per document
Smart & Efficient
```

**Result: 60% cost savings, 67% time savings**

---

## LangGraph Enhancement (Future)

### Current (LangChain Sequential)
```
Document
  ↓
Router (10s)
  ↓
Agent 1 (30s)
  ↓
Agent 2 (30s)
  ↓
Agent 3 (30s)
  ↓
Total: 100s
```

### With LangGraph (Parallel)
```
Document
  ↓
Router (10s)
  ↓
  ├─→ Agent 1 (30s)
  ├─→ Agent 2 (30s)
  └─→ Agent 3 (30s)
  ↓
Total: 40s
```

**Result: 60% faster processing**

---

## Performance Metrics Flow

```
10 Documents Processed
  ↓
Router Pre-Screening
  ↓
  → 30 possible scenarios (10 docs × 3)
  → 8 scenarios flagged as relevant
  → 22 scenarios skipped (73% efficiency)
  ↓
Deep Investigation
  ↓
  → 8 investigations performed
  → Average 5-7 LLM calls each
  → Total ~60 API calls
  ↓
Results
  ↓
  → Processing time: 5 min (vs 15 min naive)
  → Cost: $0.60-$3.00 (vs $1.50-$7.50 naive)
  → Savings: 60% cost, 67% time, 73% efficiency gain
```
