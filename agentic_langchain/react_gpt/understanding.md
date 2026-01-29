# Hybrid Approach Implementation

## Overview

This implementation adds intelligent uncertainty handling to the SIAP compliance detection system. Instead of forcing YES/NO answers when confidence is low, the system now admits when it doesn't know and provides actionable next steps.

---

## The 4 Verdicts

| Verdict | Description | Action |
|---------|-------------|--------|
| `HIT` | Risk detected with high confidence | Escalate/reject client |
| `NO_HIT` | Clean, no risks found | Approve client |
| `MISSING_INFO` | Article lacks critical information | Request more documents |
| `PENDING` | Processing not complete yet | System still working |

---

## Confidence-Based Routing

### Confidence Levels

```
HIGH   (>75%)  : Answer is reliable, continue normally
MEDIUM (50-75%): Answer accepted but flagged for review
LOW    (<50%)  : Cannot determine, mark as MISSING, stop processing
```

### Flow Diagram

```
Question → ReAct Agent → Check Confidence:
  │
  ├── HIGH (>75%): Answer YES/NO → Continue to next question
  │
  ├── MEDIUM (50-75%): Answer YES/NO → Flag as uncertain → Continue
  │
  └── LOW (<50%): Don't answer → Mark as MISSING → Stop → MISSING_INFO verdict
```

---

## Files Modified

### 1. `state.py`

Added:
- `MISSING_INFO` to `Verdict` enum
- `ConfidenceLevel` enum (HIGH, MEDIUM, LOW)
- `MissingFieldRecord` TypedDict for tracking missing fields
- New state fields:
  - `missing_fields`: List of questions that couldn't be answered
  - `uncertain_answers`: Question IDs with MEDIUM confidence
  - `early_termination`: Flag for stopping due to LOW confidence
  - `recommended_action`: What documents to request

### 2. `nodes.py`

Added:
- `get_confidence_level()`: Classifies confidence into HIGH/MEDIUM/LOW
- `get_suggested_documents()`: Maps question types to document requests

Modified:
- `run_react_agent()`: Now returns confidence_level and is_missing flag
- `question_node()`: Implements confidence-based routing:
  - LOW: Adds to missing_fields, triggers early_termination
  - MEDIUM: Adds to uncertain_answers, continues
  - HIGH: Continues normally
- `verdict_node()`: Checks for MISSING_INFO first, includes recommended actions
- `should_continue_questions()`: Handles early termination routing

### 3. `graph.py`

Modified:
- `process_article()`: Prints missing fields and recommended actions
- `process_batch()`: Counts all verdict types, lists documents needed

---

## Example Outputs

### Scenario 1: Complete Information, Clean Client
```
Verdict: NO_HIT
Reason: No compliance risks identified
Recommended Action: Approve - proceed with onboarding
```

### Scenario 2: Complete Information, Risk Found
```
Verdict: HIT
Reason: Q4: Does client hold executive position...
Recommended Action: Escalate to compliance team for manual review
```

### Scenario 3: Incomplete Article (NEW)
```
Verdict: MISSING_INFO
Reason: Cannot determine compliance status. Missing: Q4: Is client in direct operations...
Missing Fields: 1
Recommended Action: Please provide: business structure documents, operational agreements
```

### Scenario 4: Partial Information with Uncertainty
```
Verdict: NO_HIT
Reason: No compliance risks identified [Note: 2 answer(s) had medium confidence]
Recommended Action: Approve - proceed with onboarding [Review flagged questions: Q3, Q5]
```

---

## Document Suggestion Mapping

When a question can't be answered, the system suggests relevant documents:

| Question Contains | Suggested Documents |
|-------------------|---------------------|
| "income" | income statements, tax returns, financial records |
| "percentage" | ownership documents, shareholder registry, tax returns |
| "executive" | corporate filings, org charts, employment contracts |
| "board" | corporate filings, board minutes, annual reports |
| "ownership" | shareholder registry, ownership certificates, cap table |
| "direct" | business structure documents, operational agreements |
| "illegal" | business licenses, regulatory filings, compliance certificates |
| "hemp" | product documentation, regulatory permits, lab certificates |
| "cannabis" | business licenses, state permits, operational documents |

---

## Batch Processing Summary

The batch summary now shows:
```
BATCH SUMMARY
=============
Total: 100
HITs: 15 (escalate/reject)
NO_HITs: 60 (approve)
MISSING_INFO: 20 (request documents)
PENDING/Errors: 5
Flagged for review: 8 (medium confidence answers)

MISSING INFO CASES - Documents Needed:
  - article_001: Please provide: income statements, tax returns
  - article_007: Please provide: business licenses, state permits
```

---

## Benefits

1. **Better Risk Management**
   - Don't approve clients when uncertain (avoid risk)
   - Don't reject clients when uncertain (avoid false alarms)
   - Clear guidance on what documents to request

2. **Operational Efficiency**
   - Only review MISSING_INFO cases manually
   - Know exactly what documents to request
   - Audit trail shows why system couldn't decide

3. **Reduced Errors**
   - No more guessing on low confidence answers
   - Medium confidence answers flagged for human review
   - Early termination prevents cascading errors
