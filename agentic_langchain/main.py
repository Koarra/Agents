"""
Main Compliance Analysis System
Optimized agentic analyzer with Router Agent for 3 SIAP scenarios:
- Professional Art Dealing
- Cannabis Business
- Commodity Trading
"""

import os
import json
import re
from typing import Dict, List
from glob import glob
from langchain_community.llms import Ollama
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate


class ComplianceTools:
    """Tools for the compliance agent to use during investigation."""

    def __init__(self, document_text: str, guidelines: Dict):
        self.document_text = document_text
        self.guidelines = guidelines

    def search_document(self, query: str) -> str:
        """Search for keywords or patterns in the document."""
        lines = self.document_text.split('\n')
        matches = []

        query_lower = query.lower()
        for i, line in enumerate(lines):
            if query_lower in line.lower():
                context_start = max(0, i-1)
                context_end = min(len(lines), i+2)
                context = ' '.join(lines[context_start:context_end])
                matches.append(context.strip())

        if matches:
            result = f"Found {len(matches)} matches for '{query}':\n"
            for idx, match in enumerate(matches[:5], 1):
                result += f"{idx}. ...{match[:200]}...\n"
            return result
        else:
            return f"No matches found for '{query}'"

    def extract_entities(self, entity_type: str) -> str:
        """Extract specific types of entities from the document."""
        entities = []

        if entity_type.lower() == "amounts":
            pattern = r'\$[\d,]+(?:\.\d{2})?(?:K|M|million|thousand)?'
            entities = re.findall(pattern, self.document_text)

        elif entity_type.lower() == "companies":
            patterns = [
                r'\b([A-Z][A-Za-z\s&]+(?:LLC|Inc\.|Corporation|Corp\.|Ltd\.|Co\.|Group|International|Holdings|Partners))\b',
                r'\b([A-Z][A-Za-z\s&]+(?:Trust|Investments|Capital|Services|Solutions))\b'
            ]
            for pattern in patterns:
                entities.extend(re.findall(pattern, self.document_text))

        elif entity_type.lower() == "countries" or entity_type.lower() == "locations":
            keywords = ['Iran', 'North Korea', 'Russia', 'Syria', 'Cuba', 'Venezuela',
                       'Cayman Islands', 'Panama', 'Cyprus', 'British Virgin Islands',
                       'Switzerland', 'Luxembourg', 'offshore', 'overseas', 'Colorado',
                       'California', 'Oregon', 'Washington']
            for keyword in keywords:
                if keyword.lower() in self.document_text.lower():
                    entities.append(keyword)

        elif entity_type.lower() == "people":
            pattern = r'\b([A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b'
            entities = re.findall(pattern, self.document_text)

        entities = list(set(entities))

        if entities:
            return f"Extracted {len(entities)} {entity_type}: {', '.join(entities[:10])}"
        else:
            return f"No {entity_type} found"

    def analyze_transactions(self, dummy_input: str = "") -> str:
        """Analyze transaction patterns for suspicious activity."""
        analysis = []

        if re.search(r'\$9[,\d]{3}', self.document_text):
            analysis.append("⚠️ Possible structuring detected: Transactions just under $10,000 threshold")

        if 'cash' in self.document_text.lower() and re.search(r'\$\d{2,3},\d{3}', self.document_text):
            analysis.append("⚠️ Large cash transactions mentioned")

        keywords = ['rapid', 'quick', 'immediately', 'same day', 'within days', 'layering']
        if any(kw in self.document_text.lower() for kw in keywords):
            analysis.append("⚠️ Rapid fund movement indicators present")

        if any(word in self.document_text.lower() for word in ['shell', 'offshore', 'intermediary', 'front company']):
            analysis.append("⚠️ Complex ownership or intermediary structures mentioned")

        if self.document_text.lower().count('cash') > 5:
            analysis.append("⚠️ Cash-intensive business model indicated")

        if analysis:
            return "Transaction Pattern Analysis:\n" + "\n".join(analysis)
        else:
            return "No suspicious transaction patterns detected"

    def consult_guidelines(self, scenario: str) -> str:
        """Consult compliance guidelines for a specific scenario."""
        if scenario not in self.guidelines:
            available = list(self.guidelines.keys())
            return f"Scenario '{scenario}' not found. Available scenarios: {', '.join(available)}"

        scenario_data = self.guidelines[scenario]
        result = f"Compliance Guidelines for {scenario}:\n"
        result += f"Description: {scenario_data.get('description', 'N/A')}\n\n"
        result += "Key Indicators to Investigate:\n"

        questions = scenario_data.get('questions', {})
        for q_id, q_data in list(questions.items())[:3]:
            result += f"- {q_data['text']}\n"

        return result

    def calculate_risk_score(self, evidence: str) -> str:
        """Calculate a risk score based on collected evidence."""
        risk_score = 0

        red_flags = ['structuring', 'layering', 'offshore', 'shell company', 'anonymous',
                    'cash', 'unreported', 'sanctions', 'disguise', 'front company',
                    'tax haven', 'evade', 'conceal', 'illicit', 'cannabis', 'marijuana',
                    'art', 'antiques', 'commodity', 'trading', 'forgery', 'smuggling']

        for flag in red_flags:
            if flag in evidence.lower():
                risk_score += 1

        if risk_score >= 5:
            level = "HIGH"
        elif risk_score >= 3:
            level = "MEDIUM"
        else:
            level = "LOW"

        return f"Risk Score: {risk_score}/10 - Level: {level}"


def create_router_agent(document_text: str, scenarios: List[str], llm: Ollama) -> Dict:
    """
    Router agent that identifies relevant scenarios for investigation.
    Only includes the 3 SIAP scenarios.
    """

    scenario_descriptions = {
        "Professional Art Dealing": "Professional sale, brokerage, auction, or storage of fine art, antiques, antiquities, artefacts; rare/high-value objects, cultural heritage, forgery/looting risks",
        "Cannabis Business": "Direct/indirect cannabis operations, state licensing, executive management in cannabis, income from cannabis activities, cultivation, distribution",
        "Commodity Trading": "Trading of energy, metals, agriculturals, primary/secondary commodities; excludes mining/extraction, B2C retail, financial instruments only"
    }

    # Use first 2000 characters for quick assessment
    doc_preview = document_text[:2000]

    prompt = f"""You are a compliance triage specialist. Your job is to quickly identify which risk scenarios are potentially relevant to this document.

Document Preview (first 2000 characters):
{doc_preview}

Available Risk Scenarios:
{json.dumps(scenario_descriptions, indent=2)}

Analyze the document and determine which scenarios warrant deeper investigation.

IMPORTANT:
- Only flag scenarios with clear indicators present in the document
- If a scenario seems completely unrelated, DO NOT flag it
- Be conservative - it's better to flag something questionable than miss a risk
- Provide brief reasoning for each flagged scenario

Return your answer in this EXACT JSON format:
{{
  "relevant_scenarios": ["Scenario Name 1", "Scenario Name 2"],
  "reasoning": {{
    "Scenario Name 1": "Brief explanation why this is relevant",
    "Scenario Name 2": "Brief explanation why this is relevant"
  }}
}}

If NO scenarios are relevant, return:
{{
  "relevant_scenarios": [],
  "reasoning": {{}}
}}

Your JSON response:"""

    try:
        response = llm.invoke(prompt)
        # Extract JSON from response
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            result = json.loads(json_str)
            return result
        else:
            # Fallback: investigate all scenarios if parsing fails
            print("⚠️ Router parsing failed, defaulting to all scenarios")
            return {
                "relevant_scenarios": list(scenarios),
                "reasoning": {s: "Router parsing failed, investigating all" for s in scenarios}
            }
    except Exception as e:
        print(f"⚠️ Router error: {e}, defaulting to all scenarios")
        return {
            "relevant_scenarios": list(scenarios),
            "reasoning": {s: f"Router error: {e}" for s in scenarios}
        }


def create_compliance_agent(document_text: str, guidelines: Dict, llm: Ollama) -> AgentExecutor:
    """Create a ReAct agent for compliance investigation."""

    compliance_tools = ComplianceTools(document_text, guidelines)

    tools = [
        Tool(
            name="SearchDocument",
            func=compliance_tools.search_document,
            description="Search for keywords or patterns in the document. Input should be a search query like 'cannabis cultivation' or 'art auction'."
        ),
        Tool(
            name="ExtractEntities",
            func=compliance_tools.extract_entities,
            description="Extract entities from document. Input should be entity type: 'amounts', 'companies', 'locations', 'people', or 'countries'."
        ),
        Tool(
            name="AnalyzeTransactions",
            func=compliance_tools.analyze_transactions,
            description="Analyze transaction patterns for suspicious activity. Input 'analyze' to use this tool."
        ),
        Tool(
            name="ConsultGuidelines",
            func=compliance_tools.consult_guidelines,
            description="Consult compliance guidelines for a specific risk scenario. Input should be scenario name: 'Professional Art Dealing', 'Cannabis Business', or 'Commodity Trading'."
        ),
        Tool(
            name="CalculateRiskScore",
            func=compliance_tools.calculate_risk_score,
            description="Calculate overall risk score based on evidence collected. Input should be a summary of the evidence found."
        )
    ]

    template = """You are an expert compliance officer investigating potential SIAP (Significant Industry Activity Profile) risks.
You have access to tools to analyze the document systematically.

IMPORTANT: Follow this investigation process:
1. Start by consulting guidelines for the risk scenario you're investigating
2. Search the document for key indicators mentioned in guidelines
3. Extract relevant entities (amounts, companies, locations)
4. Analyze transaction patterns if applicable
5. Calculate risk score based on evidence
6. Provide final assessment with evidence

You have access to the following tools:

{tools}

Use the following format:

Question: the risk scenario you must investigate
Thought: think about what to investigate first
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now have enough evidence to make a final assessment
Final Answer: A detailed risk assessment with evidence, risk level (LOW/MEDIUM/HIGH), and recommendations

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
            "tool_names": ", ".join([tool.name for tool in tools])
        }
    )

    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True
    )

    return agent_executor


def load_risk_scenarios(scenarios_dir: str = "scenarios_1") -> Dict:
    """Load all risk scenario JSON files from scenarios_1 directory."""
    scenarios = {}

    json_files = glob(os.path.join(scenarios_dir, "*.json"))

    if not json_files:
        raise FileNotFoundError(f"No scenario JSON files found in {scenarios_dir}/")

    for json_file in sorted(json_files):
        with open(json_file, 'r', encoding='utf-8') as f:
            scenario_data = json.load(f)
            scenario_name = scenario_data.get("name")
            if scenario_name:
                scenarios[scenario_name] = scenario_data

    return scenarios


def analyze_document_optimized(document_path: str, llm: Ollama, guidelines: Dict) -> Dict:
    """
    Optimized document analysis using Router Agent.
    Only analyzes the 3 SIAP scenarios.
    """

    print(f"\n{'='*60}")
    print(f"SIAP ANALYSIS: {os.path.basename(document_path)}")
    print(f"{'='*60}\n")

    # Load document
    with open(document_path, 'r', encoding='utf-8') as f:
        document_text = f.read()

    # Step 1: Router Agent Pre-Screening
    print("🔍 STEP 1: Router Agent Pre-Screening")
    print("-" * 60)

    router_result = create_router_agent(document_text, list(guidelines.keys()), llm)
    relevant_scenarios = router_result.get("relevant_scenarios", [])
    reasoning = router_result.get("reasoning", {})

    print(f"\n✅ Router identified {len(relevant_scenarios)} relevant scenario(s):")
    if relevant_scenarios:
        for scenario in relevant_scenarios:
            reason = reasoning.get(scenario, "No reason provided")
            print(f"   • {scenario}: {reason}")
    else:
        print("   • No SIAP scenarios detected - document appears low risk")

    print(f"\n⏭️  Skipping {len(guidelines) - len(relevant_scenarios)} irrelevant scenario(s)")

    results = {
        "document": os.path.basename(document_path),
        "router_screening": {
            "total_scenarios": len(guidelines),
            "relevant_scenarios": len(relevant_scenarios),
            "scenarios_flagged": relevant_scenarios,
            "screening_reasoning": reasoning
        },
        "scenarios": []
    }

    # If no relevant scenarios, return early
    if not relevant_scenarios:
        results["overall_assessment"] = "LOW RISK - No SIAP indicators detected"
        return results

    # Step 2: Deep Investigation for Relevant Scenarios Only
    print(f"\n{'='*60}")
    print(f"🔬 STEP 2: Deep Investigation ({len(relevant_scenarios)} scenario(s))")
    print(f"{'='*60}\n")

    for scenario in relevant_scenarios:
        print(f"\n🔍 Investigating: {scenario}")
        print("-" * 60)

        # Create agent for this investigation
        agent = create_compliance_agent(document_text, guidelines, llm)

        # Run agent investigation
        try:
            result = agent.invoke({
                "input": f"Investigate this document for {scenario} risks. Be thorough and use all available tools to gather evidence."
            })

            scenario_result = {
                "scenario": scenario,
                "router_reason": reasoning.get(scenario, ""),
                "agent_output": result.get("output", "No output"),
                "investigation_complete": True
            }
        except Exception as e:
            print(f"⚠️ Error during investigation: {e}")
            scenario_result = {
                "scenario": scenario,
                "router_reason": reasoning.get(scenario, ""),
                "agent_output": f"Investigation failed: {str(e)}",
                "investigation_complete": False
            }

        results["scenarios"].append(scenario_result)

    return results


def main():
    """Main function for SIAP compliance analysis."""

    print("=" * 60)
    print("SIAP COMPLIANCE ANALYSIS SYSTEM")
    print("Router Agent with 3 Scenarios")
    print("=" * 60)

    # Load guidelines from scenarios_1 folder
    print("\nLoading SIAP scenarios from scenarios_1/...")
    try:
        guidelines = load_risk_scenarios("scenarios_1")
        print(f"✅ Loaded {len(guidelines)} scenarios:")
        for scenario_name in guidelines.keys():
            print(f"   • {scenario_name}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure scenarios_1 folder exists with the 3 scenario JSON files.")
        return

    # Initialize Ollama
    print("\n🚀 Initializing Ollama (llama3)...")
    llm = Ollama(model="llama3", temperature=0.3)

    # Get documents
    documents_dir = "documents"
    if not os.path.exists(documents_dir):
        print(f"❌ Documents directory '{documents_dir}/' not found")
        return

    doc_files = sorted([f for f in os.listdir(documents_dir) if f.endswith('.txt')])

    if not doc_files:
        print(f"❌ No .txt files found in {documents_dir}/")
        return

    print(f"📄 Found {len(doc_files)} documents to analyze\n")

    # Analyze documents
    all_results = []

    for doc_file in doc_files:
        doc_path = os.path.join(documents_dir, doc_file)
        result = analyze_document_optimized(doc_path, llm, guidelines)
        all_results.append(result)

    # Save results
    output_file = "siap_analysis_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\n📊 Results saved to: {output_file}")

    # Print efficiency summary
    print("\n" + "=" * 60)
    print("EFFICIENCY SUMMARY")
    print("=" * 60)

    total_scenarios_available = len(guidelines) * len(doc_files)
    total_scenarios_investigated = sum(len(r["scenarios"]) for r in all_results)
    scenarios_skipped = total_scenarios_available - total_scenarios_investigated
    efficiency = (scenarios_skipped / total_scenarios_available * 100) if total_scenarios_available > 0 else 0

    print(f"Documents analyzed: {len(doc_files)}")
    print(f"Total scenarios available: {total_scenarios_available}")
    print(f"Scenarios investigated: {total_scenarios_investigated}")
    print(f"Scenarios skipped by router: {scenarios_skipped}")
    print(f"Efficiency gain: {efficiency:.1f}%")
    print(f"\n💰 Estimated API call reduction: ~{efficiency * 0.85:.0f}%")

    # Print summary by document
    print("\n" + "=" * 60)
    print("RESULTS BY DOCUMENT")
    print("=" * 60)

    for result in all_results:
        doc_name = result['document']
        flagged = result['router_screening']['scenarios_flagged']
        if flagged:
            print(f"\n📄 {doc_name}")
            for scenario in result['scenarios']:
                print(f"   • {scenario['scenario']}: Investigated")
        else:
            print(f"\n📄 {doc_name}: ✅ No SIAP scenarios detected")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
