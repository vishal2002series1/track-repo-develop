# src/agents/fabricator.py
import os
import json
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# 🟢 AZURE MIGRATION
from langchain_openai import AzureChatOpenAI

load_dotenv()

# Schema for the LLM to strictly output
class AgentBlueprint(BaseModel):
    name: str = Field(description="The exact variable name for the agent (e.g., 'meeting_prep_domain_agent'). Must end in '_domain_agent'")
    routing_description: str = Field(description="A 1-2 sentence description for the Supervisor to know when to route to this agent.")
    persona: str = Field(description="The system prompt for the agent. Must include the exact tool names it should use.")
    authorized_tools: List[str] = Field(default_factory=list, description="A list of string names of the tools this agent has permission to use.")

class FabricatorOutput(BaseModel):
    thought_process: str = Field(description="Analyze the required data sources. Explain exactly which existing agents cover which sources, and identify the gaps that require brand new agents.")
    domain_agents: List[AgentBlueprint] = Field(default_factory=list, description="The list of newly fabricated agents to fill the gaps.")
    final_resolved_agents: List[str] = Field(default_factory=list, description="The complete list of all agent names required for this workflow, including new ones, reused existing ones, and the mandatory ones.")
    
    # 🟢 EPIC 2 ADDITION: Workflow-Level Defensive Prompts
    proposed_supervisor_rules: str = Field(
        default="", 
        description="Custom routing rules for the Supervisor. Tell it exactly WHEN to route to which agent based on the test cases."
    )
    proposed_synthesizer_persona: str = Field(
        default="", 
        description="Custom persona for the Synthesizer. Give it a specific voice or specific Markdown formatting instructions based on the workflow intent."
    )
class DomainFabricator:
    def __init__(self):
        # 🟢 AZURE MIGRATION: Dynamically pull credentials
        api_key = os.getenv("API_KEYS")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("OPENAI_API_VERSION")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.4")
        
        self.llm = AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            azure_deployment=deployment_name,
            temperature=0.0,
            max_tokens=8000
        )
        self.structured_llm = self.llm.with_structured_output(FabricatorOutput)
        
        # 🟢 MOVED: Path points to the new Git-tracked config directory
        prompt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/prompt_library.json'))
        with open(prompt_path, 'r') as f:
            prompts = json.load(f)
        self.system_prompt = prompts.get("fabricator_system_prompt", "")
        
        self.available_tools = [
            "execute_sql",
            "get_database_schema",
            "get_current_time"
        ]

    
    # 🟢 FIX: Updated signature to explicitly inject tools and agents into the prompt string
    # 🟢 FIX: Aligned with core graph engine to prevent duplicate synthesizers and infinite loops
    def fabricate(self, wf_name: str, description: str, test_cases: List[Dict], available_tools: List[Dict], existing_agents: List[Dict], user_mandatory_agents: List[str]) -> FabricatorOutput:
        
        prompt = f"""
        {self.system_prompt}
        
        =========================================
        FACTORY INVENTORY (CRITICAL CONTEXT):
        =========================================
        AVAILABLE TOOLS IN FACTORY INVENTORY:
        {json.dumps(available_tools, indent=2)}
        
        EXISTING AGENTS IN FACTORY INVENTORY:
        {json.dumps(existing_agents, indent=2)}
        
        USER MANDATORY AGENTS:
        {", ".join(user_mandatory_agents) if user_mandatory_agents else "None"}
        =========================================
        
        WORKFLOW TO FABRICATE:
        Name: {wf_name}
        Description: {description}
        
        GOLDEN TEST DATASET (Questions this workflow MUST be able to answer):
        {json.dumps(test_cases, indent=2)}
        
        CRITICAL INSTRUCTIONS:
        1. THOUGHT PROCESS: Analyze the Golden Test Dataset. Determine exactly what data is needed to answer these questions. Check the EXISTING AGENTS IN FACTORY INVENTORY to see if they can fetch this data using their tools. Identify any capability gaps.
        
        2. DOMAIN AGENTS: Generate blueprints for ONLY the BRAND NEW agents required to fill the gaps. Keep personas concise and focused on passing the tests.
           - 🛑 RULE: You MUST use Defensive Prompting. Give the agent strict step-by-step instructions. Tell the agent exactly what it is NOT allowed to do.
           - 🛑 ANTI-PATTERN RULE: DO NOT create any agent whose job is to "synthesize", "summarize", or "format the final report". The graph engine already has a built-in global synthesizer node.
           
        3. FINAL ROSTER: Populate 'final_resolved_agents' with the names of ALL agents required (your new ones + the existing ones you are reusing).
           - 🛑 EXCLUSION RULE: You MUST explicitly EXCLUDE 'synthesis_agent' or any similar reporting agent from this roster.
        
        4. ZERO-HALLUCINATION TOOL RULE: When assigning 'authorized_tools', you are strictly FORBIDDEN from making up tool names. You MUST ONLY use the exact string names provided in the 'AVAILABLE TOOLS IN FACTORY INVENTORY' list above.
        
        5. 🟢 WORKFLOW PROMPTS (THE ORCHESTRATION LAYER): 
           - 'proposed_supervisor_rules': Write strict, clinical routing rules. Explicitly instruct the Supervisor: "Once the domain agents have successfully gathered the required information into the conversation history, you MUST route to the exact word 'synthesizer' to conclude the workflow." It must NEVER loop back to a worker agent if the data is already present.
           - 'proposed_synthesizer_persona': Command it to act as an Executive Synthesizer. Strictly forbid hallucinating data outside the conversation history. Require professional formatting (Markdown, headers, bullet points, or tables) to make the final output pristine.
        """
        
        print("🧠 Fabricator is reasoning about domain boundaries...")
        result = self.structured_llm.invoke(prompt)
        
        print(f"\n💭 Fabricator Thought Process:\n{result.thought_process}\n")
        
        if not result.final_resolved_agents and user_mandatory_agents:
            print("⚠️ Notice: LLM returned empty roster. Enforcing mandatory agents as fallback...")
            result.final_resolved_agents = user_mandatory_agents
            
        return result