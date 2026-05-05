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

    def fabricate(self, wf_name: str, description: str, data_sources: List[str], user_mandatory_agents: List[str], existing_agents_str: str) -> FabricatorOutput:
        
        formatted_system = self.system_prompt.format(
            available_tools=", ".join(self.available_tools),
            user_mandatory_agents=", ".join(user_mandatory_agents) if user_mandatory_agents else "None",
            existing_agents=existing_agents_str
        )
        
        prompt = f"""
        {formatted_system}
        
        WORKFLOW TO FABRICATE:
        Name: {wf_name}
        Description: {description}
        Required Database Tables: {", ".join(data_sources)}
        
        CRITICAL INSTRUCTIONS:
        1. First, write your 'thought_process'.
        2. Second, generate the 'domain_agents' array. To prevent formatting timeouts, keep the 'persona' field for each new agent concise (3-4 sentences maximum) and strictly focused on its domain and tool usage rules.
        3. Third, you MUST populate the 'final_resolved_agents' array with the exact string names of ALL agents required (e.g., ["client_profile_domain_agent", "portfolio_domain_agent", "crm_activities_domain_agent", ...]).
        4. ABSOLUTE MANDATE: You are strictly forbidden from returning empty arrays for 'domain_agents' or 'final_resolved_agents' if your thought process identified gaps.
        """
        
        print("🧠 Fabricator is reasoning about domain boundaries...")
        result = self.structured_llm.invoke(prompt)
        
        print(f"\n💭 Fabricator Thought Process:\n{result.thought_process}\n")
        
        if not result.final_resolved_agents:
            print("⚠️ Notice: LLM returned empty roster. Enforcing mandatory agents as fallback...")
            result.final_resolved_agents = user_mandatory_agents
            
        return result