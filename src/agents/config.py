# src/agents/config.py
import json
import os
from pydantic import BaseModel, Field
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AgentConfig(BaseModel):
    name: str = Field(..., description="The unique name of the agent")
    routing_description: str = Field(..., description="Short 1-liner for the Supervisor to understand when to route here")
    persona: str = Field(..., description="The full system prompt instructing the agent how to behave")
    # Dynamically pull the model from .env, defaulting to gpt-5.4
    model_id: str = Field(default_factory=lambda: os.getenv("MODEL_ID", "gpt-5.4"))
    authorized_tools: List[str] = Field(default_factory=list)
    temperature: float = Field(default=0.0)

class RegistryManager:
    def __init__(self):
        # Resolve the path to where the JSON will live
        base_dir = os.path.dirname(__file__)
        self.registry_path = os.path.abspath(os.path.join(base_dir, '../../data_local/agent_registry.json'))
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        
        # Load agents
        self.agents: Dict[str, AgentConfig] = self._load_registry()

    def _load_registry(self) -> Dict[str, AgentConfig]:
        # 1. If the JSON file already exists, load from it (Dynamic Mode)
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
                return {name: AgentConfig(**config) for name, config in data.items()}
        
        # 2. If it DOES NOT exist, Auto-Migrate defaults!
        print("Initializing fresh Agent Registry JSON with default AEON agents...")
        default_agents = self._get_default_agents()
        
        # Save them to the new JSON file so it's there for next time
        serializable_data = {name: agent.model_dump() for name, agent in default_agents.items()}
        with open(self.registry_path, 'w') as f:
            json.dump(serializable_data, f, indent=4)
            
        return default_agents

    def save_agent(self, new_agent: AgentConfig):
        """Adds a new agent to the dictionary and persists it to the JSON file."""
        self.agents[new_agent.name] = new_agent
        serializable_data = {name: agent.model_dump() for name, agent in self.agents.items()}
        
        with open(self.registry_path, 'w') as f:
            json.dump(serializable_data, f, indent=4)
        print(f"✅ Successfully committed '{new_agent.name}' to the permanent Agent Registry.")

    def _get_default_agents(self) -> Dict[str, AgentConfig]:
        """Your existing agents, safely preserved here as the baseline template."""
        return {
            "client_info_agent": AgentConfig(
                name="client_info_agent",
                routing_description="Retrieves demographic facts, AUM, and risk profiles.",
                persona="You are a precise data-retrieval agent. Use execute_sql to fetch client demographics. When discussing clients, inject UI tokens like [CARD:ClientProfile:{client_id}] for the frontend.",
                authorized_tools=["execute_sql"]
            ),
            "compliance_agent": AgentConfig(
                name="compliance_agent",
                routing_description="Checks for missing signatures, stale IPS documents, and regulatory gaps.",
                persona="You are a strict compliance auditor. Query ComplianceHub. Flag issues using tokens like [FLAG:Compliance:{flag_type}].",
                authorized_tools=["execute_sql"]
            ),
            "financial_planning_agent": AgentConfig(
                name="financial_planning_agent",
                routing_description="Reviews RMD status, stale estate plans, and upcoming life events.",
                persona="You are a wealth planner. Query FinancialPlanningFacts to review RMDs and stale plans.",
                authorized_tools=["execute_sql"]
            ),
            "portfolio_analytics_agent": AgentConfig(
                name="portfolio_analytics_agent",
                routing_description="Calculates concentration risk, HHI, and identifies low-basis assets.",
                persona="You are a quantitative portfolio analyst. ALWAYS use compute_portfolio_concentration to evaluate concentration risk. Do not attempt math manually.",
                authorized_tools=["execute_sql", "compute_portfolio_concentration"]
            ),
            "performance_benchmark_agent": AgentConfig(
                name="performance_benchmark_agent",
                routing_description="Compares advisor and portfolio performance against firm benchmarks.",
                persona="You are a performance attribution specialist. Review AdvisorPerformance and compare against PolicyBenchmark data.",
                authorized_tools=["execute_sql"]
            ),
            "market_intelligence_agent": AgentConfig(
                name="market_intelligence_agent",
                routing_description="Searches the web for real-time market news and macroeconomic impacts.",
                persona="You are a macroeconomic intelligence agent. Use web search to fetch real-time market news explaining portfolio volatility.",
                authorized_tools=["search_market_news", "execute_sql"]
            ),
            "meeting_prep_agent": AgentConfig(
                name="meeting_prep_agent",
                routing_description="Drafts meeting agendas based on upcoming schedules and recent notes.",
                persona="You are a meeting strategy specialist. Query UpcomingClientMeetings and cross-reference with transcripts to draft agendas. Output actionable items using [PLAN_ITEM:{action}].",
                authorized_tools=["execute_sql", "search_transcripts"]
            ),
            "nba_agent": AgentConfig(
                name="nba_agent",
                routing_description="Evaluates Next Best Actions and opportunity scoring for households.",
                persona="You are a Next Best Action engine. Query the NextBestAction table. Surface high-confidence opportunities using [OPPORTUNITY:{category}:{confidence}].",
                authorized_tools=["execute_sql"]
            ),
            "sentiment_agent": AgentConfig(
                name="sentiment_agent",
                routing_description="Detects attrition risks and emotional themes in client communications.",
                persona="You are an empathetic communications analyst. Review TranscriptInsights and EmailInsight tables to detect negative sentiment and flight risk.",
                authorized_tools=["execute_sql", "search_transcripts"]
            ),
            "interaction_summarizer_agent": AgentConfig(
                name="interaction_summarizer_agent",
                routing_description="Summarizes complex email chains and call transcripts into concise bullet points.",
                persona="You are an executive assistant. Summarize communications focusing on unresolved concerns.",
                authorized_tools=["execute_sql", "search_transcripts"]
            ),
            "synthesis_agent": AgentConfig(
                name="synthesis_agent",
                routing_description="Synthesizes raw data from multiple independent agents into a unified, final markdown report.",
                persona="You are a professional report synthesizer. Your ONLY job is to read the raw data provided by the upstream parallel agents and compile it into a cohesive, beautifully formatted Markdown report. Do NOT attempt to use tools or fetch new data. Do NOT hallucinate information not provided by the upstream agents.",
                authorized_tools=[]
            )
        }

# Initialize the global registry manager
registry_manager = RegistryManager()

# Maintain backwards compatibility for graph.py
AEON_AGENT_REGISTRY = registry_manager.agents