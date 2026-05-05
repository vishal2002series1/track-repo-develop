# src/agents/graph.py
import operator
import sqlite3
import os
from typing import Annotated, Sequence, TypedDict, Literal, List
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_aws import ChatBedrock
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.types import interrupt
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agents.config import AEON_AGENT_REGISTRY
from src.agents.factory import AgentFactory

# --- 1. The Universal State ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_agent: Annotated[list[str], operator.add]
    routing_log: Annotated[list[str], operator.add]

# --- 2. The Supervisor's Brain (Intent Router) ---
agent_names = list(AEON_AGENT_REGISTRY.keys())
route_options = agent_names + ["ASK_HUMAN", "FINISH"]

class RouteDecision(BaseModel):
    next_nodes: List[str] = Field(
        ...,
        description="A list of precise agents to route to based on intent. If tasks can be done simultaneously, list multiple agents. Use ['ASK_HUMAN'] if ambiguous. Use ['FINISH'] if fully answered."
    )
    reasoning: str = Field(..., description="Explanation of why this route aligns with the advisor's intent.")

# --- 3. Build the Graph ---
def build_aeon_graph():
    factory = AgentFactory()
    
    supervisor_llm = ChatBedrock(
        model_id="us.anthropic.claude-sonnet-4-6", 
        region_name="us-east-1",
        temperature=0.0
    ).with_structured_output(RouteDecision)

    def supervisor_node(state: AgentState):
        # 🛡️ THE CIRCUIT BREAKER
        if len(state.get("routing_log", [])) > 8: 
            return {
                "messages": [AIMessage(content="Circuit Breaker Activated: Maximum reasoning steps reached.")],
                "current_agent": ["supervisor"],
                "routing_log": ["Routed to FINISH because: Circuit breaker triggered"]
            }
            
        # 🧠 DYNAMIC INTENT CATALOG (OPTIMIZED)
        agent_descriptions = "\n".join(
            [f"- {name}: {config.routing_description} (Tools available: {', '.join(config.authorized_tools)})" 
             for name, config in AEON_AGENT_REGISTRY.items()]
        )
            
        system_prompt = f"""You are the AEON Wealth Orchestrator and Intent Router.
        Your job is to classify the advisor's intent and route the task to the exact specialized agent required.
        
        AVAILABLE AGENT CATALOG:
        {agent_descriptions}
        
        CRITICAL RULES:
        1. Classify Intent: Match the user's request to the agent(s) with the right tools.
        2. PARALLEL EXECUTION: If a request requires data from multiple independent sources (e.g., getting SQL portfolio data AND searching the web for news), you MUST route to multiple agents simultaneously by including them all in your list.
        3. Ambiguity: If the query is ambiguous (e.g., asking for a first name with multiple matches), route to 'ASK_HUMAN'.
        4. Completion & Synthesis: If you just ran agents in parallel, you CANNOT route to FINISH. You must route to the `synthesis_agent` to merge their data first. ONLY route to 'FINISH' after the synthesis_agent has generated the final report, or if a simple single-agent task is fully resolved.
        CRITICAL: Do NOT route a query back to an agent that has already completed its task."""
        
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        
        # 🛑 BEDROCK FIX: The conversation history cannot end with an AI message.
        if messages and getattr(messages[-1], "type", None) == "ai":
            nudge = "Review the agent's output above. If the advisor's request is fully resolved, route to FINISH. If more data is needed, route to the next appropriate agent(s)."
            messages.append(HumanMessage(content=nudge))
        
        decision = supervisor_llm.invoke(messages)
        nodes_str = ",".join(decision.next_nodes)
        
        return {
            "current_agent": ["supervisor"],
            "routing_log": [f"Routed to {nodes_str} because: {decision.reasoning}"]
        }

    # 🛑 THE HUMAN-IN-THE-LOOP NODE
    def human_clarification_node(state: AgentState):
        question_to_ask = state["routing_log"][-1]
        human_response = interrupt(f"Clarification needed: {question_to_ask}")
        return {
            "messages": [HumanMessage(content=f"[Advisor Clarification]: {human_response}")],
            "current_agent": ["human"]
        }

    # Initialize Graph
    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("ASK_HUMAN", human_clarification_node)
    
    # Dynamically load all Factory Agents
    for agent_name, config in AEON_AGENT_REGISTRY.items():
        worker_node = factory.build_node(config)
        workflow.add_node(agent_name, worker_node)
        workflow.add_edge(agent_name, "supervisor")
        
    workflow.add_edge("ASK_HUMAN", "supervisor")

    def route_from_supervisor(state: AgentState) -> List[str]:
        last_log = state.get("routing_log", [""])[-1]
        
        # If the circuit breaker hit or it explicitly routed to FINISH
        if "Routed to FINISH" in last_log:
            return ["FINISH"]
            
        # Parse the 'Routed to agent1,agent2 because...' string
        try:
            nodes_str = last_log.split("Routed to ")[1].split(" because:")[0]
            active_nodes = [node.strip() for node in nodes_str.split(",") if node.strip() in route_options]
            
            if active_nodes:
                return active_nodes
        except IndexError:
            pass
            
        return ["FINISH"]

    edge_mapping = {name: name for name in route_options if name != "FINISH"}
    edge_mapping["FINISH"] = END 

    workflow.add_conditional_edges("supervisor", route_from_supervisor, edge_mapping)
    workflow.add_edge(START, "supervisor")
    
    # 🧠 PERSISTENT MEMORY INTEGRATION
    local_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data_local'))
    os.makedirs(local_data_dir, exist_ok=True)
    
    conn = sqlite3.connect(os.path.join(local_data_dir, "checkpoints.sqlite"), check_same_thread=False)
    memory = SqliteSaver(conn)
    
    return workflow.compile(checkpointer=memory)

aeon_app = build_aeon_graph()