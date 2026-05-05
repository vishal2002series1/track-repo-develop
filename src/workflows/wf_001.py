# AUTO-COMPILED DOMAIN SUPERVISOR: WF_001
import os
import operator
from dotenv import load_dotenv
from typing import Annotated, Sequence, TypedDict, Literal, List
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

# 🟢 AZURE MIGRATION
from langchain_openai import AzureChatOpenAI

from src.agents.config import registry_manager
from src.agents.factory import AgentFactory
from src.utils.prompt_manager import prompt_manager 

load_dotenv()

class WF_001State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_node: str
    instructions: str

class RouterOutput(BaseModel):
    next_node: Literal['client_profile_domain_agent', 'portfolio_domain_agent', 'interactions_domain_agent', 'compliance_documents_domain_agent', 'email_communications_domain_agent', 'portfolio_holdings_domain_agent', 'performance_benchmarking_domain_agent', 'opportunity_discovery_domain_agent', 'client_context_domain_agent', 'FINISH'] = Field(
        ..., 
        description="The exact string name of the next agent to call, or 'FINISH'."
    )
    instructions: str = Field(..., description="Specific instructions for the next agent.")
    rejection_response: str = Field(
        default="", 
        description="ONLY USE THIS IF REJECTING A QUERY."
    )

def get_azure_llm(temp=0.0):
    """Helper to cleanly initialize the Azure LLM across nodes."""
    api_key = os.getenv("API_KEYS")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("OPENAI_API_VERSION")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.4")
    
    return AzureChatOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
        azure_deployment=deployment_name,
        temperature=temp,
        max_tokens=8000
    )

def build_WF_001_graph():
    factory = AgentFactory()
    builder = StateGraph(WF_001State)
    
    builder.add_node('client_profile_domain_agent', factory.build_node(registry_manager.agents.get('client_profile_domain_agent')))
    builder.add_node('portfolio_domain_agent', factory.build_node(registry_manager.agents.get('portfolio_domain_agent')))
    builder.add_node('interactions_domain_agent', factory.build_node(registry_manager.agents.get('interactions_domain_agent')))
    builder.add_node('compliance_documents_domain_agent', factory.build_node(registry_manager.agents.get('compliance_documents_domain_agent')))
    builder.add_node('email_communications_domain_agent', factory.build_node(registry_manager.agents.get('email_communications_domain_agent')))
    builder.add_node('portfolio_holdings_domain_agent', factory.build_node(registry_manager.agents.get('portfolio_holdings_domain_agent')))
    builder.add_node('performance_benchmarking_domain_agent', factory.build_node(registry_manager.agents.get('performance_benchmarking_domain_agent')))
    builder.add_node('opportunity_discovery_domain_agent', factory.build_node(registry_manager.agents.get('opportunity_discovery_domain_agent')))
    builder.add_node('client_context_domain_agent', factory.build_node(registry_manager.agents.get('client_context_domain_agent')))

    def supervisor(state: WF_001State):
        llm = get_azure_llm(temp=0.0)
        
        system_prompt = prompt_manager.get_prompt("supervisor_system_prompt", agent_names="['client_profile_domain_agent', 'portfolio_domain_agent', 'interactions_domain_agent', 'compliance_documents_domain_agent', 'email_communications_domain_agent', 'portfolio_holdings_domain_agent', 'performance_benchmarking_domain_agent', 'opportunity_discovery_domain_agent', 'client_context_domain_agent']")
        
        messages_to_pass = list(state["messages"])
        if len(messages_to_pass) > 0 and messages_to_pass[-1].type != "human":
            nudge_text = prompt_manager.get_prompt("supervisor_nudge_prompt")
            messages_to_pass.append(HumanMessage(content=nudge_text))
        
        planner = llm.with_structured_output(RouterOutput)
        
        try:
             response = planner.invoke([SystemMessage(content=system_prompt)] + messages_to_pass)
             next_agent = response.next_node
             instructions = response.instructions
             rejection_text = response.rejection_response
        except Exception as e:
             print(f"      ⚠️ Supervisor parsing error: {e}. Defaulting to FINISH.")
             next_agent = "FINISH"
             instructions = "Error in routing."
             rejection_text = "I encountered an internal routing error."
             
        print(f"\n👑 [SUPERVISOR] Routing to: {next_agent}")
        if next_agent != "FINISH":
             print(f"   ↳ Instructions: {instructions}")
             return {"next_node": next_agent, "instructions": instructions}
        else:
             if rejection_text:
                 return {"next_node": next_agent, "instructions": instructions, "messages": [AIMessage(content=rejection_text)]}
             return {"next_node": next_agent, "instructions": instructions}

    def synthesizer(state: WF_001State):
        if len(state["messages"]) > 0 and state["messages"][-1].type == "ai":
             if "internal routing error" in state["messages"][-1].content or "outside the scope" in state["messages"][-1].content or "not able to" in state["messages"][-1].content:
                 return {"messages": []}

        llm = get_azure_llm(temp=0.2)
        
        system_prompt = prompt_manager.get_prompt("synthesizer_system_prompt")
        
        messages_to_pass = list(state["messages"])
        if len(messages_to_pass) > 0 and messages_to_pass[-1].type != "human":
            nudge_text = prompt_manager.get_prompt("synthesizer_nudge_prompt")
            messages_to_pass.append(HumanMessage(content=nudge_text))
        
        print(f"\n📝 [SYNTHESIZER] Formatting final response for associate...")
        response = llm.invoke([SystemMessage(content=system_prompt)] + messages_to_pass)
        return {"messages": [response]}

    builder.add_node("supervisor", supervisor)
    builder.add_node("synthesizer", synthesizer)

    for name in ['client_profile_domain_agent', 'portfolio_domain_agent', 'interactions_domain_agent', 'compliance_documents_domain_agent', 'email_communications_domain_agent', 'portfolio_holdings_domain_agent', 'performance_benchmarking_domain_agent', 'opportunity_discovery_domain_agent', 'client_context_domain_agent']:
        builder.add_edge(name, "supervisor")

    def should_continue(state: WF_001State):
        if state["next_node"] == "FINISH":
            return "synthesizer"
        return state["next_node"]

    builder.add_conditional_edges("supervisor", should_continue)
    builder.add_edge("synthesizer", END)
    builder.add_edge(START, "supervisor")

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

if __name__ == "__main__":
    print("🎉 Successfully compiled Domain Supervisor graph with Synthesis.")