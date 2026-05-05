# AUTO-COMPILED DOMAIN SUPERVISOR: WF_003
import os
import sqlite3
import operator
from typing import Annotated, Sequence, TypedDict, Literal, List
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_aws import ChatBedrock
from pydantic import BaseModel, Field

from src.agents.config import registry_manager
from src.agents.factory import AgentFactory
from src.utils.prompt_manager import prompt_manager  # <-- Dynamic Prompts

class WF_003State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_node: str
    instructions: str

# 🧠 Supervisor Routing Schema
class RouterOutput(BaseModel):
    next_node: Literal['client_profile_domain_agent', 'portfolio_domain_agent', 'crm_activities_domain_agent', 'transactions_domain_agent', 'financial_plan_domain_agent', 'box_files_domain_agent', 'task_management_domain_agent', 'FINISH'] = Field(
        ..., 
        description="The exact string name of the next agent to call, or 'FINISH'."
    )
    instructions: str = Field(..., description="Specific instructions for the next agent.")
    rejection_response: str = Field(
        default="", 
        description="ONLY USE THIS IF REJECTING A QUERY. If the user asks something completely out of scope, write a polite refusal here."
    )

def build_WF_003_graph():
    factory = AgentFactory()
    builder = StateGraph(WF_003State)
    
    # 🏗️ Load Bounded Domain Workers
    builder.add_node('client_profile_domain_agent', factory.build_node(registry_manager.agents.get('client_profile_domain_agent')))
    builder.add_node('portfolio_domain_agent', factory.build_node(registry_manager.agents.get('portfolio_domain_agent')))
    builder.add_node('crm_activities_domain_agent', factory.build_node(registry_manager.agents.get('crm_activities_domain_agent')))
    builder.add_node('transactions_domain_agent', factory.build_node(registry_manager.agents.get('transactions_domain_agent')))
    builder.add_node('financial_plan_domain_agent', factory.build_node(registry_manager.agents.get('financial_plan_domain_agent')))
    builder.add_node('box_files_domain_agent', factory.build_node(registry_manager.agents.get('box_files_domain_agent')))
    builder.add_node('task_management_domain_agent', factory.build_node(registry_manager.agents.get('task_management_domain_agent')))

    # 👑 The Supervisor Node
    def supervisor(state: WF_003State):
        model_id = os.getenv("MODEL_ID", "us.anthropic.claude-sonnet-4-6")
        llm = ChatBedrock(model_id=model_id, region_name="us-east-1", temperature=0.0)
        
        # 🛑 Pull dynamic prompt
        system_prompt = prompt_manager.get_prompt("supervisor_system_prompt", agent_names="['client_profile_domain_agent', 'portfolio_domain_agent', 'crm_activities_domain_agent', 'transactions_domain_agent', 'financial_plan_domain_agent', 'box_files_domain_agent', 'task_management_domain_agent']")
        
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

    # 📝 The Final Synthesis Node (For Business Users)
    def synthesizer(state: WF_003State):
        # If the last message is a rejection from the supervisor, just pass it through
        if len(state["messages"]) > 0 and state["messages"][-1].type == "ai":
             if "internal routing error" in state["messages"][-1].content or "outside the scope" in state["messages"][-1].content or "not able to" in state["messages"][-1].content:
                 return {"messages": []}

        model_id = os.getenv("MODEL_ID", "us.anthropic.claude-sonnet-4-6")
        llm = ChatBedrock(model_id=model_id, region_name="us-east-1", temperature=0.2, max_tokens=8000)
        
        # 🛑 Pull dynamic prompt
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

    # 🗺️ Logic: Always return to supervisor after a worker finishes
    for name in ['client_profile_domain_agent', 'portfolio_domain_agent', 'crm_activities_domain_agent', 'transactions_domain_agent', 'financial_plan_domain_agent', 'box_files_domain_agent', 'task_management_domain_agent']:
        builder.add_edge(name, "supervisor")

    # 🏁 Logic: Supervisor routes to Synthesizer on FINISH
    def should_continue(state: WF_003State):
        if state["next_node"] == "FINISH":
            return "synthesizer"
        return state["next_node"]

    builder.add_conditional_edges("supervisor", should_continue)
    builder.add_edge("synthesizer", END)
    builder.add_edge(START, "supervisor")

    # 💾 Set up Persistent Database Memory
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data_local/sessions.db'))
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)
    memory.setup()  # Automatically creates the tables if they don't exist

    return builder.compile(checkpointer=memory)

if __name__ == "__main__":
    print("🎉 Successfully compiled Bounded Domain Supervisor graph with Synthesis.")
