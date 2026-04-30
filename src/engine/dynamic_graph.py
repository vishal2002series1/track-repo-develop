# src/engine/dynamic_graph.py
import operator
import os
import sqlite3
import json
from dotenv import load_dotenv
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

# AZURE MIGRATION: Swap Bedrock for Azure OpenAI
from langchain_openai import AzureChatOpenAI 

from sqlalchemy.orm import Session
from src.db.database import SessionLocal
from src.db.models import Workflow
from src.agents.tools import AEON_TOOLS 

# Load environment variables from .env file
load_dotenv()

# --- PROMPT LIBRARY LOADER ---
def load_prompt_library():
    """Dynamically loads global system prompts from the config folder."""
    # Maps to: src/config/prompt_library.json
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/prompt_library.json'))
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Warning: Could not load prompt_library.json: {e}")
        return {"supervisor_rules": "", "synthesizer_persona": "You are a helpful assistant."}

# --- PERSISTENT STATE MEMORY ---
# We use SqliteSaver so threads survive server restarts and cross-platform tests
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data_local/threads.sqlite'))
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Create a connection that allows access from multiple threads (crucial for FastAPI)
memory_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
workflow_memory = SqliteSaver(memory_conn)
workflow_memory.setup() # Automatically creates the necessary memory tables
# ----------------------------------

# --- 1. Graph State Definition ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# --- 2. LLM Initialization ---
def get_llm():
    # AZURE MIGRATION: Dynamically pull credentials
    api_key = os.getenv("API_KEYS")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("OPENAI_API_VERSION")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.4")
    
    return AzureChatOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
        azure_deployment=deployment_name,
        temperature=0.0,
        max_tokens=8000
    )

# --- Helper Function: Flatten History ---
def extract_clean_history(messages):
    """
    Extracts a clean, text-only representation of the conversation history,
    ignoring tool calls and intermediate routing steps.
    """
    history_text = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
             history_text += f"USER: {msg.content}\n\n"
        elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
             history_text += f"ASSISTANT: {msg.content}\n\n"
        elif isinstance(msg, ToolMessage):
             history_text += f"[Tool Result Data Available]\n\n"
    return history_text

# --- 3. Dynamic Worker Node Generator ---
def create_worker_node(persona: str, tools: list):
    """Creates a worker that can actually execute tools (React Agent)."""
    worker_agent = create_react_agent(get_llm(), tools=tools, prompt=persona)
    
    def worker_node(state: AgentState):
        messages = list(state["messages"])
        original_global_length = len(state["messages"])
        
        if messages and messages[-1].type == "ai":
            nudge = (
                "Please continue. Review the previous steps and take the next appropriate action. "
                "If your previous SQL queries failed, you MUST use the get_database_schema tool "
                "to verify the correct table and column names before writing SQL again."
            )
            messages.append(HumanMessage(content=nudge))
            
        result = worker_agent.invoke({"messages": messages})
        
        new_messages = result["messages"][original_global_length:]
        return {"messages": new_messages}
        
    return worker_node

# --- NEW: Synthesizer Node ---
def synthesizer_node(state: AgentState):
    """Formats the final response beautifully for the user."""
    llm = get_llm()
    prompts = load_prompt_library() # 👈 Load dynamically
    
    clean_history = extract_clean_history(state["messages"])
            
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompts.get("synthesizer_persona", "You are a helpful assistant.")),
        ("human", "Here is the conversation history and data:\n\n{history}")
    ])
    
    chain = prompt | llm
    result = chain.invoke({"history": clean_history})
    
    return {"messages": [result]}

# --- 4. The Master Compiler ---
def build_dynamic_graph(workflow_id: str, db: Session):
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    if not workflow:
        raise ValueError(f"Workflow {workflow_id} not found in database.")
        
    agents = workflow.agents
    if not agents:
        raise ValueError(f"No agents attached to {workflow_id}.")

    agent_names = [agent.id for agent in agents]
    options = ["synthesizer"] + agent_names
    
    prompts = load_prompt_library() # 👈 Load dynamically
    
    system_prompt = (
        "You are the Workflow Supervisor. Your job is to route the conversation to the correct expert agent based on the user's request.\n"
        "Available Agents:\n"
    )
    for agent in agents:
        system_prompt += f"- {agent.id}: {agent.routing_description}\n"
        
    # Append the dynamic rules instead of hardcoded strings
    system_prompt += prompts.get("supervisor_rules", "")

    class Route(BaseModel):
        next: str

    def supervisor_node(state: AgentState):
        llm = get_llm()
        
        clean_history = extract_clean_history(state["messages"])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Here is the conversation history:\n\n{history}\n\nBased on the history and the latest user request, who should act next? Select one of: {options}")
        ]).partial(options=str(options))
        
        supervisor_chain = prompt | llm.with_structured_output(Route)
        result = supervisor_chain.invoke({"history": clean_history})
        
        print(f"🔗 [SUPERVISOR] Routing to: {result.next}")
        return {"next": result.next}

    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("synthesizer", synthesizer_node)
    
    for agent in agents:
        agent_tools = [tool for tool in AEON_TOOLS if tool.name in agent.authorized_tools]
        builder.add_node(agent.id, create_worker_node(agent.persona, agent_tools))
        builder.add_edge(agent.id, "supervisor")
        
    builder.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {**{name: name for name in agent_names}, "synthesizer": "synthesizer"}
    )
    
    builder.add_edge("synthesizer", END)
    builder.set_entry_point("supervisor")
    
    return builder.compile(checkpointer=workflow_memory)