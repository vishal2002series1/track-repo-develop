# src/engine/dynamic_graph.py
import operator
import os
import json
from dotenv import load_dotenv
from typing import Annotated, Sequence, TypedDict, List ## Change 1 : Parallel : added List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

# 🟢 AZURE POSTGRESQL CHECKPOINTER (replaces local SqliteSaver)
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field  ## Change 2: Parallel : added Field

# AZURE MIGRATION: Swap Bedrock for Azure OpenAI
from langchain_openai import AzureChatOpenAI

from langchain_core.runnables import RunnableConfig

from sqlalchemy.orm import Session
from src.db.database import SessionLocal
from src.db.models import Workflow
from src.agents.tools import AEON_TOOLS
from src.db.pg_connection import (
    AZURE_PG_HOST,
    AZURE_PG_DATABASE,
    AZURE_PG_USER,
    AZURE_PG_PASSWORD,
    AZURE_PG_PORT,
    AZURE_PG_SSLMODE,
)

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

# --- PERSISTENT STATE MEMORY (AZURE POSTGRESQL) ---
# 🟢 LangGraph checkpoints now live in Azure PostgreSQL instead of the local
# data_local/threads.sqlite file. Tables are created on first run inside the
# schema configured by CHECKPOINTER_PG_SCHEMA (default: "Checkpoints").
#
# Requires psycopg v3 (not psycopg2) — `pip install psycopg[binary] psycopg-pool langgraph-checkpoint-postgres`.

CHECKPOINTER_PG_SCHEMA = os.getenv("CHECKPOINTER_PG_SCHEMA", "Checkpoints")

# psycopg v3 connection string (note: no "+psycopg2" driver suffix; this is a
# raw DBAPI URL, not a SQLAlchemy URL).
_CHECKPOINTER_CONN_STRING = (
    f"postgresql://{AZURE_PG_USER}:{AZURE_PG_PASSWORD}"
    f"@{AZURE_PG_HOST}:{AZURE_PG_PORT}/{AZURE_PG_DATABASE}"
    f"?sslmode={AZURE_PG_SSLMODE}"
)

# Eagerly create the checkpointer schema (PostgresSaver.setup() creates its
# tables but does not create the containing schema).
def _ensure_checkpointer_schema() -> None:
    import psycopg
    with psycopg.connect(_CHECKPOINTER_CONN_STRING, autocommit=True) as _c:
        with _c.cursor() as _cur:
            _cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{CHECKPOINTER_PG_SCHEMA}"')

_ensure_checkpointer_schema()

# PostgresSaver requires autocommit and prepare_threshold=0 per LangGraph docs.
# We set search_path via a `configure` callback (not via libpq `options=-c
# search_path=...`) because libpq lowercases unquoted identifiers in options,
# which breaks mixed-case schema names like "Checkpoints".
def _configure_checkpointer_conn(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(f'SET search_path TO "{CHECKPOINTER_PG_SCHEMA}"')

_checkpointer_pool = ConnectionPool(
    conninfo=_CHECKPOINTER_CONN_STRING,
    max_size=20,
    kwargs={
        "autocommit": True,
        "prepare_threshold": 0,
    },
    configure=_configure_checkpointer_conn,
)

workflow_memory = PostgresSaver(_checkpointer_pool)
workflow_memory.setup()  # Idempotent — creates checkpoint tables on first run.

# Ensure the connection pool's background worker threads are shut down cleanly
# on interpreter exit. Without this, psycopg_pool prints warnings like:
#   "couldn't stop thread 'pool-1-worker-0' within 5.0 seconds"
import atexit as _atexit

def _close_checkpointer_pool() -> None:
    try:
        _checkpointer_pool.close()
    except Exception:
        pass

_atexit.register(_close_checkpointer_pool)
# ----------------------------------

# --- 1. Graph State Definition ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: List[str] ## Change 3 : Parallel : Added List[str] to allow multiple next nodes for parallel execution

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
        max_tokens=8000,
        streaming=True  ### Change : Parallel : Enable streaming for real-time token updates
    )

# --- Helper Function: Flatten History ---
def extract_clean_history(messages):
    """
    Extracts a clean, text-only representation of the conversation history
    for the synthesizer. Includes:
      - The user's original question (always first)
      - AI prose responses (even from intermediate ReAct steps)
      - Tool results as structured data blocks
    """
    history_text = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history_text += f"USER: {msg.content}\n\n"
        elif isinstance(msg, AIMessage):
            # Include AI content whether or not it has tool_calls.
            # ReAct agents often emit content="" on tool-calling steps — skip those.
            if msg.content and str(msg.content).strip():
                history_text += f"ASSISTANT: {msg.content}\n\n"
        elif isinstance(msg, ToolMessage):
            history_text += f"TOOL DATA: {msg.content}\n\n"
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

    custom_sup = getattr(workflow, 'supervisor_prompt', None)
    custom_syn = getattr(workflow, 'synthesizer_prompt', None)

    active_supervisor_rules = custom_sup.strip() if custom_sup and custom_sup.strip() else prompts.get("supervisor_rules", "")
    active_synthesizer_persona = custom_syn.strip() if custom_syn and custom_syn.strip() else prompts.get("synthesizer_persona", "You are a helpful assistant.")

    # UNIVERSAL_WIDGET_RULE = """
    # CRITICAL UI FORMATTING RULE:
    # When the user asks for a chart, graph, or visual representation, DO NOT describe it in text paragraphs.
    # You MUST output your data using this EXACT markdown code block format:
    # ```json:widget
    # {
    # "chart_type": "donut",
    # "title": "Title Here",
    # "labels": ["A", "B", "C"],
    # "values": [10, 20, 30]
    # }
    # ```
    # CRITICAL RULES:
    # NEVER output raw streaming syntax like data: {"type": ....
    # ONLY use the markdown block exactly as shown above.
    # Supported chart_types are "donut" and "bar".
    # """

    
    # active_synthesizer_persona += "\n" + UNIVERSAL_WIDGET_RULE
    UNIVERSAL_WIDGET_RULE = r"""
CRITICAL UI FORMATTING RULE:
When the user asks for a chart, graph, or visual representation, you MUST output data using the
exact markdown block below. Do NOT describe charts in prose paragraphs.

Supported chart_type values and when to use them:
- "donut"     → single-series percentage/proportion breakdown (e.g. allocation by asset class)
- "pie"       → same as donut but without a hole
- "bar"       → comparing discrete categories (labels = category names, values = amounts)
- "line"      → trend over time (labels = dates/periods, values = numeric series)
- "scatter"   → two continuous numeric dimensions, one per client/entity
                 Use keys: "x" (list of x-values), "y" (list of y-values), "labels" (point names)
- "area"      → filled line chart for cumulative/stacked trends
- "table"     → raw tabular data; use keys "columns" (list) and "rows" (list of lists)

Format for donut / pie / bar / line / area:
```json:widget
{
  "chart_type": "bar",
  "title": "Title Here",
  "labels": ["A", "B", "C"],
  "values": [10, 20, 30]
}
```

Format for scatter (IMPORTANT — use x/y/labels, NOT labels/values):
```json:widget
{
  "chart_type": "scatter",
  "title": "Portfolio Value vs Equity Exposure",
  "labels": ["Client A", "Client B", "Client C"],
  "x": [1200000, 850000, 3100000],
  "y": [42.5, 31.0, 67.8],
  "x_label": "Portfolio Value ($)",
  "y_label": "Equity Exposure (%)"
}
```

Format for table:
```json:widget
{
  "chart_type": "table",
  "title": "Client Summary",
  "columns": ["Name", "Portfolio Value", "Equity %"],
  "rows": [["Alice", 1200000, 42.5], ["Bob", 850000, 31.0]]
}
```

CRITICAL RULES:
- Output ONLY the markdown block(s) — no surrounding prose description of the chart itself.
- You MAY output a brief text summary BEFORE the block, then the block.
- Multiple charts in one response are allowed — just output multiple blocks sequentially.
- Never output raw SSE syntax (data: {...}).
"""

    GLOBAL_TRUST_RULE = """
    CRITICAL RULE - CITATIONS:
    Whenever you state a specific fact, metric, date, or detail retrieved by a worker agent, you MUST cite the source inline. 
    Format your citations strictly like this: [Source: TableName] or [Source: DocumentName].
    Example: "Jonathan has an AUM of $47.10M [Source: PortfolioData]."
    Never invent a source. If you don't know where the data came from, do not add a citation.
    """
    # active_synthesizer_persona += "\n" + GLOBAL_TRUST_RULE

    # Append BOTH rules
    active_synthesizer_persona += "\n" + UNIVERSAL_WIDGET_RULE + "\n" + GLOBAL_TRUST_RULE

    # 👈 Fixed: Define agent_descriptions BEFORE using it
    agent_descriptions = "\n".join([f"- {agent.id}: {agent.routing_description}" for agent in agents])
    
    # 🟢 Modified to use active_supervisor_rules
    system_prompt = f"""
    You are the Supervisor orchestrating a team of domain expert agents.
    
    YOUR AVAILABLE AGENTS:
    {agent_descriptions}
    
    {active_supervisor_rules}
    """

    # --- NEW: Synthesizer Node (Moved INSIDE so it can read active_synthesizer_persona) ---

    ## Change 
    def synthesizer_node(state: AgentState, config: RunnableConfig):
        """Formats the final response beautifully for the user."""
        llm = get_llm()
        clean_history = extract_clean_history(state["messages"])
                
        # Use SystemMessage + HumanMessage directly to avoid LangChain parsing
        # the {JSON} examples inside active_synthesizer_persona as template variables.
        from langchain_core.messages import SystemMessage as _SystemMessage
        human_text = (
            "Here is the full conversation history including tool data collected by worker agents:\n\n"
            f"{clean_history}\n\n"
            "The USER's question is at the top of the history above (marked 'USER:').\n"
            "The TOOL DATA sections contain raw query results that answer the question.\n\n"
            "Your job: synthesize a complete, well-formatted response to the user's question "
            "using the tool data. If the question asks for a chart or graph, output a json:widget "
            "block as instructed in your system prompt. Do NOT say the data is unavailable — "
            "it is in the TOOL DATA above."
        )
        result = llm.invoke(
            [_SystemMessage(content=active_synthesizer_persona), HumanMessage(content=human_text)],
            config=config,
        )
        
        return {"messages": [result]}
        
    
    
        
    # Append the dynamic rules instead of hardcoded strings
    # system_prompt += active_supervisor_rules
    print(f"🔍 [SYSTEM PROMPT LENGTH]: {len(system_prompt)}")
    print(f"🔍 [SYSTEM PROMPT TAIL]:\n{system_prompt[-500:]}")
    ## Change 4: Parallel : # 👈 CHANGED: Dynamic Pydantic schema expects a list

    class Route(BaseModel):
        next: List[str] = Field(
            description=(
                f"A list of agent names to route to. Options are: {options}. "
                "If multiple distinct tasks are required, list multiple agents to run them in parallel. "
                "If the task is fully complete, return ['synthesizer']."
            )
        )

    def supervisor_node(state: AgentState):
        llm = get_llm()
        
        clean_history = extract_clean_history(state["messages"])

        # TEMPORARY DEBUG
        print(f"🔍 [SUPERVISOR DEBUG] Messages in state: {len(state['messages'])}")
        print(f"🔍 [SUPERVISOR DEBUG] History passed to supervisor:\n{clean_history}")
        
        from langchain_core.messages import SystemMessage as _SystemMessage
        supervisor_result = llm.with_structured_output(Route).invoke([
            _SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"Here is the conversation history:\n\n{clean_history}\n\n"
                f"Based on the history and the latest user request, who should act next? "
                f"Select one or more from: {options}"
            ))
        ])
        
        print(f"🔗 [SUPERVISOR] Routing to: {supervisor_result.next}")
        return {"next": supervisor_result.next}

    ## Change 4: End

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