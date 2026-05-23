# main.py
import os
import uuid
import sys
import json
import threading
from datetime import datetime, timezone
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry.sdk.trace import TracerProvider as _SDKTracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult, SimpleSpanProcessor

### SSE Change1: Added for streaming execution endpoint
from fastapi.responses import StreamingResponse
from src.engine.router import execute_plan_stream

### SSE Change 1 End

from src.db.database import engine, SessionLocal, Base
from src.db.models import DomainAgent, Workflow

#Router specific imports
from fastapi import BackgroundTasks
from src.memory.episodic import EpisodicMemory
from src.engine.router import plan_route, execute_plan, RoutePlan

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.engine.dynamic_graph import build_dynamic_graph, get_llm, workflow_memory
from src.agents.tools import AEON_TOOLS
from langgraph.prebuilt import create_react_agent

from arize.otel import register, Transport
from openinference.instrumentation.langchain import LangChainInstrumentor


class _SpanStore:
    def __init__(self):
        self._data: dict = {}
        self._trace_to_session: dict = {}
        self._lock = threading.Lock()

    def register_session(self, trace_id: str, session_id: str):
        with self._lock:
            self._trace_to_session[trace_id] = session_id

    def add_span(self, span):
        trace_id = format(span.context.trace_id, '032x')
        with self._lock:
            session_id = self._trace_to_session.get(trace_id)
            if not session_id:
                return
            start_ns = span.start_time
            end_ns = span.end_time or start_ns
            start_dt = datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc)
            if session_id not in self._data:
                self._data[session_id] = []
            self._data[session_id].append({
                "trace_id": trace_id,
                "span_id": format(span.context.span_id, '016x'),
                "name": span.name,
                "start_time": start_dt.isoformat(),
                "end_time": end_dt.isoformat(),
                "latency_ms": round((end_ns - start_ns) / 1e6, 2),
                "status": span.status.status_code.name,
            })

    def get_session(self, session_id: str) -> list:
        with self._lock:
            spans = list(self._data.get(session_id, []))
        return sorted(spans, key=lambda x: x["start_time"])


class _LocalSpanExporter(SpanExporter):
    def __init__(self, store: _SpanStore):
        self._store = store

    def export(self, spans):
        for span in spans:
            self._store.add_span(span)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


span_store = _SpanStore()

# Single process-wide episodic memory store (Chroma, MiniLM, local disk).
# Migrating to pgvector later will require ONLY swapping the class behind
# this name — no other code in main.py changes.
episodic_memory = EpisodicMemory()

# --- Arize Telemetry Initialization ---
load_dotenv()
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_arize_space_id = os.getenv("ARIZE_SPACE_ID")
_arize_api_key = os.getenv("ARIZE_API_KEY")

if _arize_space_id and _arize_api_key:
    _cert_path = os.path.join(PROJECT_ROOT, "certificates", "windows-ca-bundle.pem")
    os.environ["REQUESTS_CA_BUNDLE"] = _cert_path
    os.environ["SSL_CERT_FILE"] = _cert_path
    # Connectivity check — confirms the OTLP endpoint is reachable before registering
    try:
        import urllib.request
        req = urllib.request.Request("https://otlp.arize.com/v1/traces", method="POST")
        urllib.request.urlopen(req, timeout=5)
    except urllib.error.HTTPError as _e:
        # Any HTTP error (400, 401, 405...) means the host IS reachable — expected without a valid payload
        print(f"✅  Arize endpoint reachable (HTTP {_e.code})")
    except Exception as _e:
        print(f"❌  Arize endpoint NOT reachable: {_e}")
        print("    Traces will be silently dropped. Check corporate proxy / firewall settings.")
    _tracer_provider = register(
        space_id=_arize_space_id,
        api_key=_arize_api_key,
        project_name="AEON-2.0",
        transport=Transport.HTTP,
        endpoint="https://otlp.arize.com/v1/traces"
    )
    LangChainInstrumentor().instrument(tracer_provider=_tracer_provider)
    # Use the base SDK method to ADD alongside Arize's BatchSpanProcessor,
    # not Arize's overridden add_span_processor which would replace it.
    _SDKTracerProvider.add_span_processor(_tracer_provider, SimpleSpanProcessor(_LocalSpanExporter(span_store)))
    print("👁️  Arize Telemetry Active: Tracing all LLM calls and agent workflows.")
else:
    print("⚠️  Arize keys not found in .env. Telemetry disabled.")

def debug_backend(message: str) -> None:
    print(f"[AEON BACKEND] {message}", file=sys.stderr, flush=True)

# Ensure tables exist
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Aeon Agent Factory API", 
    description="Headless backend for dynamic LangGraph workflows.",
    version="2.0"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend's origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Dependency: Database Session ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Pydantic Schemas (API Contracts for the Frontend) ---
class AgentSchema(BaseModel):
    id: str
    name: str
    routing_description: str
    persona: str
    authorized_tools: list

    class Config:
        from_attributes = True

class AgentUpdateRequest(BaseModel):
    name: Optional[str] = None
    routing_description: Optional[str] = None
    persona: Optional[str] = None
    authorized_tools: Optional[list] = None

class WorkflowSchema(BaseModel):
    id: str
    name: str
    description: str
    supervisor_prompt: Optional[str] = None  # <-- Added for it to be  endpoint updatable
    synthesizer_prompt: Optional[str] = None # <-- Added for it to be  endpoint updatable


    class Config:
        from_attributes = True

class WorkflowCreateRequest(BaseModel):
    id: str
    name: str
    description: str
    supervisor_prompt: Optional[str] = None  # <-- Added for it to be  endpoint updatable
    synthesizer_prompt: Optional[str] = None # <-- Added for it to be  endpoint updatable


class WorkflowUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    supervisor_prompt: Optional[str] = None  # <-- Added for it to be  endpoint updatable
    synthesizer_prompt: Optional[str] = None # <-- Added for it to be  endpoint updatable


class MapAgentRequest(BaseModel):
    agent_id: str

class ChatRequest(BaseModel):
    # workflow_id is now optional. If omitted or set to "auto", the router decides.
    workflow_id: Optional[str] = None
    prompt: str
    session_id: Optional[str] = None
    # Identity scoping for episodic memory. Optional today — wire from the
    # frontend when auth is available; defaults keep behavior backward-compatible.
    user_id: Optional[str] = None
    tenant_id: Optional[str] = "default"

class PlanPreviewRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = "default"

class ToolSchema(BaseModel):
    name: str
    description: str

class PlaygroundRequest(BaseModel):
    persona: str
    prompt: str
    tools: List[str] = []

# --- 🟢 SYSTEM & HEALTH ENDPOINTS ---
@app.get("/", tags=["Health"])
def root():
    return {"status": "online", "message": "Aeon Agent Factory is running. Visit /docs for Swagger UI."}

@app.get("/api/system/stats", tags=["Health"])
def get_system_stats(db: Session = Depends(get_db)):
    """Returns high-level statistics for the admin dashboard UI."""
    return {
        "total_agents": db.query(DomainAgent).count(),
        "total_workflows": db.query(Workflow).count(),
        "status": "Healthy"
    }

# --- 🛠️ TOOL & PLAYGROUND ENDPOINTS ---
@app.get("/api/tools", response_model=List[ToolSchema], tags=["Tools"])
def list_available_tools():
    """Get all available tools that can be assigned to agents."""
    return [{"name": t.name, "description": t.description} for t in AEON_TOOLS]

@app.post("/api/playground", tags=["Agent Playground"])
def test_agent_prompt(request: PlaygroundRequest):
    """Test an agent prompt/persona directly without saving it to the database."""
    try:
        debug_backend(f"/api/playground called with tools={request.tools}")
        selected_tools = [t for t in AEON_TOOLS if t.name in request.tools]
        debug_backend(f"Resolved tools for playground: {[t.name for t in selected_tools]}")
        temp_agent = create_react_agent(get_llm(), tools=selected_tools, prompt=request.persona)
        inputs = {"messages": [HumanMessage(content=request.prompt)]}
        debug_backend("Invoking temporary playground agent")
        result = temp_agent.invoke(inputs)
        debug_backend("Playground agent completed successfully")
        
        return {
            "status": "success",
            "persona_tested": request.persona,
            "tools_used": [t.name for t in selected_tools],
            "final_answer": result["messages"][-1].content
        }
    except Exception as e:
        debug_backend(f"Playground agent failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 🤖 AGENT CRUD ENDPOINTS ---
@app.get("/api/agents", response_model=List[AgentSchema], tags=["Agents"])
def list_all_agents(db: Session = Depends(get_db)):
    return db.query(DomainAgent).all()

@app.post("/api/agents", response_model=AgentSchema, tags=["Agents"])
def create_agent(agent: AgentSchema, db: Session = Depends(get_db)):
    db_agent = DomainAgent(**agent.model_dump())
    db.add(db_agent)
    db.commit()
    db.refresh(db_agent)
    return db_agent

@app.put("/api/agents/{agent_id}", response_model=AgentSchema, tags=["Agents"])
def update_agent(agent_id: str, request: AgentUpdateRequest, db: Session = Depends(get_db)):
    """Update an existing agent's configuration."""
    db_agent = db.query(DomainAgent).filter(DomainAgent.id == agent_id).first()
    if not db_agent:
        raise HTTPException(status_code=404, detail="Agent not found")
        
    update_data = request.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_agent, key, value)
        
    db.commit()
    db.refresh(db_agent)
    return db_agent

@app.delete("/api/agents/{agent_id}", tags=["Agents"])
def delete_agent(agent_id: str, db: Session = Depends(get_db)):
    """Delete an agent completely."""
    db_agent = db.query(DomainAgent).filter(DomainAgent.id == agent_id).first()
    if not db_agent:
        raise HTTPException(status_code=404, detail="Agent not found")
        
    db.delete(db_agent)
    db.commit()
    return {"message": f"Agent '{agent_id}' deleted successfully."}

# --- ⛓️ WORKFLOW CRUD ENDPOINTS ---
@app.get("/api/workflows", response_model=List[WorkflowSchema], tags=["Workflows"])
def list_workflows(db: Session = Depends(get_db)):
    return db.query(Workflow).all()

@app.post("/api/workflows", response_model=WorkflowSchema, tags=["Workflows"])
def create_workflow(request: WorkflowCreateRequest, db: Session = Depends(get_db)):
    """Create a new, empty workflow."""
    db_wf = Workflow(**request.model_dump())
    db.add(db_wf)
    db.commit()
    db.refresh(db_wf)
    return db_wf

@app.put("/api/workflows/{workflow_id}", response_model=WorkflowSchema, tags=["Workflows"])
def update_workflow(workflow_id: str, request: WorkflowUpdateRequest, db: Session = Depends(get_db)):
    """Update a workflow's name or description."""
    db_wf = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    if not db_wf:
        raise HTTPException(status_code=404, detail="Workflow not found")
        
    update_data = request.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_wf, key, value)
        
    db.commit()
    db.refresh(db_wf)
    return db_wf

@app.delete("/api/workflows/{workflow_id}", tags=["Workflows"])
def delete_workflow(workflow_id: str, db: Session = Depends(get_db)):
    """Delete a workflow."""
    db_wf = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    if not db_wf:
        raise HTTPException(status_code=404, detail="Workflow not found")
        
    db.delete(db_wf)
    db.commit()
    return {"message": f"Workflow '{workflow_id}' deleted successfully."}

@app.get("/api/workflows/{workflow_id}/agents", response_model=List[AgentSchema], tags=["Workflows"])
def get_agents_for_workflow(workflow_id: str, db: Session = Depends(get_db)):
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow.agents

@app.post("/api/workflows/{workflow_id}/map", tags=["Workflows"])
def map_agent_to_workflow(workflow_id: str, request: MapAgentRequest, db: Session = Depends(get_db)):
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    agent = db.query(DomainAgent).filter(DomainAgent.id == request.agent_id).first()
    
    if not workflow or not agent:
        raise HTTPException(status_code=404, detail="Workflow or Agent not found")
        
    if agent not in workflow.agents:
        workflow.agents.append(agent)
        db.commit()
        
    return {"message": f"Agent {agent.name} successfully mapped to {workflow.name}"}

@app.delete("/api/workflows/{workflow_id}/agents/{agent_id}", tags=["Workflows"])
def unmap_agent_from_workflow(workflow_id: str, agent_id: str, db: Session = Depends(get_db)):
    """Remove an agent from a workflow without deleting the agent entirely."""
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    agent = db.query(DomainAgent).filter(DomainAgent.id == agent_id).first()
    
    if not workflow or not agent:
        raise HTTPException(status_code=404, detail="Workflow or Agent not found")
        
    if agent in workflow.agents:
        workflow.agents.remove(agent)
        db.commit()
        return {"message": f"Agent {agent.name} unmapped from {workflow.name}"}
    return {"message": "Agent was not mapped to this workflow."}

# --- 🧠 MEMORY & EXECUTION ENDPOINTS ---
@app.get("/api/sessions/{thread_id}/history", tags=["Execution"])
def get_chat_history(thread_id: str):
    """Retrieve the conversation history for a specific thread from the LangGraph Checkpointer."""
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Fetch the thread state from the checkpointer
        state_tuple = workflow_memory.get(config)
        
        if not state_tuple:
            return {"thread_id": thread_id, "messages": []}
            
        # Extract messages from the state 
        state_data = state_tuple.channel_values if hasattr(state_tuple, 'channel_values') else state_tuple
        messages = state_data.get("messages", [])
        
        # Format them for the UI
        formatted_history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage) and msg.content:
                formatted_history.append({"role": "ai", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                formatted_history.append({"role": "tool", "content": f"[System: Executed tool '{msg.name}']"})
                
        return {
            "thread_id": thread_id,
            "messages": formatted_history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading history: {str(e)}")

# @app.post("/api/chat", tags=["Execution"])
# def execute_chat_workflow(
#     request: ChatRequest,
#     background_tasks: BackgroundTasks,
#     db: Session = Depends(get_db),
# ):
#     try:
#         session_id = request.session_id or str(uuid.uuid4())
#         debug_backend(
#             f"/api/chat called workflow_id={request.workflow_id} "
#             f"session_id={session_id} user={request.user_id} tenant={request.tenant_id}"
#         )

#         # ----- Decide which path to run --------------------------------------
#         use_auto = (not request.workflow_id) or request.workflow_id.lower() == "auto"

#         if not use_auto:
#             # ===== Existing explicit-workflow path (unchanged behavior) =====
#             config = {"configurable": {"thread_id": session_id}}
#             graph = build_dynamic_graph(request.workflow_id, db)
#             inputs = {"messages": [HumanMessage(content=request.prompt)]}

#             trace, final_answer = [], ""
#             for event in graph.stream(inputs, config=config, stream_mode="updates"):
#                 if not event:
#                     continue
#                 for node_name, state_update in event.items():
#                     trace.append(node_name)
#                     if state_update is not None:
#                         messages = state_update.get("messages")
#                         if messages and isinstance(messages, list) and len(messages) > 0:
#                             if hasattr(messages[-1], "content") and messages[-1].content:
#                                 final_answer = messages[-1].content

#             routed_plan = None  # explicit mode has no plan

#         else:
#             # ===== Auto-route path (planner + executor) ======================
#             plan: RoutePlan = plan_route(
#                 prompt=request.prompt,
#                 db=db,
#                 episodic_mem=episodic_memory,
#                 session_id=session_id,
#                 user_id=request.user_id,
#                 tenant_id=request.tenant_id or "default",
#             )
#             debug_backend(
#                 f"Router plan: mode={plan.mode} "
#                 f"steps={[s.workflow_id for s in plan.steps]} "
#                 f"confidence={plan.confidence}"
#             )
#             final_answer, trace = execute_plan(plan, request.prompt, session_id)
#             routed_plan = plan.model_dump()

#         # ----- Compute next turn index from the episodic store ---------------
#         prior = episodic_memory.recent(session_id=session_id, k=1)
#         next_turn_idx = (
#             (prior[0]["metadata"].get("turn_idx", -1) + 1) if prior else 0
#         )

#         # ----- Schedule episode write AFTER response is sent -----------------
#         background_tasks.add_task(
#             _write_episode_safe,
#             session_id=session_id,
#             turn_idx=next_turn_idx,
#             user_prompt=request.prompt,
#             final_answer=final_answer or "",
#             plan=routed_plan or {
#                 "mode": "single",
#                 "steps": [{"workflow_id": request.workflow_id, "subprompt": request.prompt}],
#                 "confidence": "high",
#             },
#             confidence=(routed_plan or {}).get("confidence", "high"),
#             tenant_id=request.tenant_id or "default",
#             user_id=request.user_id,
#         )

#         return {
#             "workflow_id": request.workflow_id if not use_auto else None,
#             "session_id": session_id,
#             "execution_trace": trace,
#             "final_answer": final_answer,
#             "routed_plan": routed_plan,  # None in explicit mode, populated in auto mode
#         }

#     except ValueError as ve:
#         debug_backend(f"Workflow execution validation error: {ve}")
#         raise HTTPException(status_code=404, detail=str(ve))
#     except Exception as e:
#         debug_backend(f"Workflow execution failed: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


## SSE Change 2: New streaming execution endpoint -------------------------------------------------------------

@app.post("/api/chat", tags=["Execution"])
def execute_chat_workflow(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    session_id = request.session_id or str(uuid.uuid4())
    debug_backend(f"/api/chat SSE called workflow_id={request.workflow_id} session_id={session_id}")

    use_auto = (not request.workflow_id) or request.workflow_id.lower() == "auto"

    def event_stream():
        # 1. Start the Stream
        yield f"data: {json.dumps({'type': 'status', 'message': 'Analyzing request...'})}\n\n"
        
        try:
            if not use_auto:
                # We mock a RoutePlan to reuse our new streaming executor
                mock_plan = RoutePlan(
                    mode="single",
                    steps=[{"workflow_id": request.workflow_id, "subprompt": request.prompt}],
                    confidence="high"
                )
                yield from execute_plan_stream(mock_plan, request.prompt, session_id)
            else:
                # 2. Plan Route
                plan: RoutePlan = plan_route(
                    prompt=request.prompt,
                    db=db,
                    episodic_mem=episodic_memory,
                    session_id=session_id,
                    user_id=request.user_id,
                    tenant_id=request.tenant_id or "default",
                )
                
                # Instantly tell the frontend what the Router decided!
                yield f"data: {json.dumps({'type': 'status', 'message': f'Router chose {plan.mode} mode with confidence: {plan.confidence}'})}\n\n"
                
                # 3. Execute and Stream
                # The generator yields "status", "token", and "complete" events natively
                final_answer = ""
                for sse_event in execute_plan_stream(plan, request.prompt, session_id):
                    yield sse_event
                    # We can parse the final answer out of the complete event to save to Episodic memory
                    if '"type": "complete"' in sse_event:
                        event_dict = json.loads(sse_event.replace("data: ", ""))
                        final_answer = event_dict.get("final_answer", "")

                # 4. Save to Memory (in the background, so it doesn't slow down the stream closure)
                prior = episodic_memory.recent(session_id=session_id, k=1)
                next_turn_idx = ((prior[0]["metadata"].get("turn_idx", -1) + 1) if prior else 0)
                
                background_tasks.add_task(
                    _write_episode_safe,
                    session_id=session_id,
                    turn_idx=next_turn_idx,
                    user_prompt=request.prompt,
                    final_answer=final_answer,
                    plan=plan.model_dump(),
                    confidence=plan.confidence,
                    tenant_id=request.tenant_id or "default",
                    user_id=request.user_id,
                )

        except Exception as e:
            debug_backend(f"Streaming execution failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    # Return the stream! The frontend must consume this using an EventSource or fetch reader.
    return StreamingResponse(event_stream(), media_type="text/event-stream")

### Change SSE 2 End



def _write_episode_safe(**kwargs) -> None:
    """Background task wrapper that never propagates errors to the response."""
    try:
        episodic_memory.record(**kwargs)
    except Exception as e:
        debug_backend(f"Episode write failed (non-fatal): {e}")

@app.post("/api/router/plan", tags=["Execution"])
def preview_route_plan(request: PlanPreviewRequest, db: Session = Depends(get_db)):
    """Return what the router WOULD do, without executing. Useful for UI preview."""
    session_id = request.session_id or "preview"
    try:
        plan = plan_route(
            prompt=request.prompt,
            db=db,
            episodic_mem=episodic_memory,
            session_id=session_id,
            user_id=request.user_id,
            tenant_id=request.tenant_id or "default",
        )
        return {"session_id": session_id, "plan": plan.model_dump()}
    except Exception as e:
        debug_backend(f"Plan preview failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/client-details", tags=["Delta API"])
def get_client_details():
    """Return a mock client list payload for the frontend."""
    return {
        "clients": [
            {
                "id": "thompson-family",
                "title": "Thompson Family",
                "subtitle": "Large discretionary spend detected",
                "detail": "43% increase in non-essential spending vs. 3-month baseline",
                "metric": "High",
                "urgency": "high",
                "signalType": "spending_anomaly",
                "impactScore": 92,
                "recommendedAction": "Open Summary"
            },
            {
                "id": "martinez-household",
                "title": "Martinez Household",
                "subtitle": "Goal progress off-track",
                "detail": "College savings falling behind by $12,400 this quarter",
                "metric": "Med",
                "urgency": "medium",
                "signalType": "goal_slippage",
                "impactScore": 78,
                "recommendedAction": "Open Summary"
            },
            {
                "id": "chen-associates",
                "title": "Chen & Associates",
                "subtitle": "Portfolio drift detected",
                "detail": "Equity allocation 8% over target after market rally",
                "metric": "Med",
                "urgency": "medium",
                "signalType": "allocation_drift",
                "impactScore": 74,
                "recommendedAction": "Open Summary"
            },
            {
                "id": "williams-estate",
                "title": "Williams Estate",
                "subtitle": "Tax-loss opportunity",
                "detail": "Potential $18K harvest available before year-end",
                "metric": "High",
                "urgency": "high",
                "signalType": "tax_harvest_opportunity",
                "impactScore": 88,
                "recommendedAction": "Open Summary"
            }
        ],
        "total": 4,
        "asOf": "2026-05-13T10:30:00Z"
    }


@app.get("/api/opportunities", tags=["Delta API"])
def get_opportunities():
    """Return a list of top opportunities to prioritize."""
    return {
        "opportunities": [
            {
            "id": "1",
            "clientName": "Zara & Barry Block",
            "title": "Cash Drag Detected",
            "priority": "high",
            "description": "Large idle cash position of $650K leading to under-investment vs long-term goals",
            "metric": "Impact: 9.2",
            "urgency": "high",
            "opportunityId": "cash-drag"
            },
            {
            "id": "2",
            "clientName": "Marcus & Diana Chen",
            "title": "RSU Concentration Risk",
            "priority": "high",
            "description": "75% of $2.1M portfolio in employer stock after recent vesting cycle",
            "metric": "Impact: 9.0",
            "urgency": "high",
            "opportunityId": "rsu-concentration"
            },
            {
            "id": "3",
            "clientName": "Jennifer Okonkwo",
            "title": "Retirement Catch-Up Window",
            "priority": "high",
            "description": "Age 52, under-funded 401(k) with $180K gap to target retirement income",
            "metric": "Impact: 8.7",
            "urgency": "high",
            "opportunityId": "retirement-catchup"
            },
            {
            "id": "4",
            "clientName": "David & Rachel Kim",
            "title": "Estate Planning Gap",
            "priority": "medium",
            "description": "Net worth crossed $5M threshold; no trust structure or beneficiary updates since 2019",
            "metric": "Impact: 8.3",
            "urgency": "medium",
            "opportunityId": "estate-planning"
            },
            {
            "id": "5",
            "clientName": "Sofia Ramirez",
            "title": "Business Exit Opportunity",
            "priority": "medium",
            "description": "Consulting practice valued at $1.2M; interested buyers identified, no succession plan",
            "metric": "Impact: 8.1",
            "urgency": "medium",
            "opportunityId": "business-exit"
            },
            {
            "id": "6",
            "clientName": "Tom & Lisa Brennan",
            "title": "Insurance Coverage Review",
            "priority": "medium",
            "description": "Life insurance gap of $500K identified; term policy expiring in 8 months",
            "metric": "Impact: 7.6",
            "urgency": "medium",
            "opportunityId": "insurance-review"
            }
        ]
}

@app.get("/api/upcoming-meetings", tags=["Delta API"])
def get_upcoming_meetings():
    """Return a list of upcoming client and business meetings."""
    return {
  "meetings": [
    {
      "id": "meeting-1",
      "title": "Meeting with Thomas Anderson",
      "subtitle": "Today 2pm",
      "detail": "529 plans, life insurance for new baby, estate planning, asset transfer follow-up",
      "metric": "Pending",
      "urgency": "medium",
      "scheduledFor": "2026-05-13T14:00:00-04:00",
      "type": "client-meeting",
      "clientId": "thomas-anderson",
      "status": "pending"
    },
    {
      "id": "meeting-2",
      "title": "Sarah Johnson - Q4 Review",
      "subtitle": "In 2 hours",
      "detail": "Portfolio performance review, rebalancing discussion, tax planning",
      "metric": "Prepped",
      "urgency": "high",
      "scheduledFor": "2026-05-13T16:00:00-04:00",
      "type": "client-meeting",
      "clientId": "sarah-johnson",
      "status": "prepped"
    },
    {
      "id": "meeting-3",
      "title": "Miller Enterprises",
      "subtitle": "Tomorrow 10am",
      "detail": "Business succession planning, liquidity event discussion",
      "metric": "Ready",
      "urgency": "medium",
      "scheduledFor": "2026-05-14T10:00:00-04:00",
      "type": "business-meeting",
      "clientId": "miller-enterprises",
      "status": "ready"
    },
    {
      "id": "meeting-4",
      "title": "Davidson Retirement",
      "subtitle": "Friday 2pm",
      "detail": "Social Security claiming strategy, income planning review",
      "metric": "Pending",
      "urgency": "low",
      "scheduledFor": "2026-05-17T14:00:00-04:00",
      "type": "client-meeting",
      "clientId": "davidson-retirement",
      "status": "pending"
    }
  ]
}

@app.get("/api/tasks-and-followups", tags=["Delta API"])
def tasks_and_followups():
    """Return a list of top tasks and follow-ups."""
    return {
        "tasks": [
            {
            "id": "task-001",
            "type": "task",
            "title": "Follow up: Thompson spending",
            "subtitle": "Created by AI from recent alert",
            "detail": "Draft check-in message about unusual spending patterns",
            "metric": "Today",
            "urgency": "high",
            "status": "pending",
            "source": "ai",
            "clientId": "client-thompson",
            "clientName": "Thompson",
            "dueDate": "2026-05-14",
            "createdAt": "2026-05-14T08:00:00Z"
            },
            {
            "id": "task-002",
            "type": "task",
            "title": "Send: Chen rebalancing proposal",
            "subtitle": "AI-generated recommendation",
            "detail": "Review and send portfolio rebalancing suggestion",
            "metric": "Today",
            "urgency": "high",
            "status": "pending",
            "source": "ai",
            "clientId": "client-chen",
            "clientName": "Chen",
            "dueDate": "2026-05-14",
            "createdAt": "2026-05-14T08:00:00Z"
            },
            {
            "id": "task-003",
            "type": "task",
            "title": "Schedule: Williams tax planning",
            "subtitle": "Opportunity window closing",
            "detail": "Book meeting to discuss tax-loss harvesting by Oct 15",
            "metric": "This Week",
            "urgency": "medium",
            "status": "pending",
            "source": "ai",
            "clientId": "client-williams",
            "clientName": "Williams",
            "dueDate": "2026-05-18",
            "createdAt": "2026-05-14T08:00:00Z"
            }
        ],
        "meta": {
            "total": 3,
            "pending": 3,
            "completed": 0
        }
    }
@app.get("/api/tools", tags=["Delta API"])
def tools_api():
    """"Return a list of available tools and their descriptions."""
    return {
  "tools": {
    "commandPane": [
      {
        "id": "prospecting",
        "name": "Prospecting",
        "icon": "UserSearch",
        "route": "/prospecting",
        "status": "active",
        "backendRequired": false
      },
      {
        "id": "financial-planning",
        "name": "Financial Planning",
        "icon": "FileText",
        "route": "/financial-planning",
        "status": "active",
        "backendRequired": false
      },
      {
        "id": "portfolio-analysis",
        "name": "Portfolio Analysis",
        "icon": "BarChart3",
        "route": "/portfolio-analysis",
        "status": "stub",
        "backendRequired": true
      },
      {
        "id": "client-management",
        "name": "Client Management",
        "icon": "Users",
        "route": "/client-management",
        "status": "stub",
        "backendRequired": true
      },
      {
        "id": "proposal-generator",
        "name": "Proposal Generator",
        "icon": "FileSignature",
        "route": "/proposal-generator",
        "status": "stub",
        "backendRequired": true
      },
      {
        "id": "performance-reports",
        "name": "Performance Reports",
        "icon": "TrendingUp",
        "route": "/performance-reports",
        "status": "stub",
        "backendRequired": true
      }
    ],
    "intelligence": [
      {
        "id": "market-insights",
        "name": "Market Insights",
        "icon": "BarChart3",
        "route": "/market-insights",
        "status": "stub",
        "backendRequired": true
      },
      {
        "id": "client-analytics",
        "name": "Client Analytics",
        "icon": "PieChart",
        "route": "/client-analytics",
        "status": "stub",
        "backendRequired": true
      }
    ]
  }
}   