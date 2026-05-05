# main.py
import uuid
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel

from src.db.database import engine, SessionLocal, Base
from src.db.models import DomainAgent, Workflow

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.engine.dynamic_graph import build_dynamic_graph, get_llm, workflow_memory
from src.agents.tools import AEON_TOOLS
from langgraph.prebuilt import create_react_agent

# Ensure tables exist
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Aeon Agent Factory API", 
    description="Headless backend for dynamic LangGraph workflows.",
    version="2.0"
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

    class Config:
        from_attributes = True

class WorkflowCreateRequest(BaseModel):
    id: str
    name: str
    description: str

class WorkflowUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class MapAgentRequest(BaseModel):
    agent_id: str

class ChatRequest(BaseModel):
    workflow_id: str
    prompt: str
    session_id: Optional[str] = None

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
        selected_tools = [t for t in AEON_TOOLS if t.name in request.tools]
        temp_agent = create_react_agent(get_llm(), tools=selected_tools, prompt=request.persona)
        inputs = {"messages": [HumanMessage(content=request.prompt)]}
        result = temp_agent.invoke(inputs)
        
        return {
            "status": "success",
            "persona_tested": request.persona,
            "tools_used": [t.name for t in selected_tools],
            "final_answer": result["messages"][-1].content
        }
    except Exception as e:
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

@app.post("/api/chat", tags=["Execution"])
def execute_chat_workflow(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": session_id}}
        
        graph = build_dynamic_graph(request.workflow_id, db) 
        
        inputs = {"messages": [HumanMessage(content=request.prompt)]}
        trace = []
        final_answer = ""
        
        for event in graph.stream(inputs, config=config, stream_mode="updates"):
            if not event:
                continue
                
            for node_name, state_update in event.items():
                trace.append(node_name)
                
                if state_update is not None:
                    messages = state_update.get("messages")
                    if messages and isinstance(messages, list) and len(messages) > 0:
                        if hasattr(messages[-1], 'content') and messages[-1].content:
                            final_answer = messages[-1].content

        return {
            "workflow_id": request.workflow_id,
            "session_id": session_id,
            "execution_trace": trace,
            "final_answer": final_answer
        }
        
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))