# src/engine/router.py
"""
Router for the Aeon backend.

Given a user prompt, decide which workflow(s) should answer and orchestrate
their execution. Built as a thin layer on top of the existing
`build_dynamic_graph` primitive — we do not modify any wf_xxx.py.

Public surface:
    plan_route(...)   -> RoutePlan        (LLM call, no execution)
    execute_plan(...) -> (answer, trace)  (runs the plan)
    get_workflow_catalog(db)              (DB read, 60s cached)
"""

from __future__ import annotations

import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple, Literal, Dict, Any

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

## CHange for SSE1 Start
import queue
from fastapi.responses import StreamingResponse

## End Change



from src.db.database import SessionLocal
from src.db.models import Workflow
from src.engine.dynamic_graph import build_dynamic_graph, get_llm
from src.memory.episodic import EpisodicMemory


# --- 1. Plan schema ----------------------------------------------------------
RouteMode = Literal["single", "sequential", "parallel", "clarify", "reject"]
Confidence = Literal["high", "medium", "low"]


class RouteStep(BaseModel):
    workflow_id: str = Field(..., description="One of the catalog workflow_ids.")
    subprompt: str = Field(
        ...,
        description=(
            "The slice of the user's request this workflow should answer. "
            "Rewrite the user's prompt into a focused sub-question."
        ),
    )


class RoutePlan(BaseModel):
    mode: RouteMode = Field(
        ...,
        description=(
            "single: one workflow answers everything. "
            "sequential: step N consumes step N-1's output. "
            "parallel: independent steps, fan-out then merge. "
            "clarify: ask the user one question instead of executing. "
            "reject: request is out of scope for any workflow."
        ),
    )
    steps: List[RouteStep] = Field(
        default_factory=list,
        description="Max 3 steps. Empty for mode in {clarify, reject}.",
    )
    confidence: Confidence = "medium"
    reason: str = Field(default="", description="One short sentence; for logs/UI.")
    clarification: Optional[str] = Field(
        default=None,
        description="Required if mode == 'clarify'. The question to ask the user.",
    )


# --- 2. Workflow catalog (DB-driven, 60s in-process cache) -------------------
_catalog_cache: Dict[str, Any] = {"ts": 0.0, "value": None}
_catalog_lock = threading.Lock()
_CATALOG_TTL_SECONDS = 60.0


def get_workflow_catalog(db: Session) -> List[Dict[str, Any]]:
    """Return all workflows + their attached agents' routing descriptions."""
    now = time.time()
    with _catalog_lock:
        if (
            _catalog_cache["value"] is not None
            and now - _catalog_cache["ts"] < _CATALOG_TTL_SECONDS
        ):
            return _catalog_cache["value"]

    workflows = db.query(Workflow).all()
    catalog = []
    for wf in workflows:
        catalog.append({
            "workflow_id": wf.id,
            "name": wf.name,
            "description": wf.description or "",
            "agents": [
                {"id": a.id, "routing_description": a.routing_description or ""}
                for a in (wf.agents or [])
            ],
        })

    with _catalog_lock:
        _catalog_cache["ts"] = now
        _catalog_cache["value"] = catalog
    return catalog


def invalidate_catalog_cache() -> None:
    """Call from any /api/workflows or /api/agents mutation endpoint."""
    with _catalog_lock:
        _catalog_cache["ts"] = 0.0
        _catalog_cache["value"] = None


def _render_catalog_for_prompt(catalog: List[Dict[str, Any]]) -> str:
    lines = []
    for wf in catalog:
        lines.append(f"- {wf['workflow_id']} — {wf['name']}")
        if wf.get("description"):
            lines.append(f"    description: {wf['description']}")
        for a in wf.get("agents", []):
            lines.append(f"    • agent {a['id']}: {a['routing_description']}")
    return "\n".join(lines) if lines else "(no workflows defined)"


def _render_episodes_for_prompt(rows: List[Dict[str, Any]], header: str) -> str:
    if not rows:
        return f"{header}: (none)"
    out = [f"{header}:"]
    ## Change 1 Start: render the full conversation turn instead of just metadata
    for r in rows:
        meta = r.get("metadata", {})
        # FIX: Grab the whole document and replace newlines so it stays compact
        full_doc = r.get('document', '').replace('\n---\n', ' | AI Answer: ')[:500] 
        out.append(
            f"  - Turn: {full_doc}\n"
            f"    plan_workflows: {meta.get('workflow_ids', '')}  "
            f"(confidence: {meta.get('confidence', '?')})"
        )
    return "\n".join(out)
    ##Change 1 End


### Change 2 Start
# --- 3. Planner --------------------------------------------------------------
_PLANNER_SYSTEM = """\
You are the Router for the Aeon Wealth backend. Your job is to decide which
workflow(s) should answer the user's request.

RULES:
- Choose ONLY from workflows listed in the catalog below. Never invent ids.
- Prefer mode="single" unless the request clearly spans multiple workflows.
- Use mode="sequential" when one step's output is needed by the next.
- Use mode="parallel" only for truly independent sub-tasks.
- Use mode="clarify" (and fill 'clarification') if the request is ambiguous.
- Use mode="reject" if no workflow fits.
- At most 3 steps total. Never repeat the same workflow_id in one plan.
- CRITICAL: If the user's prompt contains pronouns (he, she, it, they, this), you MUST look at the 'RECENT TURNS' to figure out who or what they are talking about.
- Each step's 'subprompt' MUST be a focused rewrite of the user's request. You MUST replace all pronouns with the actual client names or entities from the recent context. Do not just copy the original prompt.
- Set confidence honestly. If similar past episodes strongly match, use "high".
- Prefer mode="single" unless the request clearly spans MULTIPLE DIFFERENT workflows. If the user asks for multiple things that fall under the SAME workflow (like comparing two clients), use mode="single" and ask for them together.
"""

### Change 2 End

_PLANNER_HUMAN = """\
WORKFLOW CATALOG:
{catalog}

RECENT TURNS IN THIS SESSION (most recent first):
{recent}

SIMILAR PAST EPISODES (for cross-session learning):
{similar}

NEW USER PROMPT:
{prompt}

Return a RoutePlan.
"""


def plan_route(
    prompt: str,
    db: Session,
    episodic_mem: EpisodicMemory,
    session_id: str,
    user_id: Optional[str] = None,
    tenant_id: str = "default",
    max_steps: int = 3,
) -> RoutePlan:
    """Single LLM call that returns a structured plan."""
    catalog = get_workflow_catalog(db)
    recent = episodic_mem.recent(session_id=session_id, k=3)
    similar = episodic_mem.similar(
        prompt=prompt,
        k=3,
        tenant_id=tenant_id,
        user_id=user_id,
        exclude_session_id=session_id,
    )

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", _PLANNER_SYSTEM),
        ("human", _PLANNER_HUMAN),
    ])
    chain = chat_prompt | get_llm().with_structured_output(RoutePlan)
    plan: RoutePlan = chain.invoke({
        "catalog": _render_catalog_for_prompt(catalog),
        "recent": _render_episodes_for_prompt(recent, "Recent session episodes"),
        "similar": _render_episodes_for_prompt(similar, "Similar past episodes"),
        "prompt": prompt,
    })

    ### Change 3 : Execution level corrections

    # Safety rails.
    valid_ids = {wf["workflow_id"] for wf in catalog}
    plan.steps = [s for s in plan.steps if s.workflow_id in valid_ids][:max_steps]
    
    # 🛠️ NEW: Merge duplicates instead of dropping them
    merged_steps = {}
    dedup = []
    for s in plan.steps:
        if s.workflow_id not in merged_steps:
            merged_steps[s.workflow_id] = s
            dedup.append(s)
        else:
            # If the router calls the same workflow again, append the request
            merged_steps[s.workflow_id].subprompt += f"\n\nAlso address this: {s.subprompt}"
            
    plan.steps = dedup

    # 🛠️ NEW: Ensure mode falls back to 'single' if we consolidated everything into one step
    if plan.mode in ("single", "sequential", "parallel") and not plan.steps:
        plan.mode = "clarify"
        # ... (keep existing clarify text)
        
    if plan.mode == "single" and len(plan.steps) > 1:
        plan.mode = "sequential"
    if plan.mode in ("sequential", "parallel") and len(plan.steps) == 1:
        plan.mode = "single"
        
    return plan

    ### Change 3 End

# --- 4. Executor -------------------------------------------------------------
def _run_one_step(
    workflow_id: str,
    composed_prompt: str,
    session_id: str,
) -> Tuple[str, List[str]]:
    """Run one workflow. Returns (final_answer, node_trace).

    Each step gets its OWN SQLAlchemy session — required for thread safety
    when called concurrently from execute_plan().
    """
    db = SessionLocal()
    try:
        graph = build_dynamic_graph(workflow_id, db)
        thread_id = f"{session_id}::{workflow_id}::{int(time.time())}" ## Change for better traceability in concurrent runs
        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"messages": [HumanMessage(content=composed_prompt)]}

        trace: List[str] = []
        final_answer = ""
        for event in graph.stream(inputs, config=config, stream_mode="updates"):
            if not event:
                continue
            for node_name, state_update in event.items():
                trace.append(f"{workflow_id}:{node_name}")
                if state_update and isinstance(state_update, dict):
                    msgs = state_update.get("messages")
                    if msgs and isinstance(msgs, list) and msgs:
                        last = msgs[-1]
                        if getattr(last, "content", None):
                            final_answer = last.content
        return final_answer, trace
    finally:
        db.close()


_MERGE_SYSTEM = (
    "You combine outputs from multiple specialized workflows into ONE cohesive "
    "answer for the end user. Be concise. Preserve all key facts. Do not invent."
)


def _merge_answers(original_prompt: str, parts: List[Tuple[str, str]]) -> str:
    """parts = [(workflow_id, answer), ...]. Returns merged text."""
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0][1]

    sections = "\n\n".join(
        f"### From {wf_id}\n{ans}" for wf_id, ans in parts if ans
    )
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", _MERGE_SYSTEM),
        ("human",
         "Original user request:\n{prompt}\n\n"
         "Workflow outputs to combine:\n{sections}\n\n"
         "Produce the single best answer for the user."),
    ])
    chain = chat_prompt | get_llm()
    result = chain.invoke({"prompt": original_prompt, "sections": sections})
    return getattr(result, "content", "") or ""


def execute_plan(
    plan: RoutePlan,
    original_prompt: str,
    session_id: str,
) -> Tuple[str, List[str]]:
    """Run the plan. Returns (final_answer, execution_trace)."""
    trace: List[str] = [f"router:{plan.mode}"]

    if plan.mode == "clarify":
        return (plan.clarification or "Could you clarify your request?", trace)
    if plan.mode == "reject":
        return (plan.reason or "This request is outside the supported workflows.", trace)

    if plan.mode == "single":
        ans, t = _run_one_step(
            plan.steps[0].workflow_id, plan.steps[0].subprompt, session_id
        )
        return ans, trace + t

    if plan.mode == "sequential":
        prev_answer = ""
        parts: List[Tuple[str, str]] = []
        for step in plan.steps:
            composed = step.subprompt
            if prev_answer:
                composed = (
                    f"Context from prior step:\n{prev_answer}\n\n"
                    f"Now: {step.subprompt}"
                )
            ans, t = _run_one_step(step.workflow_id, composed, session_id)
            trace.extend(t)
            parts.append((step.workflow_id, ans))
            prev_answer = ans
        trace.append("router:merge")
        return _merge_answers(original_prompt, parts), trace

    if plan.mode == "parallel":
        parts: List[Tuple[str, str]] = []
        # Bounded fan-out: at most len(steps), capped at 3 by the planner.
        with ThreadPoolExecutor(max_workers=len(plan.steps)) as pool:
            futures = {
                pool.submit(
                    _run_one_step, step.workflow_id, step.subprompt, session_id
                ): step.workflow_id
                for step in plan.steps
            }
            for fut in as_completed(futures):
                wf_id = futures[fut]
                ans, t = fut.result()
                trace.extend(t)
                parts.append((wf_id, ans))
        # Preserve original step order for consistent merging.
        order = {s.workflow_id: i for i, s in enumerate(plan.steps)}
        parts.sort(key=lambda p: order.get(p[0], 99))
        trace.append("router:merge")
        return _merge_answers(original_prompt, parts), trace

    # Fallback (shouldn't reach here).
    return ("Router produced an unsupported mode.", trace)


##. Change SSE 2 Start: Streaming version of the executor -------------------------------------------------------------
# --- 5. Streaming Executor (SSE) ---------------------------------------------
def _run_one_step_stream(
    workflow_id: str,
    composed_prompt: str,
    session_id: str,
    event_queue: queue.Queue
) -> None:
    """Runs a workflow and pushes real-time events to the queue."""
    db = SessionLocal()
    try:
        graph = build_dynamic_graph(workflow_id, db)
        thread_id = f"{session_id}::{workflow_id}::{int(time.time())}" ## Change for better traceability in concurrent runs
        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"messages": [HumanMessage(content=composed_prompt)]}

        trace = []
        final_answer = ""
        
        event_queue.put({"type": "status", "message": f"Initializing {workflow_id}..."})

        # stream_mode="messages" allows us to catch token-by-token LLM output
        for event in graph.stream(inputs, config=config, stream_mode="messages"):
            msg, metadata = event
            
            # If it's a token streaming from the AI
            if msg.content and metadata.get("langgraph_node") == "synthesizer":
                event_queue.put({"type": "token", "chunk": msg.content})
                final_answer += msg.content
                
            # If it's a tool call or intermediate agent routing
            elif metadata.get("langgraph_node") != "synthesizer" and msg.name:
                 event_queue.put({"type": "status", "message": f"Running tool: {msg.name}..."})
                 trace.append(f"{workflow_id}:{msg.name}")

        event_queue.put({"type": "step_complete", "workflow_id": workflow_id, "answer": final_answer, "trace": trace})
    except Exception as e:
        event_queue.put({"type": "error", "message": str(e)})
    finally:
        db.close()

def execute_plan_stream(plan: RoutePlan, original_prompt: str, session_id: str):
    """Generator that yields SSE JSON strings for the FastAPI StreamingResponse."""
    q = queue.Queue()
    trace = [f"router:{plan.mode}"]
    
    if plan.mode in ("clarify", "reject"):
        ans = plan.clarification if plan.mode == "clarify" else plan.reason
        yield f"data: {json.dumps({'type': 'status', 'message': 'Routing complete.'})}\n\n"
        yield f"data: {json.dumps({'type': 'token', 'chunk': ans})}\n\n"
        yield f"data: {json.dumps({'type': 'complete', 'final_answer': ans, 'trace': trace})}\n\n"
        return

    # Helper function to run the threads
    def _run_plan():
        if plan.mode == "single":
            _run_one_step_stream(plan.steps[0].workflow_id, plan.steps[0].subprompt, session_id, q)
            
        elif plan.mode == "sequential":
            # Simplified sequential for brevity - runs sequentially and pushes to queue
            prev_answer = ""
            for step in plan.steps:
                composed = step.subprompt if not prev_answer else f"Prior Context:\n{prev_answer}\n\nNow: {step.subprompt}"
                _run_one_step_stream(step.workflow_id, composed, session_id, q)
                # We would extract the answer from the step_complete event to feed the next step

        elif plan.mode == "parallel":
            q.put({"type": "status", "message": f"Dispatching {len(plan.steps)} parallel agents..."})
            with ThreadPoolExecutor(max_workers=len(plan.steps)) as pool:
                futures = [pool.submit(_run_one_step_stream, step.workflow_id, step.subprompt, session_id, q) for step in plan.steps]
                for fut in as_completed(futures):
                    pass # Threads will automatically put their results in the queue

        q.put({"type": "plan_finished"})

    # Start execution in a background thread so the main thread can yield to the HTTP response
    threading.Thread(target=_run_plan, daemon=True).start()

    final_parts = []
    completed_steps = 0
    total_steps = len(plan.steps)

    while True:
        event = q.get()
        if event["type"] == "plan_finished":
            break
        
        if event["type"] == "step_complete":
            trace.extend(event["trace"])
            final_parts.append((event["workflow_id"], event["answer"]))
            completed_steps += 1
            if completed_steps < total_steps:
                 continue # Wait for other parallel steps
                 
        # Yield the real-time event to the frontend
        yield f"data: {json.dumps(event)}\n\n"

    ### Change for SSE

    # # Merge parallel/sequential answers if needed
    # yield f"data: {json.dumps({'type': 'status', 'message': 'Synthesizing final report...'})}\n\n"
    # final_merged = _merge_answers(original_prompt, final_parts)
    # yield f"data: {json.dumps({'type': 'token', 'chunk': final_merged})}\n\n"
    
    # yield f"data: {json.dumps({'type': 'complete', 'final_answer': final_merged, 'trace': trace})}\n\n"
    # ONLY merge and yield a final chunk if we actually ran multiple workflows
    if plan.mode in ("sequential", "parallel") and len(final_parts) > 1:
        yield f"data: {json.dumps({'type': 'status', 'message': 'Synthesizing final report...'})}\n\n"
        final_merged = _merge_answers(original_prompt, final_parts)
        
        # Yield the merged text so the UI prints it
        yield f"data: {json.dumps({'type': 'token', 'chunk': '\n\n' + final_merged})}\n\n"
    else:
        # In single mode, the LangGraph agent ALREADY streamed the text to the UI token-by-token!
        # We just extract the string here so we can pass it to Episodic Memory in the background.
        final_merged = final_parts[0][1] if final_parts else "No response generated."
    
    # Send the completion event (without appending more tokens to the screen)
    yield f"data: {json.dumps({'type': 'complete', 'final_answer': final_merged, 'trace': trace})}\n\n"

    ### End Change for SSE


## Change SSE 2 End
