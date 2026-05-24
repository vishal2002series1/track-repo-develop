# streamlit_app.py
import streamlit as st
import requests
import pandas as pd
import json
import requests
import streamlit as st
import re

## Change for Showing Graphs in Streamlit

import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# WIDGET RENDERER — parses ```json:widget and ```json:suggestions blocks
# Supports multiple widgets and a single suggestions block in one response.
# ---------------------------------------------------------------------------
WIDGET_PATTERN = re.compile(r"```json:widget\s*(\{.*?\})\s*```", re.DOTALL)
SUGGESTIONS_PATTERN = re.compile(r"```json:suggestions\s*(\{.*?\})\s*```", re.DOTALL)
CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def _render_single_widget(spec: dict):
    """Render one widget spec dict as a Streamlit Plotly chart."""
    chart_type = spec.get("chart_type", "").lower()
    title      = spec.get("title", "")
    labels     = spec.get("labels", [])
    values     = spec.get("values", [])
    x          = spec.get("x") if spec.get("x") is not None else labels
    y          = spec.get("y") if spec.get("y") is not None else values

    try:
        if chart_type in ("donut", "pie"):
            hole = 0.45 if chart_type == "donut" else 0.0
            fig = go.Figure(go.Pie(labels=labels, values=values, hole=hole,
                                   textinfo="label+percent", hoverinfo="label+value+percent"))
            fig.update_layout(title_text=title, showlegend=True)

        elif chart_type == "bar":
            fig = px.bar(x=x, y=y, labels={"x": spec.get("x_label",""), "y": spec.get("y_label","")},
                         title=title, text_auto=True)

        elif chart_type == "line":
            fig = px.line(x=x, y=y, labels={"x": spec.get("x_label",""), "y": spec.get("y_label","")},
                          title=title, markers=True)

        elif chart_type == "area":
            fig = px.area(x=x, y=y, labels={"x": spec.get("x_label",""), "y": spec.get("y_label","")},
                          title=title)

        elif chart_type == "scatter":
            point_labels = spec.get("labels", [])
            fig = px.scatter(
                x=x, y=y,
                text=point_labels if point_labels else None,
                labels={"x": spec.get("x_label", "X"), "y": spec.get("y_label", "Y")},
                title=title
            )
            if point_labels:
                fig.update_traces(textposition="top center")

        elif chart_type == "histogram":
            fig = px.histogram(x=x if x else values, nbins=spec.get("bins", 20), title=title)

        elif chart_type == "table":
            columns = spec.get("columns", labels)
            rows    = spec.get("rows", [])
            if rows:
                fig = go.Figure(go.Table(
                    header=dict(values=columns, fill_color="#0078D4", font=dict(color="white")),
                    cells=dict(values=list(zip(*rows)) if rows else [])
                ))
                fig.update_layout(title_text=title)
            else:
                st.warning(f"Table widget '{title}' has no rows.")
                return
        else:
            st.warning(f"Unknown chart_type '{chart_type}' — skipping widget.")
            return

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to render widget '{title}': {e}")


def _style_citations(text: str) -> str:
    """Turn [1] [2] etc. into small clickable-looking blue badges."""
    def repl(m):
        n = m.group(1)
        return (
            f" <a href='#source-{n}' style='text-decoration:none;color:#0078D4;"
            f"font-size:0.85em;background-color:#E7F1FB;padding:1px 6px;"
            f"border-radius:4px;border:1px solid #c8e0f4;font-weight:600;'>{n}</a> "
        )
    return CITATION_PATTERN.sub(repl, text)


def render_response_with_widgets(container, text: str, streaming: bool = False):
    """
    Splits `text` into:
      - prose (with citation numbers styled as badges)
      - ```json:widget``` blocks (rendered as Plotly charts)
      - ```json:suggestions``` block (returned for separate rendering)

    Returns: list of suggestion strings (empty if none found).
    """
    # Extract suggestions FIRST so they don't leak into prose
    suggestions = []
    sug_match = SUGGESTIONS_PATTERN.search(text)
    if sug_match and not streaming:
        try:
            sug_data = json.loads(sug_match.group(1))
            suggestions = sug_data.get("suggestions", []) or []
        except json.JSONDecodeError:
            pass
    # Strip the suggestions block from displayed text either way
    text_no_sug = SUGGESTIONS_PATTERN.sub("", text)

    # Split on widget blocks
    parts = WIDGET_PATTERN.split(text_no_sug)
    prose_parts = parts[0::2]
    json_parts  = parts[1::2]

    full_prose = "".join(prose_parts).strip()
    cursor     = " ▌" if streaming else ""
    if full_prose or streaming:
        styled = _style_citations(full_prose)
        container.markdown(styled + cursor, unsafe_allow_html=True)

    if not streaming:
        for raw_json in json_parts:
            try:
                spec = json.loads(raw_json)
                _render_single_widget(spec)
            except json.JSONDecodeError:
                st.warning("Could not parse widget JSON — skipping.")

    return suggestions


def render_sources_panel(sources: list):
    """Render the numbered sources list below an AI response.

    sources is a list of dicts: [{"sequence": 1, "source": "PortfolioData",
                                  "tool": "execute_sql", "content": "..."}]
    """
    if not sources:
        return
    with st.expander(f"📎 Sources ({len(sources)})", expanded=False):
        for src in sources:
            n     = src.get("sequence", "?")
            label = src.get("source", "data")
            tool  = src.get("tool", "")
            body  = src.get("content", "")
            st.markdown(f"**[{n}] {label}** · `{tool}`")
            st.code(body[:4000], language="text")
            if len(body) > 4000:
                st.caption(f"…showing first 4000 of {len(body)} chars")
            st.divider()


# --- CONFIGURATION ---
st.set_page_config(page_title="Aeon Agent Factory", layout="wide")
API_BASE = "http://127.0.0.1:8000/api"

# --- Playground Presets ---
PLAYGROUND_EXAMPLE_AGENTS = {
    "Portfolio Risk Analyst": {
        "persona": "You are a portfolio risk analyst. Use tools to evaluate concentration, identify single-name risk, and summarize risk mitigation options in concise bullet points.",
        "tools": ["get_database_schema", "execute_sql", "compute_portfolio_concentration"],
        "prompt": "For CLI-002, assess portfolio concentration risk, include HHI-style interpretation, and suggest top 3 diversification actions.Also provide personal details",
    },
    "Compliance Auditor": {
        "persona": "You are a strict compliance auditor. Check for missing signatures, stale KYC, and unresolved compliance flags. Report issues by severity and recommend next actions.",
        "tools": ["get_database_schema", "execute_sql"],
        "prompt": "Run a compliance review for CLI-002 and list unresolved flags by severity with recommended remediation steps.",
    },
    "Client Communications Reviewer": {
        "persona": "You analyze client communications for sentiment, urgency, and unresolved asks. Use transcript and email search, then provide a short action-focused summary.",
        "tools": ["search_transcripts", "search_client_emails", "execute_sql"],
        "prompt": "For CLI-002, analyze recent transcripts and emails for urgency, sentiment, and open requests, then provide follow-up actions.",
    },
    "Meeting Prep Assistant": {
        "persona": "You prepare pre-meeting briefs for advisors. Extract key client context, open action items, and agenda recommendations.",
        "tools": ["get_database_schema", "execute_sql", "search_transcripts", "search_client_emails"],
        "prompt": "Prepare a meeting brief for CLI-001 with agenda items, unresolved issues, and recommended talking points.",
    },
}


def suggest_prompt_from_tools(tools):
    tool_set = set(tools or [])

    if "compute_portfolio_concentration" in tool_set:
        return "For client_id 2, evaluate concentration risk and provide practical diversification recommendations."

    if "search_transcripts" in tool_set and "search_client_emails" in tool_set:
        return "For client_id 2, summarize key concerns from transcripts and emails, then list immediate advisor follow-ups."

    if "execute_sql" in tool_set and "get_database_schema" in tool_set:
        return "For client_id 2, retrieve the relevant records and provide a concise risk and action summary."

    if "execute_sql" in tool_set:
        return "Use SQL-backed data to provide a concise status summary for client_id 2 with next best actions."

    return "Ask a question to test persona behavior and tool usage."

# --- HELPER FUNCTIONS ---
def fetch_data(endpoint):
    try:
        response = requests.get(f"{API_BASE}/{endpoint}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return []

def post_data(endpoint, payload):
    try:
        response = requests.post(f"{API_BASE}/{endpoint}", json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error posting data: {e}")
        return None

#  NEW HELPER: For updating existing workflows
def put_data(endpoint, payload):
    try:
        response = requests.put(f"{API_BASE}/{endpoint}", json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error updating data: {e}")
        return None

def delete_data(endpoint):
    try:
        response = requests.delete(f"{API_BASE}/{endpoint}")
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Error deleting data: {e}")
        return False

# --- UI NAVIGATION ---
st.sidebar.title("🤖 Aeon Factory")
page = st.sidebar.radio("Navigation", ["Dashboard", "Agent Builder", "Workflow Manager", "Playground", "Execution Chat"])

# ==========================================
# 📊 PAGE 1: DASHBOARD
# ==========================================
if page == "Dashboard":
    st.title("System Dashboard")
    stats = fetch_data("system/stats")
    
    if stats:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Agents", stats.get("total_agents", 0))
        col2.metric("Total Workflows", stats.get("total_workflows", 0))
        col3.metric("System Status", stats.get("status", "Unknown"))
        
        st.success("API Backend is successfully connected and healthy.")

# ==========================================
# 🛠️ PAGE 2: AGENT BUILDER (CRUD)
# ==========================================
elif page == "Agent Builder":
    st.title("Agent Builder")
    
    # List Existing Agents
    st.subheader("Current Agents")
    agents = fetch_data("agents")
    if agents:
        df = pd.DataFrame(agents)
        st.dataframe(df[["id", "name", "routing_description", "authorized_tools"]], use_container_width=True)
    
    st.divider()
    
    # Create Agent Form
    st.subheader("Create New Agent")
    with st.form("create_agent_form"):
        col1, col2 = st.columns(2)
        a_id = col1.text_input("Agent ID (e.g., AGT_PORTFOLIO)")
        a_name = col2.text_input("Agent Name (e.g., Portfolio Manager)")
        
        a_desc = st.text_input("Routing Description (Used by Supervisor)")
        a_persona = st.text_area("System Persona / Prompt")
        
        # Fetch available tools dynamically
        available_tools = [t["name"] for t in fetch_data("tools")]
        a_tools = st.multiselect("Authorized Tools", available_tools)
        
        submitted = st.form_submit_button("Deploy Agent")
        if submitted:
            payload = {
                "id": a_id,
                "name": a_name,
                "routing_description": a_desc,
                "persona": a_persona,
                "authorized_tools": a_tools
            }
            if post_data("agents", payload):
                st.success(f"Agent {a_name} deployed successfully!")
                st.rerun()

# ==========================================
# ⛓️ PAGE 3: WORKFLOW MANAGER
# ==========================================
elif page == "Workflow Manager":
    st.title("Workflow Manager")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Create Workflow")
        with st.form("create_wf_form"):
            wf_id = st.text_input("Workflow ID (e.g., WF_ONBOARDING)")
            wf_name = st.text_input("Workflow Name")
            wf_desc = st.text_input("Description")

            # CUSTOM WORKFLOW PROMPTS ADDITIONS: Optional prompts on creation
            st.markdown("**(Optional) Override Global Prompts**")
            wf_sup_prompt = st.text_area("Custom Supervisor Routing Rules", help="Leave blank to use global defaults.")
            wf_syn_prompt = st.text_area("Custom Synthesizer Persona", help="Leave blank to use global defaults.")
            
            if st.form_submit_button("Create Workflow"):
                payload = {"id": wf_id, "name": wf_name, "description": wf_desc}
                if wf_sup_prompt.strip():
                    payload["supervisor_prompt"] = wf_sup_prompt.strip()
                if wf_syn_prompt.strip():
                    payload["synthesizer_prompt"] = wf_syn_prompt.strip()
                    
                if post_data("workflows", payload):
                    st.success("Workflow created!")
                    st.rerun()

    with col2:
        st.subheader("Map Agents to Workflows")
        workflows = fetch_data("workflows")
        agents = fetch_data("agents")
        
        if workflows and agents:
            selected_wf = st.selectbox("Select Workflow", workflows, format_func=lambda x: x["name"])
            selected_agent = st.selectbox("Select Agent to Map", agents, format_func=lambda x: x["name"])
            
            if st.button("Map Agent"):
                if post_data(f"workflows/{selected_wf['id']}/map", {"agent_id": selected_agent["id"]}):
                    st.success("Agent successfully mapped!")
                    st.rerun()

    st.divider()
    st.subheader("Workflow Configurations")
    for wf in workflows:
        with st.expander(f"⚙️ {wf['name']} ({wf['id']})"):
            st.write(f"**Description:** {wf['description']}")

            # 🟢 EPIC 1 ADDITIONS: Editable Custom Prompts Form
            with st.form(f"update_prompts_{wf['id']}"):
                st.markdown("**🧠 Custom Workflow Prompts**")
                new_sup = st.text_area("Supervisor Routing Rules", value=wf.get("supervisor_prompt", "") or "", height=120)
                new_syn = st.text_area("Synthesizer Persona", value=wf.get("synthesizer_prompt", "") or "", height=120)
                
                if st.form_submit_button("💾 Save Custom Prompts"):
                    update_payload = {
                        "supervisor_prompt": new_sup.strip() if new_sup.strip() else None,
                        "synthesizer_prompt": new_syn.strip() if new_syn.strip() else None
                    }
                    if put_data(f"workflows/{wf['id']}", update_payload):
                        st.success("Prompts updated successfully!")
                        st.rerun()

            st.write("---")
            
            wf_agents = fetch_data(f"workflows/{wf['id']}/agents")
            if wf_agents:
                st.write("**Mapped Agents:**")
                for a in wf_agents:
                    col_a, col_b = st.columns([4, 1])
                    col_a.write(f"- {a['name']}")
                    if col_b.button("Unmap", key=f"unmap_{wf['id']}_{a['id']}"):
                        if delete_data(f"workflows/{wf['id']}/agents/{a['id']}"):
                            st.rerun()
            else:
                st.warning("No agents mapped to this workflow yet.")

# ==========================================
# 🧪 PAGE 4: AGENT PLAYGROUND
# ==========================================
elif page == "Playground":
    st.title("Agent Playground")
    st.markdown("Test an agent's persona and tool calling without saving to the DB.")
    
    available_tools = [t["name"] for t in fetch_data("tools")]
    existing_agents = fetch_data("agents")

    preset_labels = ["Custom (No Preset)"]
    preset_labels += [f"Example: {name}" for name in PLAYGROUND_EXAMPLE_AGENTS.keys()]
    if existing_agents:
        preset_labels += [f"Saved Agent: {a['name']} ({a['id']})" for a in existing_agents]

    selected_preset = st.selectbox("Quick Start Preset", preset_labels)

    default_persona = "You are a helpful assistant. Use tools if necessary."
    default_tools = []
    default_test_prompt = "Ask a question..."

    if selected_preset.startswith("Example: "):
        example_name = selected_preset.replace("Example: ", "", 1)
        selected_example = PLAYGROUND_EXAMPLE_AGENTS.get(example_name, {})
        default_persona = selected_example.get("persona", default_persona)
        default_tools = selected_example.get("tools", [])
        default_test_prompt = selected_example.get("prompt", default_test_prompt)
    elif selected_preset.startswith("Saved Agent: "):
        selected_saved_agent = next(
            (a for a in existing_agents if selected_preset.endswith(f"({a['id']})")),
            None
        )
        if selected_saved_agent:
            default_persona = selected_saved_agent.get("persona", default_persona)
            default_tools = selected_saved_agent.get("authorized_tools", [])
            default_test_prompt = suggest_prompt_from_tools(default_tools)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        test_persona = st.text_area("Inject Persona Prompt", height=200, value=default_persona)
        valid_default_tools = [t for t in default_tools if t in available_tools]
        test_tools = st.multiselect("Give access to tools:", available_tools, default=valid_default_tools)
        
    with col2:
        test_prompt = st.text_area("User Message", height=200, value=default_test_prompt, placeholder="Ask a question...")
        if st.button("Run Test", type="primary"):
            with st.spinner("Executing stateless agent..."):
                payload = {"persona": test_persona, "prompt": test_prompt, "tools": test_tools}
                result = post_data("playground", payload)
                if result:
                    st.success("Execution Complete")
                    st.info(result.get("final_answer", "No answer generated."))

# ==========================================
# 💬 PAGE 5: EXECUTION CHAT
# ==========================================

### Change it entirely to accomodate SSE 
elif page == "Execution Chat":
    st.title("LangGraph Chat Execution")

    workflows = fetch_data("workflows")
    if workflows:
        AUTO_OPTION = {"id": "__auto__", "name": "🤖 Auto-route (Router decides)"}
        wf_options = [AUTO_OPTION] + list(workflows)

        col1, col2 = st.columns([1, 1])
        selected_wf = col1.selectbox(
            "Select Active Workflow",
            wf_options,
            format_func=lambda x: x["name"],
        )
        session_id = col2.text_input(
            "Session ID (Leave blank to generate new, or enter an old one to resume)",
            value="thread-test-1",
        )

        col3, col4 = st.columns([1, 1])
        user_id = col3.text_input("User ID (optional)", value="")
        tenant_id = col4.text_input("Tenant ID", value="default")

        st.divider()

        # Load History
        if session_id:
            history = fetch_data(f"sessions/{session_id}/history")
            if history and history.get("messages"):
                for msg in history["messages"]:
                    with st.chat_message(msg["role"]):
                        if msg["role"] == "ai":
                            render_response_with_widgets(st, msg["content"], streaming=False)
                        else:
                            st.write(msg["content"])

        # ─────────────────────────────────────────────────────────────────────
        # Suggestion-click handler: if a previous suggestion was clicked,
        # treat it as the next prompt. Stored in session_state.
        # ─────────────────────────────────────────────────────────────────────
        prompt = None
        if "pending_suggestion" in st.session_state and st.session_state.pending_suggestion:
            prompt = st.session_state.pending_suggestion
            st.session_state.pending_suggestion = None  # consume it

        # Normal chat input
        chat_input_value = st.chat_input("Send a message to the workflow...")
        if chat_input_value:
            prompt = chat_input_value

        if prompt:
            with st.chat_message("user"):
                st.write(prompt)

            is_auto = selected_wf["id"] == "__auto__"

            payload = {
                "prompt": prompt,
                "session_id": session_id if session_id else None,
            }
            if not is_auto:
                payload["workflow_id"] = selected_wf["id"]
            if user_id.strip():
                payload["user_id"] = user_id.strip()
            if tenant_id.strip():
                payload["tenant_id"] = tenant_id.strip()

            with st.chat_message("ai"):
                status_text       = st.empty()
                message_placeholder = st.empty()
                sources_container = st.container()
                suggestions_container = st.container()

                full_response = ""
                execution_trace = []
                collected_sources = []  # list of dicts from tool_result events
                suggestions = []

                try:
                    response = requests.post(
                        "http://127.0.0.1:8000/api/chat",
                        json=payload,
                        stream=True,
                    )
                    response.raise_for_status()

                    for line in response.iter_lines():
                        if not line:
                            continue
                        decoded_line = line.decode("utf-8")
                        if not decoded_line.startswith("data: "):
                            continue

                        try:
                            event_data = json.loads(decoded_line[6:])
                        except json.JSONDecodeError:
                            continue
                        event_type = event_data.get("type")

                        # ── Ephemeral status updates ──────────────────────────
                        if event_type == "status":
                            status_text.caption(f"🔄 {event_data.get('message', '')}")

                        elif event_type == "node_enter":
                            status_text.caption(f"🟢 {event_data.get('message', '')}")

                        elif event_type == "tool_call":
                            status_text.caption(f"🛠️  {event_data.get('message', '')}")

                        # ── Capture structured tool result for the sources panel
                        elif event_type == "tool_result":
                            collected_sources.append({
                                "sequence": event_data.get("sequence"),
                                "source": event_data.get("source", ""),
                                "tool": event_data.get("tool", ""),
                                "content": event_data.get("content", ""),
                            })

                        # ── Token streaming from synthesizer ──────────────────
                        elif event_type == "token":
                            full_response += event_data.get("chunk", "")
                            render_response_with_widgets(
                                message_placeholder, full_response, streaming=True
                            )

                        # ── Final answer ──────────────────────────────────────
                        elif event_type == "complete":
                            final_text = event_data.get("final_answer") or full_response
                            suggestions = render_response_with_widgets(
                                message_placeholder, final_text, streaming=False
                            )
                            status_text.empty()
                            execution_trace = event_data.get("trace", [])

                        elif event_type == "error":
                            st.error(f"Backend Error: {event_data.get('message')}")
                            status_text.empty()

                    # ── After stream closes: render sources & suggestions ────
                    if collected_sources:
                        with sources_container:
                            render_sources_panel(collected_sources)

                    if suggestions:
                        with suggestions_container:
                            st.markdown("**💡 Suggested follow-ups:**")
                            sug_cols = st.columns(min(len(suggestions), 3))
                            for i, sug in enumerate(suggestions[:3]):
                                with sug_cols[i]:
                                    # A unique key per (session, turn, index) avoids reruns clobbering
                                    btn_key = f"sug_{session_id}_{len(history.get('messages', []) if history else [])}_{i}"
                                    if st.button(sug, key=btn_key, use_container_width=True):
                                        st.session_state.pending_suggestion = sug
                                        st.rerun()

                    if execution_trace:
                        with st.expander("🔍 Graph Trace / Router Plan"):
                            st.markdown("**Execution Trace**")
                            st.json(execution_trace)

                except Exception as e:
                    st.error(f"Failed to communicate with the backend stream: {e}")

    else:
        st.warning("Please create a workflow in the Workflow Manager first.")