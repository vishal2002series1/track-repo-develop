# streamlit_app.py
import streamlit as st
import requests
import pandas as pd

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
elif page == "Execution Chat":
    st.title("LangGraph Chat Execution")
    
    workflows = fetch_data("workflows")
    if workflows:
        col1, col2 = st.columns([1, 1])
        selected_wf = col1.selectbox("Select Active Workflow", workflows, format_func=lambda x: x["name"])
        session_id = col2.text_input("Session ID (Leave blank to generate new, or enter an old one to resume)", value="thread-test-1")
        
        st.divider()
        
        # Load History
        if session_id:
            history = fetch_data(f"sessions/{session_id}/history")
            if history and history.get("messages"):
                for msg in history["messages"]:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])
        
        # Chat Input
        if prompt := st.chat_input("Send a message to the workflow..."):
            with st.chat_message("user"):
                st.write(prompt)
                
            with st.spinner("Executing Workflow Graph..."):
                payload = {
                    "workflow_id": selected_wf["id"],
                    "prompt": prompt,
                    "session_id": session_id if session_id else None
                }
                res = post_data("chat", payload)
                
                if res:
                    with st.chat_message("ai"):
                        st.write(res["final_answer"])
                        with st.expander("View Graph Trace"):
                            st.json(res["execution_trace"])
    else:
        st.warning("Please create a workflow in the Workflow Manager first.")