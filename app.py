# app.py
import asyncio

# Force Python to use the standard event loop instead of uvloop (Mac/Linux fix)
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import json
import os
import subprocess
import glob
from langchain_core.messages import HumanMessage
from src.workflows.wf_002 import build_WF_002_graph
from src.workflows.wf_003 import build_WF_003_graph

# Disable gRPC fork support to prevent threading crashes with Arize/Bedrock in Streamlit
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_TRACE"] = ""

# --- Configuration & Paths ---
st.set_page_config(page_title="Aeon Agent Factory", page_icon="🏭", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTRY_PATH = os.path.join(BASE_DIR, 'data_local', 'agent_registry.json')
PROMPT_PATH = os.path.join(BASE_DIR, 'data_local', 'prompt_library.json')
WORKFLOWS_DIR = os.path.join(BASE_DIR, 'workflows')
SCRIPTS_DIR = os.path.join(BASE_DIR, 'src', 'scripts')

# --- Helper Functions ---
def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}

def save_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def run_script(script_name, args=None):
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    command = ['python', script_path]
    if args:
        command.append(args)
        
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        st.success(f"Successfully ran {script_name}")
        st.code(result.stdout, language="bash")
    except subprocess.CalledProcessError as e:
        st.error(f"Error running {script_name}")
        st.code(e.stdout + "\n" + e.stderr, language="bash")

# --- Sidebar Navigation ---
st.sidebar.title("⚙️ Control Center")
page = st.sidebar.radio("Navigate", ["🏭 Agent Registry", "📚 Prompt Library", "📦 Workflow Manager", "💬 Associate Workbench"])

# ==========================================
# PAGE 1: AGENT REGISTRY
# ==========================================
if page == "🏭 Agent Registry":
    st.title("🏭 Global Agent Registry")
    st.markdown("View all active domain agents currently available in the factory pool.")
    
    registry_data = load_json(REGISTRY_PATH)
    
    if not registry_data:
        st.warning("Registry is empty.")
    else:
        st.metric("Total Agents Available", len(registry_data))
        
        for agent_name, config in registry_data.items():
            with st.expander(f"🟢 {agent_name}"):
                st.write(f"**Routing Description:** {config.get('routing_description')}")
                st.write(f"**Authorized Tools:** `{', '.join(config.get('authorized_tools', []))}`")
                st.text_area("Persona / Instructions", config.get('persona', ''), height=150, disabled=True, key=f"persona_{agent_name}")

# ==========================================
# PAGE 2: PROMPT LIBRARY
# ==========================================
elif page == "📚 Prompt Library":
    st.title("📚 Prompt Library")
    st.markdown("Edit the core system prompts. Changes are saved instantly and affect all future graph executions.")
    
    prompt_data = load_json(PROMPT_PATH)
    updated_prompts = {}
    
    for prompt_key, prompt_text in prompt_data.items():
        st.subheader(prompt_key.replace("_", " ").title())
        updated_prompts[prompt_key] = st.text_area(f"Edit {prompt_key}", value=prompt_text, height=200, key=prompt_key)
        st.markdown("---")
        
    if st.button("💾 Save All Prompts", type="primary"):
        save_json(PROMPT_PATH, updated_prompts)
        st.success("Prompts saved successfully! The Orchestrator and Synthesizer will now use these rules.")

# ==========================================
# PAGE 3: WORKFLOW MANAGER
# ==========================================
elif page == "📦 Workflow Manager":
    st.title("📦 Workflow Manager & Compiler")
    st.markdown("Fabricate missing agents and compile bounded workflows.")
    
    workflow_files = [os.path.basename(f) for f in glob.glob(os.path.join(WORKFLOWS_DIR, "*.json"))]
    selected_wf = st.selectbox("Select Workflow Requirements File", workflow_files)
    
    if selected_wf:
        wf_data = load_json(os.path.join(WORKFLOWS_DIR, selected_wf))
        st.write(f"**Workflow ID:** `{wf_data.get('workflow_id')}`")
        st.write(f"**Description:** {wf_data.get('description')}")
        
        st.write("**Mandatory Agents:**")
        st.code(wf_data.get('user_mandatory_agents', []))
        
        st.write("**Final Resolved Roster (Bounded State):**")
        st.code(wf_data.get('final_resolved_agents', []))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("Step 1: Read requirements and build missing agents.")
            if st.button("🚀 Run Fabricator", use_container_width=True):
                with st.spinner("Fabricating Domain Agents..."):
                    run_script("batch_autofabricator.py", selected_wf) # <--- Passes the filename!
                    
        with col2:
            st.success("Step 2: Compile the strictly bounded graph.")
            if st.button("🔨 Compile Bounded Graph", use_container_width=True):
                with st.spinner("Compiling LangGraph..."):
                    run_script("workflow_compiler.py", selected_wf) # <--- Passes the filename!

# ==========================================
# PAGE 4: ASSOCIATE WORKBENCH (CHAT)
# ==========================================
elif page == "💬 Associate Workbench":
    st.title("💬 Associate Workbench")
    st.markdown("Interact with your bounded Super-Workflows in real-time.")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Session Settings")
        session_id = st.text_input("Session ID", value="streamlit_demo_01")
        
        # 🔀 NEW: Workflow Selector Dropdown
        active_workflow = st.selectbox(
            "Select Active Workflow", 
            ["WF_002: Portfolio & Performance", "WF_003: Meeting Strategy"]
        )
        
        st.caption("Powered by LangGraph & Azure-Ready SQLite Checkpointing")
        
        if st.button("🗑️ Clear UI Chat History"):
            st.session_state.messages = []
            st.rerun()

    with col2:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        chat_container = st.container(height=600, border=False)

        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    if "trace" in message:
                        with st.expander("🔍 Agent Thought Trace (X-Ray)"):
                            for step in message["trace"]:
                                st.write(f"- `{step}`")
                    st.markdown(message["content"])

        if prompt := st.chat_input("Ask Aeon AI..."):
            
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
            
            st.session_state.messages.append({"role": "user", "content": prompt})

            with chat_container:
                with st.chat_message("assistant"):
                    status_text = st.empty()
                    thought_trace = []
                    
                    # 🔀 NEW: Dynamically load the correct graph
                    if "WF_002" in active_workflow:
                        app = build_WF_002_graph()
                    else:
                        app = build_WF_003_graph()
                        
                    config = {"configurable": {"thread_id": session_id}}
                    final_answer = ""
                    
                    # Stream the LangGraph execution
                    try:
                        for event in app.stream({"messages": [HumanMessage(content=prompt)]}, config=config, stream_mode="updates"):
                            # 1. Ironclad safety check for network blips
                            if not event:
                                continue
                                
                            for node_name, node_data in event.items():
                                status_text.markdown(f"⏳ *Agent Active:* `{node_name}` *is processing...*")
                                thought_trace.append(node_name)
                                
                                # 2. Ironclad safety check for NoneType payloads
                                if node_data is not None:
                                    messages = node_data.get("messages")
                                    # If messages is a valid list and has items
                                    if messages is not None and isinstance(messages, list) and len(messages) > 0:
                                        # Only grab the final answer if it's an AIMessage (not empty)
                                        if hasattr(messages[-1], 'content') and messages[-1].content:
                                            final_answer = messages[-1].content
                        
                        status_text.empty()
                        
                        # Display the X-Ray trace and the final answer
                        if thought_trace:
                            with st.expander("🔍 Agent Thought Trace (X-Ray)"):
                                for step in thought_trace:
                                    st.write(f"- Routed to: `{step}`")
                                    
                        st.markdown(final_answer)
                        
                        # Save both the answer and the trace to session state
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": final_answer,
                            "trace": thought_trace
                        })
                        
                    except Exception as e:
                        status_text.empty()
                        import traceback
                        st.error(f"Execution Error: {e}")
                        print(f"Detailed Traceback:\n{traceback.format_exc()}")