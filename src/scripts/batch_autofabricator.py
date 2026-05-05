# src/scripts/batch_autofabricator.py
import os
import json
import sys
from dotenv import load_dotenv

# 🟢 LOAD ENVIRONMENT VARIABLES
load_dotenv()

# 🟢 BULLETPROOF PATHING: OS-Agnostic and Absolute
# 1. Get the absolute path of the directory this script lives in
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Safely navigate up two levels to the Project Root (AI-agent-factory)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# 3. Add root to system path so absolute imports always work
sys.path.append(PROJECT_ROOT)

from src.agents.fabricator import DomainFabricator
from src.agents.config import AgentConfig, registry_manager

os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_TRACE"] = ""

def run_domain_fabrication(target_file="workflow_2_portfolio_performance.json"):
    print("🚀 Starting Bounded Domain Fabrication...\n")
    
    # 🟢 OS-AGNOSTIC FILE TARGETING
    workflows_dir = os.path.join(PROJECT_ROOT, 'workflows')
    wf_path = os.path.join(workflows_dir, target_file)
    
    # 🛑 DEFENSIVE GUARDRAIL: Check if file exists before trying to open it
    if not os.path.exists(wf_path):
        print(f"\n❌ CRITICAL PATH ERROR: The file does not exist at:\n{wf_path}")
        print("\n💡 HOW TO FIX THIS:")
        print("1. Check your .gitignore file. Make sure the 'workflows' folder is not being ignored.")
        print("2. Make sure you actually pushed the JSON files from your Mac to GitHub.")
        print("3. Alternatively, manually copy the 'workflows' folder from your Mac to this Windows machine.")
        sys.exit(1)
    
    with open(wf_path, 'r') as f:
        wf_data = json.load(f)
        
    wf_name = wf_data.get("workflow_name", "Unknown")
    description = wf_data.get("description", "")
    data_sources = wf_data.get("required_data_sources", [])
    user_mandatory_agents = wf_data.get("user_mandatory_agents", [])
    
    print(f"📦 Analyzing Domain: {wf_name}")
    print(f"📌 Mandatory Agents: {user_mandatory_agents}")
    
    existing_info = []
    for name, config in registry_manager.agents.items():
        existing_info.append(f"- {name}: {config.routing_description}")
    existing_agents_str = "\n".join(existing_info) if existing_info else "None"
    
    print(f"🔍 Found {len(registry_manager.agents)} existing agents in global registry.")
    
    fabricator = DomainFabricator()
    output = fabricator.fabricate(wf_name, description, data_sources, user_mandatory_agents, existing_agents_str)
    
    if output.domain_agents:
        print(f"\n🎉 Fabricated {len(output.domain_agents)} NEW Domain Agents:")
        for blueprint in output.domain_agents:
            print(f"   🟢 {blueprint.name}")
            config = AgentConfig(
                name=blueprint.name,
                routing_description=blueprint.routing_description,
                persona=blueprint.persona,
                authorized_tools=blueprint.authorized_tools,
                # 🟢 DYNAMIC MODEL ID: Safe fallback if .env is missing
                model_id=os.getenv("MODEL_ID", "gpt-5.4") 
            )
            registry_manager.save_agent(config)
    else:
        print("\n✅ No new agents required. Existing registry is sufficient.")

    final_roster = output.final_resolved_agents
    print(f"\n📋 Final Resolved Agent Roster for {wf_name}:")
    for agent in final_roster:
        print(f"   🔹 {agent}")
        
    wf_data["final_resolved_agents"] = final_roster
    
    with open(wf_path, 'w') as f:
        json.dump(wf_data, f, indent=2)
        
    print(f"\n💾 Successfully bound {len(final_roster)} agents to {wf_path}")

if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else "workflow_2_portfolio_performance.json"
    run_domain_fabrication(target_file)