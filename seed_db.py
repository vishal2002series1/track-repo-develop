import os
import json
import glob
import sys

# 1. Absolute pathing for cross-platform stability
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from src.db.database import engine, SessionLocal, Base
from src.db.models import DomainAgent, Workflow

# 🛑 CRITICAL IMPORT: This automatically creates data_local/agent_registry.json 
# on fresh Windows clones using your new default configuration if the file is missing!
from src.agents.config import registry_manager 

def seed():
    print("⏳ Creating core database tables...")
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    
    REGISTRY_PATH = os.path.join(PROJECT_ROOT, 'data_local', 'agent_registry.json')
    WORKFLOWS_DIR = os.path.join(PROJECT_ROOT, 'workflows')
    
    # --- 1. SEED AGENTS ---
    print(f"\n📥 Synchronizing agents to SQL database...")
    if not os.path.exists(REGISTRY_PATH):
        print(f"❌ CRITICAL ERROR: registry_manager failed to generate {REGISTRY_PATH}")
        return

    with open(REGISTRY_PATH, 'r') as f:
        data = json.load(f)
        
    for agent_id, agent_data in data.items():
        existing = db.query(DomainAgent).filter(DomainAgent.id == agent_id).first()
        if not existing:
            new_agent = DomainAgent(
                id=agent_id,
                name=agent_data.get("name", agent_id),
                routing_description=agent_data.get("routing_description", ""),
                persona=agent_data.get("persona", ""),
                authorized_tools=agent_data.get("authorized_tools", [])
            )
            db.add(new_agent)
            print(f"   ✅ Added Agent to DB: {agent_id}")
        else:
            # Safely update existing agents if personas or descriptions changed
            existing.persona = agent_data.get("persona", existing.persona)
            existing.routing_description = agent_data.get("routing_description", existing.routing_description)
    
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"   ❌ Error committing agents: {e}")

    # --- 2. SEED WORKFLOWS FROM JSON ---
    if os.path.exists(WORKFLOWS_DIR):
        print(f"\n🏗️ Loading Workflows from {WORKFLOWS_DIR}...")
        workflow_files = glob.glob(os.path.join(WORKFLOWS_DIR, "*.json"))
        
        for wf_file in workflow_files:
            try:
                with open(wf_file, 'r') as f:
                    wf_data = json.load(f)
                    
                wf_id = wf_data.get("workflow_id")
                if not wf_id:
                    continue

                existing_wf = db.query(Workflow).filter(Workflow.id == wf_id).first()
                
                if not existing_wf:
                    new_wf = Workflow(
                        id=wf_id,
                        name=wf_data.get("workflow_name", wf_id),
                        description=wf_data.get("description", "")
                    )
                    db.add(new_wf)
                    print(f"   ✅ Added Workflow: {wf_id}")
                    
                    # Map the Agents to the Workflow
                    resolved_agents = wf_data.get("final_resolved_agents", [])
                    for agent_id in resolved_agents:
                        agent = db.query(DomainAgent).filter(DomainAgent.id == agent_id).first()
                        if agent:
                            new_wf.agents.append(agent)
                            print(f"      🔗 Mapped {agent_id} to {wf_id}")
                else:
                    print(f"   ⚠️ Workflow {wf_id} already exists. Skipping.")
                    
                db.commit()
                
            except Exception as e:
                db.rollback()
                print(f"   ❌ Failed to process file {os.path.basename(wf_file)}: {e}")
        
    print("\n🎉 Database Seeding Complete!")
    db.close()

if __name__ == "__main__":
    seed()