# test_runtime.py
import uuid
from langchain_core.messages import HumanMessage
from src.agents.graph import build_aeon_graph
from src.agents.config import AEON_AGENT_REGISTRY

print("="*60)
print(f"🚀 BOOTING UP AEON WEALTH ORCHESTRATOR")
print(f"📚 Total Agents Loaded in Memory: {len(AEON_AGENT_REGISTRY)}")
print("="*60)

# Build the main app (This automatically reads your new JSON registry!)
app = build_aeon_graph()

# A slightly varied query to prove the Supervisor understands intent
query = "Analyze the tax implications and concentration risk if Emily Chen liquidates her NVDA position."
query = "I have an annual review with the Smith family tomorrow. Please draft a meeting agenda, summarize their recent communication sentiment, and compare their portfolio performance against the firm benchmark."
query = "Which clients have had a major life event recently but no updated financial plan or scheduled review meeting?"
print(f"\n🗣️ USER QUERY: {query}\n")
print("🔄 ORCHESTRATOR ROUTING JOURNEY:")

# Run the graph
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
state = app.invoke({"messages": [HumanMessage(content=query)]}, config=config)

# Print out exactly which agents the Supervisor chose to hire for this job
for step in state.get("routing_log", []):
    print(f"  ➔ {step}")

print("\n" + "="*60)
print("✅ FINAL AGENT ANSWER:")
print("="*60)
print(state["messages"][-1].content)