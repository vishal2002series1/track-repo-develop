# test_fabricator.py
from src.agents.fabricator import AgentFabricator

fabricator = AgentFabricator()

print("🔨 RUNNING AGENT FABRICATOR TEST...\n")

test_query = "What are the tax implications if Emily Chen sells her NVDA stock?"

# --- SCENARIO 1: First Draft ---
print("▶️ TEST 1: Fabricating 'Version 1' Agent")
blueprint1 = fabricator.fabricate(test_query)

print(f"Name: {blueprint1.name}")
print(f"Routing: {blueprint1.routing_description}")
print(f"Tools: {blueprint1.authorized_tools}")
print(f"Persona:\n{blueprint1.persona}\n")
print("-" * 50 + "\n")


# --- SCENARIO 2: The Critic gave it a FAIL, Fabricator must fix it ---
print("▶️ TEST 2: Fabricating 'Version 2' (Adapting to Critic Feedback)")
# We inject the exact reason your critic gave us in the previous step!
critic_feedback = """Status: FAIL
Reason: The Agent fabricated the holding period and applied a standard IRS 20% tax rate from internal knowledge. 
It must explicitly admit missing data rather than guessing."""

blueprint2 = fabricator.fabricate(test_query, critic_feedback=critic_feedback)

print(f"Name: {blueprint2.name}")
print(f"Tools: {blueprint2.authorized_tools}")
print(f"Persona (Notice how the prompt tightened up):\n{blueprint2.persona}")