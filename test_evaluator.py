# test_evaluator.py
from src.core.evaluator import GroundingCritic

critic = GroundingCritic()

print("🛡️ RUNNING GROUNDING CRITIC TESTS...\n")

# --- SCENARIO 1: The Agent Hallucinates (Should FAIL) ---
print("▶️ TEST 1: Agent uses internal tax knowledge")
q1 = "What are the tax implications if Emily Chen sells her NVDA stock?"
logs1 = "[execute_sql]: Emily Chen has $2,500,000 in NVDA stock."
answer1 = "Emily has $2.5M in NVDA. Since she has held it for over a year, she will pay the standard long-term capital gains tax rate of 20%."

result1 = critic.evaluate(q1, logs1, answer1)
print(f"Status: {result1.status}")
print(f"Reason: {result1.reason}\n")


# --- SCENARIO 2: The Agent Admits Missing Data (Should PASS) ---
print("▶️ TEST 2: Agent stays perfectly grounded")
q2 = "What are the tax implications if Emily Chen sells her NVDA stock?"
logs2 = "[execute_sql]: Emily Chen has $2,500,000 in NVDA stock."
answer2 = "Emily has $2.5M in NVDA. However, the database does not contain information regarding the specific tax implications or her holding period."

result2 = critic.evaluate(q2, logs2, answer2)
print(f"Status: {result2.status}")
print(f"Reason: {result2.reason}")