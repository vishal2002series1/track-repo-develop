# test_graph.py
from src.agents.graph import aeon_app
from langchain_core.messages import HumanMessage
from langgraph.types import Command
import time

test_cases = [
    {
        "name": "1. EASY (Single Agent)",
        "query": "What is the Target Return (YTD) for the Growth Target policy benchmark?",
        "thread": "test_easy_001"
    },
    {
        "name": "2. MEDIUM (Multi-Table)",
        "query": "Review David Alverez. Does he have any compliance flags, and what are his upcoming life events?",
        "thread": "test_medium_001"
    },
    {
        "name": "3. COMPLEX (Multi-Agent & Math)",
        "query": "For Emily Chen, calculate her portfolio concentration risk. Then, check her recent emails to see if she has expressed anxiety about her portfolio.",
        "thread": "test_complex_001"
    },
    {
        "name": "4. EDGE CASE (Ambiguity -> HITL)",
        "query": "What is the next best action for Robert?",
        "thread": "test_edge_001"
    }
]

print("🚀 RUNNING AEON ENTERPRISE TEST SUITE...\n" + "="*50)

for i, test in enumerate(test_cases):
    print(f"\n\n▶️  TEST {test['name']}")
    print(f"👤 Query: {test['query']}")
    print("-" * 50)
    
    config = {"configurable": {"thread_id": test["thread"]}}
    inputs = {"messages": [HumanMessage(content=test["query"])]}
    
    # Run the graph
    for event in aeon_app.stream(inputs, config=config, stream_mode="values"):
        if "routing_log" in event and event["routing_log"]:
            print(f"🧭 [Routing]: {event['routing_log'][-1]}")
            
    # Check for Human-in-the-Loop Interrupt
    state = aeon_app.get_state(config)
    if state.next:
        print("\n--- 🛑 GRAPH PAUSED FOR HUMAN INPUT ---")
        human_input = input("Provide Clarification: ")
        print("\nResuming workflow...")
        
        for event in aeon_app.stream(Command(resume=human_input), config=config, stream_mode="values"):
            if "routing_log" in event and event["routing_log"]:
                print(f"🧭 [Routing]: {event['routing_log'][-1]}")
    
    # Print the final AI response
    final_state = aeon_app.get_state(config)
    if final_state.values and "messages" in final_state.values:
        final_message = final_state.values["messages"][-1].content
        print(f"\n🤖 [Final Answer]:\n{final_message}")
        
    time.sleep(1) # Brief pause between tests